#pragma once
// basis_grow_mpi.hpp
// ==================
// Grow-by-closure construction of a distributed ice-rule basis, shared by
// bench_apply_inmem_grow_mpi and diag_DOQSI_grow_mpi.
//
// Instead of enumerating the whole constrained basis with the work-stealing
// constraint-tree search, the basis is *grown* from a seed:
//
//   1. On rank 0 only, run the plain serial depth-first constraint-tree search
//      just long enough to collect a modest seed set of complete valid ice
//      states (>= seeds_per_rank x world_size). Remaining incomplete DFS nodes
//      are discarded -- only a handful of guaranteed-valid states are needed to
//      nucleate the growth.
//   2. Scatter those seeds to their hash-owning ranks (the mpi_context
//      state->rank hash), so each rank starts with the slice it will own.
//   3. Grow to a fixed point, mirroring the off-diagonal apply's communication
//      pattern: each rank applies every off-diagonal term of H to its frontier
//      of owned states, routes each produced neighbour to its hash-owner, and
//      inserts any genuinely new state into a locally-sorted std::set. Newly
//      inserted states form the next round's frontier. The loop ends when no
//      rank discovers a new state anywhere. Crucially the frontier is expanded
//      and *redistributed on the fly* in bounded chunks through a nonblocking
//      MPI_Ialltoall/MPI_Ialltoallv pipeline rather than materialising the whole
//      round's neighbour list and shipping it in one shot -- at ~1 TB over ~500
//      ranks a single per-round exchange buffer (frontier x #offdiag) dwarfs the
//      basis itself and OOMs, and any per-rank imbalance is fatal.
//   4. The discovery structure is a *sorted std::vector* kept deduplicated by
//      per-round merge, not a node-based std::set: at the target scale a
//      red-black tree costs ~3-4x the raw slice (three pointers + colour per
//      16-byte Uint128), and on memory-capped runs that overhead, not the comm
//      transients, is what bounds the slice that fits. Because the discovery
//      vector is already the sorted basis slice, it is returned straight to the
//      caller -- no separate drain step and no second full-size copy.
//
// IMPORTANT -- why a --sector is mandatory for callers. Growth reconstructs
// only the H-connected closure of the seeds. Ring exchange conserves the
// per-sublattice polarisation, so the constrained manifold splits into disjoint
// sectors; a seed set that does not pin a sector samples an ill-defined,
// non-reproducible union of partial sectors. Callers must therefore require a
// non-empty sector and grow within it (where the closure is a well-defined
// dynamical subspace). No trimming is applied: the grown basis is already the
// exact closure, so remove_null_states is at best a no-op and, for a
// Hamiltonian whose off-diagonal term list is not closed under conjugation,
// would drop states that are still images of kept states and break the apply.

#include "pyro_tree.hpp"       // lat_container(_with_sector), vtree_node_t, cust_stack, permute
#include "operator_mpi.hpp"    // ZBasisBase, SymbolicOpSum
#include "mpi_context.hpp"     // MPIHashContext, get_mpi_type
#include "logging.hpp"

#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>

namespace projED {
namespace basis_grow {

using state_t = ZBasisBase::state_t;


// Step 1: seed generation (rank 0 only).
// Run the plain serial DFS over the (already permuted) constraint tree, exactly
// as pyro_vtree::build_state_tree does, but stop as soon as `target` complete
// states have been emitted. Whatever partial nodes remain on the stack are
// dropped. Emitted states are un-permuted with `perm` so they live in physical
// coordinates, matching an H built from the un-permuted lattice (this is what
// the MPI searcher's shard.push(permute(...)) does per leaf).
template <typename LatC>
std::vector<state_t> generate_seeds_dfs(LatC& latc, const lattice& lat,
                                        const std::vector<size_t>& perm,
                                        size_t target)
{
    std::vector<state_t> seeds;
    seeds.reserve(target);

    const unsigned n_spins = static_cast<unsigned>(lat.spins.size());

    lat_container::cust_stack stack;
    stack.push(vtree_node_t{state_t(0), 0u, 0u});  // root: no spins fixed yet

    while (!stack.empty() && seeds.size() < target) {
        if (stack.top().curr_spin == n_spins) {
            seeds.push_back(permute(stack.top().state_thus_far, perm));
            stack.pop();
        } else {
            // Pops the top node and pushes its (0/1) children that remain
            // consistent with the per-tetrahedron ice constraints (and sector).
            latc.fork_state(stack);
        }
    }
    return seeds;
}


// Step 2: scatter the rank-0 seed set to hash-owning ranks. Rank 0 bucket-sorts
// the seeds by destination rank, then a Scatter of the counts followed by a
// Scatterv of the states hands each rank its slice.
inline std::vector<state_t> scatter_seeds(const std::vector<state_t>& seeds,
                                          const MPIHashContext& ctx)
{
    const int N = ctx.world_size;
    std::vector<int> send_counts(N, 0), send_displs(N, 0);
    std::vector<state_t> send_buf;

    if (ctx.my_rank == 0) {
        for (const auto& s : seeds) send_counts[ctx.rank_of_state(s)]++;
        for (int r = 1; r < N; ++r)
            send_displs[r] = send_displs[r-1] + send_counts[r-1];
        send_buf.resize(seeds.size());
        std::vector<int> cur(send_displs);
        for (const auto& s : seeds)
            send_buf[cur[ctx.rank_of_state(s)]++] = s;
    }

    int my_count = 0;
    MPI_Scatter(send_counts.data(), 1, get_mpi_type<int>(),
                &my_count, 1, get_mpi_type<int>(), 0, MPI_COMM_WORLD);

    std::vector<state_t> mine(my_count);
    MPI_Scatterv(send_buf.data(), send_counts.data(), send_displs.data(),
                 get_mpi_type<state_t>(),
                 mine.data(), my_count, get_mpi_type<state_t>(),
                 0, MPI_COMM_WORLD);
    return mine;
}


// Step 3: grow the basis to its Hamiltonian-closure fixed point.
// Frontier-driven distributed BFS over the graph whose edges are the
// off-diagonal terms of H. Every round each rank turns its current frontier of
// owned states into neighbours (H * state), routes them to their hash-owners
// (same routing as the apply), and folds the received states into `known`;
// states not already present become the next frontier. Terminates when every
// rank's frontier is empty (a collective MAX-reduction of the per-rank chunk
// count keeps every rank in lockstep and detects the fixed point).
//
// `known` is a *sorted std::vector*, not a std::set: membership is a binary
// search and incorporation is a single merge of the round's new states at the
// round boundary. This trades the set's continuous ~3-4x red-black-tree node
// overhead (fatal on memory-capped runs, since the owned slice is ~1x the raw
// states) for a cheap, transient per-round merge. Within a round the new
// states are staged in a second sorted vector `round_new` (received chunks are
// sorted/uniqued and filtered against both `known` and `round_new` so cross-
// chunk duplicates are caught), then merged into `known` once and handed to the
// next round as its frontier.
//
// The frontier is *not* expanded into one big per-round send buffer -- at the
// target scale (~1 TB / ~500 ranks) `frontier x #offdiag` states dwarfs the
// basis and OOMs, and per-rank imbalance in that transient is fatal. Instead the
// frontier is swept in bounded chunks of `chunk_frontier` states, each chunk
// redistributed on the fly through a depth-3 nonblocking pipeline:
//   stage A  produce(chunk c):    expand+bucket the chunk, post MPI_Ialltoall of
//                                 the send counts;
//   stage B  launch_data(c-1):    wait c-1's counts, post MPI_Ialltoallv of the
//                                 states;
//   stage C  harvest(c-2):        wait c-2's states, insert them into `known`.
// With three stages in flight the counts wait (A->B) and the states wait (B->C)
// are both on collectives posted a full iteration earlier, so their latency
// overlaps the next chunk's neighbour generation -- important on Otus where the
// per-collective latency, while modest, is paid once per chunk. Peak transient
// send/recv memory is ~3 x chunk_frontier x #offdiag states instead of the whole
// frontier's worth, and the `int`-counted alltoallv can no longer overflow.
//
// Collective correctness: MPI_Ialltoall/MPI_Ialltoallv are collective, so all
// ranks must post the same sequence. The per-round chunk count is the global MAX
// of the local counts; a rank that runs out of its own frontier keeps posting
// empty (zero-send) chunks -- it still *receives* neighbours other ranks route
// to it, which is exactly the on-the-fly redistribution we want.
//
// Returns the rank-local sorted vector of owned states (already deduplicated);
// n_rounds is set to the number of BFS rounds executed.
template <typename coeff_t>
std::vector<state_t> grow_basis(std::vector<state_t>&& my_seeds,
                                const SymbolicOpSum<coeff_t>& H,
                                const MPIHashContext& ctx,
                                size_t chunk_frontier,
                                size_t reserve_local,
                                size_t& n_rounds)
{
    const int N = ctx.world_size;
    if (chunk_frontier == 0) chunk_frontier = (1u << 16);

    // `known` is kept sorted+unique at all times; it doubles as the frontier
    // seed. my_seeds are already this rank's hash-owned slice (scatter_seeds).
    std::sort(my_seeds.begin(), my_seeds.end());
    my_seeds.erase(std::unique(my_seeds.begin(), my_seeds.end()), my_seeds.end());
    std::vector<state_t> known = my_seeds;
    std::vector<state_t> frontier = my_seeds;
    { std::vector<state_t> tmp; std::swap(tmp, my_seeds); }  // free seeds

    // If the caller supplied an expected owned-slice size, reserve `known` to it
    // up front. Over-reservation is RSS-free (untouched pages are VSZ, not RSS),
    // so this never inflates resident memory, but it ensures the per-round growth
    // below never reallocates across the final size -- the reallocation that
    // crosses B is what transiently doubles `known` and sets the ~2x build-time
    // peak. With the reserve in place the peak is ~1x the slice plus one
    // wavefront. Without it (reserve_local == 0) growth still works, falling back
    // to the 2x floor.
    if (reserve_local > known.capacity()) known.reserve(reserve_local);

    // One in-flight chunk: its hash-bucketed send/recv buffers and the two
    // nonblocking-collective requests it owns. Three slots are enough to keep
    // chunk c's counts exchange, c-1's states exchange and c-2's harvest live
    // simultaneously (slot for chunk c is slot[c % 3]).
    struct Slot {
        std::vector<int> send_counts, recv_counts, send_displs, recv_displs;
        std::vector<state_t> send_buf, recv_buf;
        MPI_Request count_req = MPI_REQUEST_NULL;
        MPI_Request data_req  = MPI_REQUEST_NULL;
        void init(int n) {
            send_counts.assign(n, 0); recv_counts.assign(n, 0);
            send_displs.assign(n, 0); recv_displs.assign(n, 0);
        }
    };
    Slot slot[3];
    for (auto& s : slot) s.init(N);

    std::vector<state_t> round_new;  // sorted+unique, disjoint from `known`

    // Stage A: expand frontier states [lo,hi) into hash-bucketed neighbours and
    // post the (nonblocking) counts exchange. Two cheap applyState passes over
    // the range keep only the compacted send buffer resident -- there is no
    // separate neighbour list -- so peak send memory is chunk x #offdiag states.
    auto produce = [&](Slot& cc, size_t lo, size_t hi) {
        std::fill(cc.send_counts.begin(), cc.send_counts.end(), 0);
        for (size_t i = lo; i < hi; ++i) {
            for (const auto& term : H.off_diag_terms) {
                state_t t = frontier[i];
                if (term.second.applyState(t) != 0)
                    cc.send_counts[ctx.rank_of_state(t)]++;
            }
        }
        cc.send_displs[0] = 0;
        for (int r = 1; r < N; ++r)
            cc.send_displs[r] = cc.send_displs[r-1] + cc.send_counts[r-1];
        size_t send_total = static_cast<size_t>(cc.send_displs[N-1]) + cc.send_counts[N-1];
        cc.send_buf.resize(send_total);
        std::vector<int> cur(cc.send_displs);
        for (size_t i = lo; i < hi; ++i) {
            for (const auto& term : H.off_diag_terms) {
                state_t t = frontier[i];
                if (term.second.applyState(t) != 0)
                    cc.send_buf[cur[ctx.rank_of_state(t)]++] = t;
            }
        }
        MPI_Ialltoall(cc.send_counts.data(), 1, get_mpi_type<int>(),
                      cc.recv_counts.data(), 1, get_mpi_type<int>(),
                      MPI_COMM_WORLD, &cc.count_req);
    };

    // Stage B: counts are back -> size the recv buffer and post the states
    // exchange (nonblocking).
    auto launch_data = [&](Slot& cc) {
        MPI_Wait(&cc.count_req, MPI_STATUS_IGNORE);
        cc.recv_displs[0] = 0;
        for (int r = 1; r < N; ++r)
            cc.recv_displs[r] = cc.recv_displs[r-1] + cc.recv_counts[r-1];
        size_t recv_total =
            std::accumulate(cc.recv_counts.begin(), cc.recv_counts.end(), size_t{0});
        cc.recv_buf.resize(recv_total);
        MPI_Ialltoallv(cc.send_buf.data(), cc.send_counts.data(), cc.send_displs.data(),
                       get_mpi_type<state_t>(),
                       cc.recv_buf.data(), cc.recv_counts.data(), cc.recv_displs.data(),
                       get_mpi_type<state_t>(), MPI_COMM_WORLD, &cc.data_req);
    };

    // Stage C: states are back -> sort/unique this chunk, drop those already in
    // `known` or already staged this round, and merge the genuinely new ones
    // into the sorted `round_new`. Both lookups are binary searches over
    // contiguous memory; `known` is immutable within a round so its search is
    // stable, and `round_new` stays sorted via the inplace_merge below.
    auto harvest = [&](Slot& cc) {
        MPI_Wait(&cc.data_req, MPI_STATUS_IGNORE);
        auto& buf = cc.recv_buf;
        if (buf.empty()) return;
        std::sort(buf.begin(), buf.end());
        buf.erase(std::unique(buf.begin(), buf.end()), buf.end());

        std::vector<state_t> fresh;
        fresh.reserve(buf.size());
        for (const auto& s : buf)
            if (!std::binary_search(known.begin(), known.end(), s) &&
                !std::binary_search(round_new.begin(), round_new.end(), s))
                fresh.push_back(s);
        if (fresh.empty()) return;

        const size_t old = round_new.size();
        round_new.insert(round_new.end(), fresh.begin(), fresh.end());
        std::inplace_merge(round_new.begin(), round_new.begin() + old,
                           round_new.end());
    };

    size_t round = 0;
    while (true) {
        const size_t fsz = frontier.size();
        size_t local_chunks = (fsz + chunk_frontier - 1) / chunk_frontier;
        size_t n_chunks = 0;
        MPI_Allreduce(&local_chunks, &n_chunks, 1, get_mpi_type<size_t>(),
                      MPI_MAX, MPI_COMM_WORLD);
        if (n_chunks == 0) break;   // no rank has any frontier left: fixed point

        round_new.clear();

        // Software-pipelined sweep over the global-max chunk count. Ranks that
        // exhaust their own frontier pad with empty (zero-send) chunks so every
        // rank posts the identical collective sequence; +2 trailing iterations
        // drain the pipeline. Post the new nonblocking ops before the waits so
        // the network can make progress underneath.
        for (size_t it = 0; it < n_chunks + 2; ++it) {
            if (it < n_chunks) {
                size_t lo = std::min(it * chunk_frontier, fsz);
                size_t hi = std::min(lo + chunk_frontier, fsz);
                produce(slot[it % 3], lo, hi);
            }
            if (it >= 1 && (it - 1) < n_chunks)
                launch_data(slot[(it - 1) % 3]);
            if (it >= 2 && (it - 2) < n_chunks)
                harvest(slot[(it - 2) % 3]);
        }

        // Incorporate this round's discoveries into the sorted `known`, then hand
        // them to the next round as the frontier. round_new is sorted and
        // disjoint from known, so we merge the two sorted runs *backwards* in
        // place: grow known to K+R (no reallocation when the reserve above has
        // sized it past the final slice) and fill it from the high end, reading
        // the old known prefix and round_new and never overwriting an unread
        // element (out >= i throughout). This replaces std::inplace_merge, whose
        // temp buffer scales with the whole merged range (~B at the last round)
        // and is the other half of the 2x floor; here the only extra memory is
        // round_new itself (one wavefront), which we then reuse as the frontier.
        const size_t K = known.size();
        const size_t R = round_new.size();
        known.resize(K + R);
        size_t i = K, j = R, out = K + R;
        while (j > 0) {
            if (i > 0 && round_new[j - 1] < known[i - 1])
                known[--out] = known[--i];
            else
                known[--out] = round_new[--j];
        }
        std::swap(frontier, round_new);
        ++round;
    }

    n_rounds = round;
    return known;
}


// Convenience: full seed -> scatter -> grow, returning this rank's
// sorted owned states. The sector must be non-empty (see the file header for
// why); this is asserted only informally -- callers validate and error out with
// a helpful message. Rank 0 performs the DFS; milestones are logged via
// logging::log so both callers report consistently. grow_batch is the frontier
// states per pipelined redistribution chunk (bounds peak comm memory). raw_local
// is set to this
// rank's owned count, raw_global to the summed global dim, n_rounds to the
// number of growth rounds. expected_global, if non-zero, is the anticipated
// global basis dimension; it is used only to reserve each rank's `known` vector
// up front (expected_global / world_size, with margin for hash imbalance) so the
// build-time memory peak stays near 1x the slice instead of the ~2x floor -- an
// over-estimate is harmless (RSS-free), an under-estimate merely reallocates.
template <typename coeff_t>
std::vector<state_t> build_grown_basis_local(
        const lattice& lat, int num_spinon_pairs,
        const std::vector<size_t>& perm,
        const std::vector<int>& sector,
        const SymbolicOpSum<coeff_t>& H,
        const MPIHashContext& ctx,
        size_t seeds_per_rank, size_t grow_batch, size_t expected_global,
        size_t& raw_local, size_t& raw_global, size_t& n_rounds)
{
    const size_t seed_target = seeds_per_rank * static_cast<size_t>(ctx.world_size);

    // --- Step 1: seed generation on rank 0 (serial DFS, stop early) ---------
    std::vector<state_t> seeds;
    if (ctx.my_rank == 0) {
        lat_container_with_sector latc(lat, num_spinon_pairs);
        latc.set_sector(sector);
        seeds = generate_seeds_dfs(latc, lat, perm, seed_target);
        logging::log(logging::INFO)
            << "[grow] rank 0 DFS produced " << seeds.size()
            << " seed states (target " << seed_target << ")\n";
        if (seeds.empty())
            logging::log(logging::INFO)
                << "[grow] WARNING: no seed states found; the grown basis will "
                   "be empty.\n";
    }

    // --- Step 2: scatter seeds to their hash owners -------------------------
    std::vector<state_t> my_seeds = scatter_seeds(seeds, ctx);
    { std::vector<state_t> tmp; std::swap(tmp, seeds); }  // free rank-0 seed set

    // --- Step 3: grow to the Hamiltonian-closure fixed point ----------------
    // The grown structure is already the sorted, deduplicated owned slice, so it
    // is returned directly -- there is no separate drain. Reserve to the expected
    // per-rank share (x1.5 margin for hash imbalance) when a global estimate is
    // given; over-reservation costs only VSZ, not RSS.
    const size_t reserve_local = expected_global
        ? static_cast<size_t>((expected_global / static_cast<double>(ctx.world_size)) * 1.5)
        : 0;
    std::vector<state_t> local = grow_basis(std::move(my_seeds), H, ctx,
                                            grow_batch, reserve_local, n_rounds);

    raw_local = local.size();
    raw_global = 0;
    MPI_Allreduce(&raw_local, &raw_global, 1, get_mpi_type<size_t>(),
                  MPI_SUM, MPI_COMM_WORLD);

    return local;
}


}  // namespace basis_grow
}  // namespace projED

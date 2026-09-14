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
//      rank discovers a new state anywhere.
//   4. Stream the discovery set into the plain std::vector that backs the basis,
//      erasing set nodes as they are copied so the set shrinks while the vector
//      fills (rather than both being fully resident at once).
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
#include <set>
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
// (same routing as the apply), and inserts the received states into `known`;
// states not already present become the next frontier. Terminates when no rank
// inserts a new state anywhere (a collective MAX-reduction keeps every rank in
// lockstep, so all ranks execute the same number of collective rounds).
// Returns the rank-local sorted set of owned states; n_rounds is set to the
// number of exchange rounds executed.
template <typename coeff_t>
std::set<state_t> grow_basis(std::vector<state_t>&& my_seeds,
                             const SymbolicOpSum<coeff_t>& H,
                             const MPIHashContext& ctx,
                             size_t& n_rounds)
{
    const int N = ctx.world_size;

    std::set<state_t> known;
    std::vector<state_t> frontier;
    for (const auto& s : my_seeds)
        if (known.insert(s).second) frontier.push_back(s);
    { std::vector<state_t> tmp; std::swap(tmp, my_seeds); }  // free seeds

    std::vector<int> send_counts(N), recv_counts(N), send_displs(N), recv_displs(N);
    std::vector<state_t> neigh, send_buf, recv_buf, next_frontier;

    size_t round = 0;
    while (true) {
        // (a) expand the frontier: neighbour = op * state for every off-diag op.
        std::fill(send_counts.begin(), send_counts.end(), 0);
        neigh.clear();
        for (const auto& s : frontier) {
            for (const auto& term : H.off_diag_terms) {
                state_t t = s;
                if (term.second.applyState(t) != 0) {
                    neigh.push_back(t);
                    send_counts[ctx.rank_of_state(t)]++;
                }
            }
        }
        { std::vector<state_t> tmp; std::swap(tmp, frontier); }  // free frontier

        // (b) bucket neighbours by destination rank into the send buffer.
        send_displs[0] = 0;
        for (int r = 1; r < N; ++r)
            send_displs[r] = send_displs[r-1] + send_counts[r-1];
        send_buf.resize(neigh.size());
        {
            std::vector<int> cur(send_displs);
            for (const auto& t : neigh)
                send_buf[cur[ctx.rank_of_state(t)]++] = t;
        }
        { std::vector<state_t> tmp; std::swap(tmp, neigh); }     // free neigh

        // (c) exchange counts then states (self bucket rides along, like apply).
        MPI_Alltoall(send_counts.data(), 1, get_mpi_type<int>(),
                     recv_counts.data(), 1, get_mpi_type<int>(), MPI_COMM_WORLD);
        recv_displs[0] = 0;
        for (int r = 1; r < N; ++r)
            recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
        size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), 0ull);
        recv_buf.resize(recv_total);
        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                      get_mpi_type<state_t>(),
                      recv_buf.data(), recv_counts.data(), recv_displs.data(),
                      get_mpi_type<state_t>(), MPI_COMM_WORLD);
        { std::vector<state_t> tmp; std::swap(tmp, send_buf); }  // free send_buf

        // (d) insert received states; the genuinely new ones are next frontier.
        next_frontier.clear();
        for (const auto& r : recv_buf)
            if (known.insert(r).second) next_frontier.push_back(r);
        { std::vector<state_t> tmp; std::swap(tmp, recv_buf); }  // free recv_buf

        // (e) fixed point when nobody added anything this round.
        int local_new = next_frontier.empty() ? 0 : 1;
        int any_new = 0;
        MPI_Allreduce(&local_new, &any_new, 1, get_mpi_type<int>(),
                      MPI_MAX, MPI_COMM_WORLD);
        std::swap(frontier, next_frontier);
        ++round;
        if (!any_new) break;
    }

    n_rounds = round;
    return known;
}


// Step 4: hand the discovery set off to a plain std::vector.
// Stream the (sorted) set into the vector in chunks, erasing the copied nodes as
// we go so the set frees memory while the vector fills, instead of holding both
// structures at full size simultaneously.
inline std::vector<state_t> drain_set_to_vector(std::set<state_t>& s,
                                                size_t chunk_states)
{
    if (chunk_states == 0) chunk_states = (1u << 20);
    std::vector<state_t> out;
    out.reserve(s.size());
    while (!s.empty()) {
        auto it = s.begin();
        size_t k = 0;
        for (; k < chunk_states && it != s.end(); ++k) {
            out.push_back(*it);
            it = s.erase(it);   // frees this node, returns next
        }
    }
    return out;
}


// Convenience: full seed -> scatter -> grow -> drain, returning this rank's
// sorted owned states. The sector must be non-empty (see the file header for
// why); this is asserted only informally -- callers validate and error out with
// a helpful message. Rank 0 performs the DFS; milestones are logged via
// logging::log so both callers report consistently. raw_local is set to this
// rank's owned count, raw_global to the summed global dim, n_rounds to the
// number of growth rounds.
template <typename coeff_t>
std::vector<state_t> build_grown_basis_local(
        const lattice& lat, int num_spinon_pairs,
        const std::vector<size_t>& perm,
        const std::vector<int>& sector,
        const SymbolicOpSum<coeff_t>& H,
        const MPIHashContext& ctx,
        size_t seeds_per_rank, size_t drain_chunk,
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
    std::set<state_t> grown = grow_basis(std::move(my_seeds), H, ctx, n_rounds);

    raw_local = grown.size();
    raw_global = 0;
    MPI_Allreduce(&raw_local, &raw_global, 1, get_mpi_type<size_t>(),
                  MPI_SUM, MPI_COMM_WORLD);

    // --- Step 4: stream the discovery set into a plain vector ---------------
    std::vector<state_t> local = drain_set_to_vector(grown, drain_chunk);
    { std::set<state_t> tmp; std::swap(tmp, grown); }  // ensure set fully freed
    return local;
}


}  // namespace basis_grow
}  // namespace projED

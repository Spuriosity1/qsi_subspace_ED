#pragma once
#include "operator.hpp"
#include <cassert>
#include <array>
#include <cstddef>
#include <functional>
#include <mpi.h>
//#include <bit>
#include "mpi_context.hpp"


// MPI-distributed wrapper around any local basis type.
// Each rank holds the subset of states whose hash maps to that rank.
// After loading and redistribution, on_states_changed() is called on the
// local basis so that search-acceleration structures (bounds, sentinels, …)
// are rebuilt automatically for whichever LocalBasis is used.
template<typename LocalBasis>
struct ZBasisMPI : public LocalBasis {
    using ctx_t = MPIHashContext;
    // Load slab from file AND redistribute to correct ranks in one step.
    void load_from_file(const fs::path& bfile, const std::string& dataset="basis");
    // Load a contiguous slab from file without any MPI redistribution.
    // Call redistribute() (optionally after remove_null_states) to finish setup.
    void load_raw(const fs::path& bfile, const std::string& dataset="basis");
    // Redistribute states to their hash-correct ranks, then rebuild search
    // structures and populate global_dim / dim_of_rank metadata.
    void redistribute();
    // Adopt an already-generated slab of states (any distribution across
    // ranks, need not be sorted). Follow with remove_null_states (optional)
    // then redistribute() to finish setup.
    void adopt_states(std::vector<ZBasisBase::state_t>&& s) {
        this->states = std::move(s);
    }

    // Per-block hook applied to each block of freshly-read states before they
    // are hash-partitioned (e.g. to drop states annihilated by every term of
    // H). The filter must be ownership-independent so it is valid to apply it
    // on the finding rank, before redistribution.
    using StateFilter = std::function<void(std::vector<ZBasisBase::state_t>&)>;

    // Streaming replacement for adopt_states()+redistribute() used by the fused
    // pipeline. Reads the rank-local binary shard at shard_path in blocks of
    // block_records Uint128s, applies filter to each block (if provided),
    // hash-partitions it to the owning ranks via a collective Alltoallv, and
    // accumulates the owned states locally. This bounds resident memory to
    // (owned states + one block + send/recv buffers) instead of the full set of
    // states this rank happened to find during the search. Every rank must call
    // this collectively; ranks whose shard is exhausted early keep participating
    // in empty rounds until all shards are drained.
    void ingest_shard_streaming(const fs::path& shard_path,
                                size_t block_records,
                                const StateFilter& filter = {});

    ZBasisBase::idx_t global_dim() const { return _global_dim; }
    ZBasisBase::idx_t dim_of_rank(int r) const { return _all_rank_dims[r]; }
    private:
    void tfer_states_to_correct_ranks(ctx_t& ctx);
    // Sort the local partition, rebuild search-acceleration structures, and
    // populate the global_dim / dim_of_rank metadata. Shared tail of
    // redistribute() and ingest_shard_streaming().
    void finalize_local_partition(ctx_t& ctx);
    ZBasisBase::idx_t _global_dim = 0;
    std::vector<ZBasisBase::idx_t> _all_rank_dims;
};

using ZBasisBST_HashMPI     = ZBasisMPI<ZBasisBST>;
using ZBasisInterp_HashMPI  = ZBasisMPI<ZBasisInterp>;
using ZBasisBSTFast_HashMPI = ZBasisMPI<ZBasisBSTFast>;

using MPIctx=MPIHashContext;


// Forward declaration so the apply helpers can take a Timer& without pulling
// timeit.hpp into this header (it is included in the .cpp).
class Timer;

enum class MPILazyOpSumStrategy {
    PIPE, PREALLOC, PREALLOC_P2P
};

inline std::ostream& operator<<(std::ostream& o, const MPILazyOpSumStrategy s){
    switch(s) {
        case MPILazyOpSumStrategy::PIPE: o<<"PIPE"; break;
        case MPILazyOpSumStrategy::PREALLOC: o<<"PREALLOC"; break;
        case MPILazyOpSumStrategy::PREALLOC_P2P: o<<"PREALLOC_P2P"; break;
        default: o<<static_cast<int>(s);
    }
    return o;
}

inline MPILazyOpSumStrategy parse_mpi_strategy(const std::string& s) {
    if (s == "pipe")         return MPILazyOpSumStrategy::PIPE;
    if (s == "prealloc")     return MPILazyOpSumStrategy::PREALLOC;
    if (s == "prealloc_p2p") return MPILazyOpSumStrategy::PREALLOC_P2P;
    throw std::runtime_error("Unrecognised strategy '" + s +
            "' (expected: pipe | prealloc | prealloc_p2p)");
}

template<RealOrCplx coeff_t, Basis B>
struct MPILazyOpSum {

    const MPILazyOpSumStrategy apply_strat;

    using Scalar = coeff_t;
    explicit MPILazyOpSum(
            const B& local_basis_, const SymbolicOpSum<coeff_t>& ops_,
            MPIctx& context_,
            MPILazyOpSumStrategy strat=MPILazyOpSumStrategy::PIPE
            ) : apply_strat(strat), basis(local_basis_), ops(ops_), ctx(context_)  {
    }

    MPILazyOpSum operator=(const MPILazyOpSum& other) = delete;

	// Core evaluator 
    // Applies y = A x (sets y=0 first)
	void evaluate(const coeff_t* x, coeff_t* y)
    {
		std::fill(y, y + basis.dim(), coeff_t(0));
        this->evaluate_add(x, y);
	}

    // Does y += A*x, where y[i] and x[i] are both indexed from the start of the local block
	void evaluate_add(const coeff_t* x, coeff_t* y);

protected:

    // Per-operator communication plan. The basis and Hamiltonian are fixed, so
    // the alltoallv counts/displs for each off-diagonal term never change; we
    // precompute them once (build_apply_metadata) and reuse across every apply.
    // Records (coeff+state) travel as two parallel native transfers: coeff as
    // coeff_t, state as uint64 (a Uint128 is two uint64), so *_sizes/_displs are
    // in record units for the coeff transfer and doubled (*_u64) for the state
    // transfer. The self->self block is included so the local update rides the
    // same exchange instead of a separate path.
    struct OperatorMetadata {
        std::vector<int> send_sizes;   // records sent to each rank (incl. self)
        std::vector<int> send_displs;  // prefix sums, in records
        std::vector<int> recv_sizes;   // records received from each rank
        std::vector<int> recv_displs;
        std::vector<int> send_sizes_u64;   // == 2*send_sizes  (uint64 units)
        std::vector<int> send_displs_u64;  // == 2*send_displs
        std::vector<int> recv_sizes_u64;   // == 2*recv_sizes
        std::vector<int> recv_displs_u64;  // == 2*recv_displs
        int64_t send_total = 0;        // == send_displs.back() + send_sizes.back()
        int64_t recv_total = 0;
    };
    mutable std::vector<OperatorMetadata> op_comm_metadata;

    // Double-buffered send/recv buffers (coeff + state, parallel arrays), so op
    // i's exchange can be in flight while op i-1's records are searched into y
    // and op i+1's sends are filled. Sized once to the largest per-operator
    // total the first time a prealloc apply is called.
    mutable std::array<std::vector<coeff_t>, 2>             send_dy_ring;
    mutable std::array<std::vector<ZBasisBase::state_t>, 2> send_state_ring;
    mutable std::array<std::vector<coeff_t>, 2>             recv_dy_ring;
    mutable std::array<std::vector<ZBasisBase::state_t>, 2> recv_state_ring;

    // Populate op_comm_metadata and size the ring buffers. Idempotent; a no-op
    // once built (basis/H are const for the object's lifetime).
    void build_apply_metadata() const;

    // Shared building blocks of the prealloc apply variants (threaded).
    // fill_sends: counting-sort operator oi's records into send_{dy,state}_ring[slot].
    // apply_recvs: search recv_state_ring[slot] into the basis and accumulate recv_dy_ring.
    void fill_sends(int slot, size_t oi, const coeff_t* x, Timer& timer) const;
    void apply_recvs(int slot, size_t oi, coeff_t* y, Timer& timer) const;

    void evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const;
    void evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y) const;
    // Prealloc + precomputed comm plan, records shipped by one Ialltoallv/op.
    void evaluate_add_off_diag_pipeline_prealloc(const coeff_t* x, coeff_t* y) const;
    // Same, but records shipped by point-to-point Isend/Irecv per peer/op.
    void evaluate_add_off_diag_prealloc_p2p(const coeff_t* x, coeff_t* y) const;

	const B& basis;
	const SymbolicOpSum<coeff_t> ops;
    MPIctx& ctx;

};


template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::evaluate_add(const coeff_t* x, coeff_t* y) {
    evaluate_add_diagonal(x, y);
    switch (apply_strat) {
        case MPILazyOpSumStrategy::PIPE:
            evaluate_add_off_diag_pipeline(x, y);
            break;
        case MPILazyOpSumStrategy::PREALLOC:
            evaluate_add_off_diag_pipeline_prealloc(x, y);
            break;
        case MPILazyOpSumStrategy::PREALLOC_P2P:
            evaluate_add_off_diag_prealloc_p2p(x, y);
            break;
        default:
            throw std::runtime_error("The developer has not implemented this strategy yet.");
    }
}






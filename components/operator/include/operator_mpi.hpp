#pragma once
#include "operator.hpp"
#include <cassert>
#include <array>
#include <cstddef>
#include <functional>
#include <mpi.h>
//#include <bit>
#include "mpi_context.hpp"


// A single off-diagonal apply contribution: the coefficient to accumulate and
// the target state it lands on. Stored contiguously (array-of-structs) so the
// pipeline ships ONE homogeneous buffer per operator instead of parallel
// coeff/state arrays, and so each record's key+value share a cache line.
// POD + trivially copyable => it maps 1:1 onto a committed MPI datatype below.
template<RealOrCplx coeff_t>
struct ApplyRecord {
    coeff_t coeff;
    ZBasisBase::state_t state;
};

// Build (once) the committed MPI struct type for ApplyRecord<coeff_t>. The
// resize to sizeof(record) pins the type's extent to the C++ array stride so
// arrays of records send/receive correctly regardless of padding.
template<RealOrCplx coeff_t>
inline MPI_Datatype make_apply_record_mpi_type() {
    using R = ApplyRecord<coeff_t>;
    int          blocklen[2] = {1, 1};
    MPI_Aint     displ[2]    = {offsetof(R, coeff), offsetof(R, state)};
    MPI_Datatype types[2]    = {get_mpi_type<coeff_t>(),
                                get_mpi_type<ZBasisBase::state_t>()};
    MPI_Datatype packed, resized;
    MPI_Type_create_struct(2, blocklen, displ, types, &packed);
    MPI_Type_create_resized(packed, 0, sizeof(R), &resized);
    MPI_Type_commit(&resized);
    MPI_Type_free(&packed);
    return resized;
}

// Same static-commit trick as get_mpi_type<Uint128>(): built lazily on first
// use (after MPI_Init), cached for the process lifetime. Magic-static init is
// thread-safe.
template<> inline MPI_Datatype get_mpi_type<ApplyRecord<double>>() {
    static MPI_Datatype dtype = make_apply_record_mpi_type<double>();
    return dtype;
}
template<> inline MPI_Datatype get_mpi_type<ApplyRecord<std::complex<double>>>() {
    static MPI_Datatype dtype = make_apply_record_mpi_type<std::complex<double>>();
    return dtype;
}


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
    // Counts/displs are in units of ApplyRecord (one collective now covers both
    // coeff and state). The self->self block is included so the local update
    // rides the same alltoallv instead of a separate path.
    struct OperatorMetadata {
        std::vector<int> send_sizes;   // records sent to each rank (incl. self)
        std::vector<int> send_displs;  // prefix sums, in records
        std::vector<int> recv_sizes;   // records received from each rank
        std::vector<int> recv_displs;
        int64_t send_total = 0;        // == send_displs.back() + send_sizes.back()
        int64_t recv_total = 0;
    };
    mutable std::vector<OperatorMetadata> op_comm_metadata;

    // Double-buffered record buffers, so op i's
    // alltoallv can be in flight while op i-1's records are searched into y and
    // op i+1's sends are filled. 
    // Sized once to the largest per-operator total the first time 
    // evaluate_add_off_diag_pipeline_prealloc is called.
    mutable std::array<std::vector<ApplyRecord<coeff_t>>, 2> send_ring;
    mutable std::array<std::vector<ApplyRecord<coeff_t>>, 2> recv_ring;

    // Populate op_comm_metadata and size the ring buffers. Idempotent; a no-op
    // once built (basis/H are const for the object's lifetime).
    void build_apply_metadata() const;

    // Shared building blocks of the prealloc apply variants (threaded).
    // fill_sends: counting-sort operator oi's records into send_ring[slot].
    // apply_recvs: search recv_ring[slot] into the local basis and accumulate.
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






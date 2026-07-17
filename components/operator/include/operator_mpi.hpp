#pragma once
#include "operator.hpp"
#include <cassert>
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


template<RealOrCplx coeff_t, Basis B>
struct MPILazyOpSum {
    using Scalar = coeff_t;
    explicit MPILazyOpSum(
            const B& local_basis_, const SymbolicOpSum<coeff_t>& ops_,
            MPIctx& context_
            ) : basis(local_basis_), ops(ops_), ctx(context_) {
    }

    MPILazyOpSum operator=(const MPILazyOpSum& other) = delete;

	// Core evaluator 
    // Applies y = A x (sets y=0 first)
	void evaluate(const coeff_t* x, coeff_t* y)
    {
		std::fill(y, y + basis.dim(), coeff_t(0));
        this->evaluate_add(x, y);
	}

    // allocates send/receive buffers for batched MPI alltoallv.
    // batch_size: number of local basis states per communication round (-1 = all).
    // If not called, evaluate_add falls back to the pipeline implementation.
    void allocate_temporaries(int batch_size = -1);

    void set_batch_size(int b) { batch_size = b; }

    // Build a static apply plan: the basis, Hamiltonian and batch boundaries
    // never change between applies, so freeze the per-batch communication
    // counts and precompute the local target index of every record this rank
    // will receive (plus every self-update record). Steady-state applies then
    // ship only dy (one Alltoallv, 8 B/record) and scatter through the index
    // lists — no search, no sort, no count exchange.
    // Costs ~4 B per off-diagonal nonzero of the local slab; if mem_cap_bytes
    // is nonzero and any rank would exceed it, the plan is not built (check
    // has_plan()) and evaluate_add keeps using the batched/pipeline path.
    // Calls allocate_temporaries(batch_size) if buffers are not yet sized.
    void build_plan(int batch_size = -1, size_t mem_cap_bytes = 0);
    bool has_plan() const { return plan_built; }
    size_t plan_bytes() const;

    // Free the state-record send/recv buffers once a plan is built; only dy
    // buffers are needed thereafter. The batched/pipeline paths become
    // unusable until allocate_temporaries() is called again.
    void release_state_buffers();

    // Does y += A*x, where y[i] and x[i] are both indexed from the start of the local block
	void evaluate_add(const coeff_t* x, coeff_t* y);

protected:
    void evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const;
    void evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y) const;
    void evaluate_add_off_diag_batched(const coeff_t* x, coeff_t* y);
    void evaluate_add_off_diag_planned(const coeff_t* x, coeff_t* y);

	const B& basis;
	const SymbolicOpSum<coeff_t> ops;
    MPIctx& ctx;

    int batch_size = -1; // -1 = all states in one round

    // Global number of communication rounds per apply. Local dims (and thus
    // ceil(dim/batch_size)) differ across ranks, but every rank must take part
    // in every collective round; set by allocate_temporaries via an Allreduce.
    int64_t n_batches = 0;

    // flat contiguous buffers; allocated by allocate_temporaries()
    std::vector<coeff_t> send_dy;
    std::vector<ZBasisBST::state_t> send_state;
    std::vector<MPI_Count> send_displs;
    std::vector<MPI_Count> send_counts;

    std::vector<coeff_t> recv_dy;
    std::vector<ZBasisBST::state_t> recv_state;
    std::vector<MPI_Count> recv_displs;
    std::vector<MPI_Count> recv_counts;

    // --- static apply plan (build_plan / evaluate_add_off_diag_planned) ---
    bool plan_built = false;
    int64_t plan_bs = 0;                  // local batch size the plan was built with
    std::vector<int> plan_sc;             // [b*N + r] frozen send counts (self slot kept)
    std::vector<int> plan_rc;             // [b*N + r] frozen recv counts (self slot zeroed)
    std::vector<uint32_t> plan_tgt;       // local target index per received record
    std::vector<int64_t>  plan_tgt_offs;  // [n_batches+1] offsets into plan_tgt
    std::vector<uint32_t> plan_self;      // local target index per self-update record
    std::vector<int64_t>  plan_self_offs; // [n_batches+1] offsets into plan_self
};


template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::evaluate_add(const coeff_t* x, coeff_t* y) {
    evaluate_add_diagonal(x, y);
    if (plan_built)
        evaluate_add_off_diag_planned(x, y);
    else if (!send_counts.empty())
        evaluate_add_off_diag_batched(x, y);
    else
        evaluate_add_off_diag_pipeline(x, y);
}






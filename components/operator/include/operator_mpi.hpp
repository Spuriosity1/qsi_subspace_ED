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

    // Does y += A*x, where y[i] and x[i] are both indexed from the start of the local block
	void evaluate_add(const coeff_t* x, coeff_t* y);

protected:

    // State for pipelined communication
    struct OperatorCommState {
        std::vector<MPI_Request> requests;

        std::vector<std::vector<coeff_t>> send_dy;
        std::vector<std::vector<ZBasisBase::state_t>> send_states;

        std::vector<std::vector<coeff_t>> recv_dy_bufs;
        std::vector<std::vector<ZBasisBase::state_t>> recv_states_bufs;

        MPI_Request count_exchange_req;
        std::vector<int> recvcounts;

        bool count_exchange_done = false;

        void resize(int world_size){
            send_dy.resize(world_size);
            send_states.resize(world_size);

            recv_dy_bufs.resize(world_size);
            recv_states_bufs.resize(world_size);

            recvcounts.resize(world_size);
        }

        void reset_for_new_op(){
            count_exchange_done=false;
            requests.clear();
            for (auto& v : send_dy)     v.clear();
            for (auto& v : send_states) v.clear();
            for (auto& v : recv_dy_bufs)     v.clear();
            for (auto& v : recv_states_bufs) v.clear();
        }
    };


    void evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const;
    void evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y) const;
    // void evaluate_add_off_diag_pipeline_prealloc(const coeff_t* x, coeff_t* y) const;

	const B& basis;
	const SymbolicOpSum<coeff_t> ops;
    MPIctx& ctx;

};


template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::evaluate_add(const coeff_t* x, coeff_t* y) {
    evaluate_add_diagonal(x, y);
    evaluate_add_off_diag_pipeline(x, y);
}






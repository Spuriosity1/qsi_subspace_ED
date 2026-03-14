#pragma once
#include "operator.hpp"
#include <cassert>
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
    void load_from_file(const fs::path& bfile, const std::string& dataset="basis");
    ZBasisBase::idx_t global_dim() const { return _global_dim; }
    ZBasisBase::idx_t dim_of_rank(int r) const { return _all_rank_dims[r]; }
    private:
    void tfer_states_to_correct_ranks(ctx_t& ctx);
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
            ) : basis(local_basis_), ops(ops_), ctx(context_),
    send_dy(ctx.world_size), send_state(ctx.world_size) {
    }

    MPILazyOpSum operator=(const MPILazyOpSum& other) = delete;

	// Core evaluator 
    // Applies y = A x (sets y=0 first)
	void evaluate(const coeff_t* x, coeff_t* y)
    {
		std::fill(y, y + basis.dim(), coeff_t(0));
        this->evaluate_add(x, y);
	}

    // allocates send/receive buffers for MPI alltoall
    // based on current matrix structure
    void allocate_temporaries();

    // Does y += A*x, where y[i] and x[i] are both indexed from the start of the local block
	void evaluate_add(const coeff_t* x, coeff_t* y); 

protected:
    void evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const;
//    void evaluate_add_off_diag_sync(const coeff_t* x, coeff_t* y) const;
    void evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y) const;
//    void evaluate_add_off_diag_batched(const coeff_t* x, coeff_t* y);

	const B& basis;
	const SymbolicOpSum<coeff_t> ops;
    MPIctx& ctx;

    // metadata
    std::vector<coeff_t> send_dy; // contiguous buffer
    std::vector<ZBasisBST::state_t> send_state; 
    std::vector<MPI_Count> send_displs;
    std::vector<MPI_Count> send_counts;

    std::vector<coeff_t> recv_dy;
    std::vector<ZBasisBST::state_t> recv_state;
    std::vector<MPI_Count> recv_displs;
    std::vector<MPI_Count> recv_counts;
private:
    static constexpr double APPLY_TOL=1e-15;

    void inplace_bucket_sort(std::vector<ZBasisBase::state_t>& states,
        std::vector<coeff_t>& c,
        std::vector<int>& bucket_sizes,
        std::vector<int>& bucket_starts
        ) const;

//    void rebalance_work(std::vector<int>& cost_per_rank);

};


template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::evaluate_add(const coeff_t* x, coeff_t* y) {
    evaluate_add_diagonal(x, y);
    evaluate_add_off_diag_pipeline(x, y);

    //evaluate_add_off_diag_batched(x, y);
}






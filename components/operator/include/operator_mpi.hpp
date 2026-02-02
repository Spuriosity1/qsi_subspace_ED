#pragma once
#include "operator.hpp"
#include <cassert>
#include <mpi.h>
//#include <bit>
#include "mpi_context.hpp"

typedef SparseMPIContext<ZBasisBST::idx_t> MPIctx;

struct MPI_ZBasisBST : public ZBasisBST 
{
     MPIctx load_from_file(const fs::path& bfile, const std::string& dataset="basis");
//     void load_state(std::vector<double>& psi, const fs::path& eig_file);
};


template<RealOrCplx coeff_t, Basis B>
struct MPILazyOpSum {
    using Scalar = coeff_t;
    explicit MPILazyOpSum(
            const B& local_basis_, const SymbolicOpSum<coeff_t>& ops_,
            MPIctx& context_
            ) : basis(local_basis_), ops(ops_), ctx(context_),
    send_dy(ctx.world_size), send_state(ctx.world_size) {
        allocate_temporaries();
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
    void evaluate_add_off_diag_batched(const coeff_t* x, coeff_t* y);

	const B& basis;
	const SymbolicOpSum<coeff_t> ops;
    MPIctx& ctx;

    // metadata
    std::vector<coeff_t> send_dy; // contiguous buffer
    std::vector<ZBasisBST::state_t> send_state; 
    std::vector<int> send_displs;
    std::vector<int> send_counts;

    std::vector<coeff_t> recv_dy;
    std::vector<ZBasisBST::state_t> recv_state;
    std::vector<int> recv_displs;
    std::vector<int> recv_counts;
private:
    static constexpr double APPLY_TOL=1e-15;

    void inplace_bucket_sort(std::vector<ZBasisBase::state_t>& states,
        std::vector<coeff_t>& c,
        std::vector<int>& bucket_sizes,
        std::vector<int>& bucket_starts
        ) const;

    void rebalance_work(std::vector<int>& cost_per_rank);

};


template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::evaluate_add(const coeff_t* x, coeff_t* y) {
    evaluate_add_diagonal(x, y);
//    evaluate_add_off_diag_pipeline(x, y);
    evaluate_add_off_diag_batched(x, y);
}






#pragma once
#include "operator.hpp"
#include <cassert>
#include <mpi.h>
//#include <bit>
#include "mpi_context.hpp"

typedef MPIContext<ZBasisBST::state_t, ZBasisBST::idx_t> MPIctx;

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
            ) : basis(local_basis_), ops(ops_), ctx(context_) {
    }

    MPILazyOpSum operator=(const MPILazyOpSum& other) = delete;

	// Core evaluator 
    // Applies y = A x (sets y=0 first)
	void evaluate(const coeff_t* x, coeff_t* y) const
    {
		std::fill(y, y + basis.dim(), coeff_t(0));
        this->evaluate_add(x, y);
	}

    // Does y += A*x, where y[i] and x[i] are both indexed from the start of the local block
	void evaluate_add(const coeff_t* x, coeff_t* y) const; 

protected:
    void evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const;
//    void evaluate_add_off_diag_sync(const coeff_t* x, coeff_t* y) const;
    void evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y) const;

	const B& basis;
	const SymbolicOpSum<coeff_t> ops;
    MPIctx& ctx;
private:
    static constexpr double APPLY_TOL=1e-15;

    void inplace_bucket_sort(std::vector<ZBasisBase::state_t>& states,
        std::vector<coeff_t>& c,
        std::vector<int>& bucket_sizes,
        std::vector<int>& bucket_starts
        ) const;

};


template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::evaluate_add(const coeff_t* x, coeff_t* y) const {
    evaluate_add_diagonal(x, y);
    evaluate_add_off_diag_pipeline(x, y);
}



#include "operator_matrix.hpp"
#include <omp.h>

// performs y <- Ax + y
template <RealOrCplx coeff_t, Basis B>
void LazyOpSum<coeff_t, B>::evaluate_add_off_diag(const coeff_t* x, coeff_t* y) const {
    
    ZBasisBase::idx_t J;
    ZBasisBase::state_t state;

    for (const auto& [c, op] : ops.off_diag_terms) {
        
        for (ZBasisBase::idx_t i = 0; i < basis.dim(); ++i) {
            state = basis[i];
            int sign = op.applyState(state);
            if ( sign != 0 && abs(x[i]) > APPLY_TOL ){
                sign *= basis.search(state, J);
                y[J] +=  c * x[i] * sign;
            }
        }
        
    }
}


// performs y <- Ax + y
// accelerated with openMPI
template <RealOrCplx coeff_t, Basis B>
void LazyOpSum<coeff_t, B>::evaluate_add_off_diag_omp(const coeff_t* x, coeff_t* y) const {
    static const size_t CHUNK_SIZE = 10000;


    for (const auto& term : ops.off_diag_terms) {
        const auto& c = term.first;   // Extract before parallel region
        const auto& op = term.second;


#pragma omp parallel
    {
        ZBasisBase::idx_t J;
        ZBasisBase::state_t state;
        double dy;

#pragma omp for schedule(static)
        for (ZBasisBase::idx_t i = 0; i < basis.dim(); ++i) {
            state = basis[i];
            int sign = op.applyState(state);
            if (sign != 0 && abs(x[i]) > APPLY_TOL ){
                sign *= basis.search(state, J);
                dy = c * x[i] * sign;
                 
                #pragma omp atomic
                y[J] += dy;
            }
        }

    } // end parallel region
        
    }
}



// performs y <- Ax + y
template <RealOrCplx coeff_t, Basis B>
void LazyOpSum<coeff_t, B>::evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const {
    for (const auto& term : ops.diagonal_terms) {
        const auto& c = term.first;   
        const auto& op = term.second;

        assert(op.is_diagonal());

        #pragma omp parallel for schedule(static)
        for (ZBasisBase::idx_t i = 0; i < basis.dim(); ++i) {
            ZBasisBase::state_t psi = basis[i];
            coeff_t dy = c * x[i] * static_cast<double>(op.applyState(psi));
            // completely in place, no i collisions
            y[i] += dy;
        }
        
    }
}

#ifdef USE_CUDA
__global__ void evaluate_add_off_diag_kernel(
    const double* x, double* y,
    const __uint128_t* states, // basis states
    const double c,
    size_t dim
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    __uint128_t state = states[i];

    // Apply your custom logic here
    int sign = applyState(state); // device version
    size_t J = searchIndex(state); // device version

    double dy = c * x[i] * static_cast<double>(sign);
    if (dy != 0.0) {
        atomicAdd(&y[J], dy); // atomic to avoid race conditions
    }
}
#endif


template <RealOrCplx coeff_t, Basis basis_t>
void LazyOpSum<coeff_t, basis_t>::evaluate_add(const coeff_t* x, coeff_t* y) const {
    evaluate_add_off_diag_omp(x, y);
    evaluate_add_diagonal(x, y);
}

// explicit template instantiations
template struct LazyOpSum<double, ZBasisBST>;
template struct LazyOpSum<double, ZBasisInterp>;
// template struct LazyOpSum<std::complex<double>>;

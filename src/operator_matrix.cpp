#include "operator_matrix.hpp"

// performs y <- Ax + y
template <RealOrCplx coeff_t>
void LazyOpSum<coeff_t>::evaluate_add(const coeff_t* x, coeff_t* y) const {
    for (const auto& [c, op] : ops.terms) {
        //op.apply_add(basis, x, y, c);
        
        #pragma omp parallel for schedule(static)
        for (ZBasis::idx_t i = 0; i < basis.dim(); ++i) {
            ZBasis::idx_t J = i;
            coeff_t sign = x[i] * static_cast<double>(op.applyIndex(basis, J));

            y[J] += sign*c;
        }
    }
}

// explicit template instantiations
template struct LazyOpSum<double>;
// template struct LazyOpSum<std::complex<double>>;

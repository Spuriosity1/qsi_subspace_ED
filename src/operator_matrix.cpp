#include "operator_matrix.hpp"
#include <omp.h>

// performs y <- Ax + y
template <RealOrCplx coeff_t>
void LazyOpSum<coeff_t>::evaluate_add(const coeff_t* x, coeff_t* y) const {
    for (const auto& [c, op] : ops.terms) {
        //op.apply_add(basis, x, y, c);
        #pragma omp parallel
        {
            std::vector<std::pair<ZBasis::idx_t, coeff_t>> local_updates;
            local_updates.reserve(basis.dim() / omp_get_num_threads() + 64);

            #pragma omp for schedule(static) nowait
            for (ZBasis::idx_t i = 0; i < basis.dim(); ++i) {
                ZBasis::idx_t J = i;
                coeff_t dy = c * x[i] * static_cast<double>(op.applyIndex(basis, J));
                local_updates.emplace_back(J, dy);
            }

            // Each thread writes its results once - minimal contention
            #pragma omp critical
            {
                for (const auto& [idx, val] : local_updates) {
                    y[idx] += val;
                }
            }
            
        }
    }
}

// explicit template instantiations
template struct LazyOpSum<double>;
// template struct LazyOpSum<std::complex<double>>;

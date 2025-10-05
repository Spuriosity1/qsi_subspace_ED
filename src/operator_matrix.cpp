#include "operator_matrix.hpp"
#include <omp.h>

// performs y <- Ax + y
template <RealOrCplx coeff_t>
void LazyOpSum<coeff_t>::evaluate_add_off_diag(const coeff_t* x, coeff_t* y) const {
    static const size_t CHUNK_SIZE = 10000;

    for (const auto& term : ops.off_diag_terms) {
        const auto& c = term.first;   // Extract before parallel region
        const auto& op = term.second;
        
        #pragma omp parallel
        {
            std::vector<std::pair<ZBasis::idx_t, coeff_t>> local_updates;
            local_updates.reserve(CHUNK_SIZE);

            #pragma omp for schedule(dynamic, CHUNK_SIZE) nowait
            for (ZBasis::idx_t i = 0; i < basis.dim(); ++i) {
                ZBasis::idx_t J = i;
                coeff_t dy = c * x[i] * static_cast<double>(op.applyIndex(basis, J));
                local_updates.emplace_back(J, dy);

                // Flush when chunk is full
                if (local_updates.size() >= CHUNK_SIZE) {
                    #pragma omp critical
                    {
                        for (const auto& [idx, val] : local_updates) {
                            y[idx] += val;
                        }
                    }
                    local_updates.clear();
                }
            }
            
            // flush remaining updates
            #pragma omp critical
            {
                for (const auto& [idx, val] : local_updates) {
                    y[idx] += val;
                }
            }       
        }
    }
}



// performs y <- Ax + y
template <RealOrCplx coeff_t>
void LazyOpSum<coeff_t>::evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const {
    for (const auto& term : ops.diagonal_terms) {
        const auto& c = term.first;   
        const auto& op = term.second;

        assert(op.is_diagonal());

        #pragma omp parallel for schedule(static)
        for (ZBasis::idx_t i = 0; i < basis.dim(); ++i) {
            ZBasis::state_t psi = basis[i];
            coeff_t dy = c * x[i] * static_cast<double>(op.applyState(psi));
            // completely in place, no i collisions
            y[i] += dy;
        }
        
    }
}




template <RealOrCplx coeff_t>
void LazyOpSum<coeff_t>::evaluate_add(const coeff_t* x, coeff_t* y) const {
    evaluate_add_off_diag(x, y);
    evaluate_add_diagonal(x, y);
}

// explicit template instantiations
template struct LazyOpSum<double>;
// template struct LazyOpSum<std::complex<double>>;

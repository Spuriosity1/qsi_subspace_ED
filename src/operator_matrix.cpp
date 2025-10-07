#include "operator_matrix.hpp"
#include <omp.h>

// performs y <- Ax + y
template <RealOrCplx coeff_t, Basis B>
void LazyOpSum<coeff_t, B>::evaluate_add_off_diag(const coeff_t* x, coeff_t* y) const {
    static const size_t CHUNK_SIZE = 10000;

    for (const auto& term : ops.off_diag_terms) {
        const auto& c = term.first;   // Extract before parallel region
        const auto& op = term.second;
        
//        std::vector<std::vector<std::pair<ZBasisBase::idx_t, coeff_t>>> thread_updates(omp_get_max_threads());
        #pragma omp parallel
        {
//            std::vector<std::pair<ZBasisBase::idx_t, coeff_t>> local_updates;
//            local_updates.reserve(CHUNK_SIZE);

            #pragma omp for schedule(static) nowait
            for (ZBasisBase::idx_t i = 0; i < basis.dim(); ++i) {
                ZBasisBase::idx_t J = i;
                auto sign = op.applyIndex(basis, J);
//                local_updates.emplace_back(J, dy);
                double dy = c * x[i] * static_cast<double>(sign);
                if (sign != 0) {
                    #pragma omp atomic
                    y[J] += dy;
                }

                // Flush when chunk is full
//                if (local_updates.size() >= CHUNK_SIZE) {
//                    #pragma omp critical
//                    {
//                        for (const auto& [idx, val] : local_updates) {
//                            y[idx] += val;
//                        }
//                    }
//                    local_updates.clear();
//                }
            }
//            
//            // flush remaining updates
//            #pragma omp critical
//            {
//                for (const auto& [idx, val] : local_updates) {
//                    y[idx] += val;
//                }
//            }       
        }
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




template <RealOrCplx coeff_t, Basis basis_t>
void LazyOpSum<coeff_t, basis_t>::evaluate_add(const coeff_t* x, coeff_t* y) const {
    evaluate_add_off_diag(x, y);
    evaluate_add_diagonal(x, y);
}

// explicit template instantiations
template struct LazyOpSum<double, ZBasisBST>;
template struct LazyOpSum<double, ZBasisInterp>;
template struct LazyOpSum<double, ZBasisHashmap>;
// template struct LazyOpSum<std::complex<double>>;

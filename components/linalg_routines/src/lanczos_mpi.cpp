#include "lanczos_mpi.hpp"
#include <random>
#include "mpi.h"
#include "timeit.hpp"
#include <iostream>
#include "common_bits_mpi.hpp"
#include "operator_mpi.hpp"

namespace projED {
namespace lanczos_mpi {


// Check convergence of the Lanczos iteration
// Returns true if converged, false otherwise
template<typename _S>
void check_lanczos_convergence(
    const std::vector<double>& alphas,
    const std::vector<double>& betas,
    double& eigval,
    size_t iteration,
    const Settings& settings,
    Result& res)
{
    // the Ritz vector must be real!
    std::vector<double> ritz_tmp;
    std::vector tmp_alphas(alphas);
    std::vector tmp_betas(betas);
    double old_eigval = eigval;

    // calculates the new eigval
    tridiagonalise_one(tmp_alphas, tmp_betas, eigval, ritz_tmp);
    
    res.eigval_error = std::abs(eigval - old_eigval);
    double rel_change = res.eigval_error / std::max(std::abs(eigval), 1e-10);
    
    if (settings.verbosity > 1) {
        std::cout << "  Convergence check at iter " << iteration 
                  << ": eigval=" << eigval 
                  << ", change=" << res.eigval_error 
                  << ", rel_change=" << rel_change << "\n";
    }
    
    // Check for convergence
    if (res.eigval_error < settings.abs_tol || rel_change < settings.rel_tol) {
        if (settings.verbosity > 0) {
            std::cout << "[Lanczos] Converged at iteration " << iteration 
                      << " with eigenvalue " << eigval << "\n";
        }
        res.converged= true;
        return;
    }

    res.converged= false;
}



// Runs the Lanczos recurrence MPI ENABLED
template<typename _S, typename ApplyFn>
Result lanczos_iterate(ApplyFn evaluate_add,
        std::vector<_S>& v, 
    std::vector<double>& alphas,
    std::vector<double>& betas,
        const Settings& settings,
        const std::vector<double>* ritz = nullptr,   // optional
        std::vector<_S>* eigvec = nullptr        // optional accumulator
        )
{
    // Work share scheme: node 0 manages the global recurrence

    auto local_dim = v.size();
    // Generating the starting vector
    std::mt19937 rng(settings.x0_seed);
    set_random_unit_mpi<_S>(v, rng);

    // v = current v_j (input normalized)
    // u = scratch/previous/next (rotates role each step)
    std::vector<_S> u(local_dim);
    
    alphas.reserve(settings.krylov_dim);
    betas.reserve(settings.krylov_dim);

    alphas.resize(0);
    betas.resize(0);
    
    Result retval;

    // Compared to implementation written in 
    // https://slepc.upv.es/documentation/reports/str5.pdf
    // betas[j] = \beta_{j+2}
    // alphas[j] = \alpha_{j+1}

    double beta = 0.0;
    double beta2_local = 0.0;
    double eigval = std::numeric_limits<double>::max();
    std::vector<_S> _tmp; // storage for the Ritz vector


    for (size_t j = 0; j < settings.max_iterations; j++) {
       // optional accumulation of Ritz combination
        if (eigvec && ritz && j < (size_t)ritz->size()) {
            axpy(*eigvec, v, (*ritz)[j]); // eigvec += ritz[j] * v
        } 
        
        if (j > 0) {
            // u = - beta_{j-1} * v_{j-1} (note: after swap, u holds v_{j-1})
            //sub(u, v, beta);
            mul(u, -beta);
        }
        // u += A * v
        TIMEIT("u += Av ", evaluate_add(v.data(), u.data());)

        // α_j = <v_j | u>
        double alpha_local = innerReal(v, u);
        double alpha;
        MPI_Allreduce(&alpha_local, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // u -= α_j * v_j
        axpy(u, v, -alpha);

        // β_j = ||u||
        beta2_local = innerReal(u,u);
        MPI_Allreduce(&beta2_local, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        beta = std::sqrt(beta);
        if (beta < settings.abs_tol) {
            if (settings.ctx.my_rank == 0 && settings.verbosity > 0){
                std::cout << "Lanczos breakdown at iteration "<<j<<std::endl;
            }
            break;
        }

        // rotate: new v = v_{j+1},
        std::swap(v, u);
        // v now contains the un-normalised "u" from before. 
        // u contains the old v_{j} for next iteration
        mul(v, 1.0/beta); // local op, no need to sync

        if (settings.ctx.my_rank == 0 && settings.verbosity > 1) {
            std::cout << "Iter "<< j << " a[j]="<<alpha<<" b[j]="<<beta << "\n";
        }

        alphas.push_back(alpha);
        betas.push_back(beta);

         // Convergence test: compute current eigenvalue estimate
        if (j >= settings.min_iterations) {
            if (settings.ctx.my_rank ==0) {
                check_lanczos_convergence<_S>(alphas, betas, eigval, j, settings, retval);
                if (settings.verbosity > 0) {
                    std::cout << "Iter "<< j << " eigval error " << retval.eigval_error << "\n";
                }
            }

            // Broadcast convergence status to all ranks
            int converged_flag = retval.converged ? 1 : 0;
            MPI_Bcast(&converged_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&retval.eigval_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&eigval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            retval.converged = (converged_flag == 1);

            if (retval.converged) {
                break;
            }
        }
    } // end lanczos iteration
      
    // Ensure all ranks have consistent final results
    if (settings.ctx.my_rank == 0) {
        retval.n_iterations = alphas.size();
    }
    MPI_Bcast(&retval.n_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&retval.converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&retval.eigval_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return retval;
}






template<typename _S, typename ApplyFn>
Result eigval0_impl(ApplyFn apply_add, double& eigval, 
        std::vector<_S>& v, 
        const Settings& settings
        )
{
    if (settings.verbosity >= 0){
        std::cout << settings;
    }

    if (v.size() == 0 ){
        throw std::logic_error("v called with size 0, did you forget to initialise it to the space's dimension?");
    }

    auto dim = v.size();

    std::vector<double> alphas;
    std::vector<double> betas;

    std::cout << "[Lanczos] finding lowest eigenvalue\n";
    Result res = lanczos_iterate(apply_add, v, alphas, betas, settings);

    std::cout << "[Lanczos] tridiagonalising in Krylov space\n";
    std::vector<double> ritz;

    std::vector tmp_alphas(alphas);
    std::vector tmp_betas(betas);
    tridiagonalise_one(tmp_alphas, tmp_betas, eigval, ritz);

    if(!settings.calc_eigenvector) return res;
                                    
    std::cout << "[Lanczos] iterating to determine eigenvector\n";
    // ----------------------------
    // Second pass: reconstruct eigenvector
    // ----------------------------
    std::vector<_S> evector(dim);
    
    lanczos_iterate(apply_add, v, alphas, betas, settings,
            &ritz, &evector
            );
    std::swap(evector, v);
    return res;
}

Result eigval0(projED::RealApplyFn f, 
        double& eigval, 
        std::vector<double>& v, 
        const Settings& settings_
        ){
    return eigval0_impl(f, eigval, v, settings_);
}

Result eigval0(projED::ComplexApplyFn f, double& eigval, 
        std::vector<std::complex<double>>& v, 
        const Settings& settings_  
        ){
    return eigval0_impl<std::complex<double>,projED::ComplexApplyFn>(f, eigval, v, settings_);
}

// end namespaces
}}

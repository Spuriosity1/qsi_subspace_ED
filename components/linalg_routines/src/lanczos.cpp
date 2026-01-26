#include "lanczos.hpp"
#include <random>
#include "blas_adapter.hpp"
#include "timeit.hpp"
#include <iostream>
#include "common_bits.hpp"

namespace projED {
namespace lanczos {


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
        res.eigval_converged= true;
        return;
    }

    res.eigval_converged= false;
}

template<typename _S>
void set_random_unit(std::vector<_S>& v, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto& x : v) x = dist(rng);

    double nrm = norm(v);
    for (auto& x : v) x /= nrm;
}


// Runs the Lanczos recurrence
// evaluate_add(src, dest) implements dest += H * src
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
    auto dim = v.size();
    // Generating the starting vector
//    v.resize(dim);
    std::mt19937 rng(settings.x0_seed);
    set_random_unit(v, rng);

    // v = current v_j (input normalized)
    // u = scratch/previous/next (rotates role each step)
    std::vector<_S> u(dim);
    
    assert( abs(norm(v) - 1) < settings.abs_tol );


    alphas.reserve(settings.krylov_dim);
    betas.reserve(settings.krylov_dim);

    alphas.resize(0);
    betas.resize(0);
    
    Result retval;
    retval.eigvec_converged = false;

    // Compared to implementation written in 
    // https://slepc.upv.es/documentation/reports/str5.pdf
    // betas[j] = \beta_{j+2}
    // alphas[j] = \alpha_{j+1}

    double beta = 0.0;
    double eigval = std::numeric_limits<double>::max();


    std::vector<_S> _tmp; // storage for the Ritz vector

    auto&iter_no = retval.n_iterations;
    for (iter_no = 0; iter_no < settings.max_iterations; iter_no++) {
       // optional accumulation of Ritz combination
        if (eigvec && ritz && iter_no < (size_t)ritz->size()) {
            axpy(*eigvec, v, (*ritz)[iter_no]); // eigvec += ritz[j] * v
        } 
        
        if (iter_no > 0) {
            // u = - beta_{j-1} * v_{j-1} (note: after swap, u holds v_{j-1})
            //sub(u, v, beta);
            mul(u, -beta);
        }
        // u += A * v
        TIMEIT("u += Av ", evaluate_add(v.data(), u.data());)

        // α_j = <v_j | u>
        double alpha = innerReal(v, u);

        // u -= α_j * v_j
        axpy(u, v, -alpha);

        // β_j = ||u||
        beta = norm(u);
        if (beta < settings.abs_tol) break;

        // rotate: new v = v_{j+1},
        std::swap(v, u);
        // v now contains the un-normalised "u" from before. 
        // u contains the old v_{j} for next iteration
        mul(v, 1/beta);

        if (settings.verbosity > 1) {
            std::cout << "Iter "<< iter_no << " a[j]="<<alpha<<" b[j]="<<beta << "\n";
        }

        alphas.push_back(alpha);
        betas.push_back(beta);

         // Convergence test: compute current eigenvalue estimate
        if (iter_no >= settings.min_iterations) {
            check_lanczos_convergence<_S>(alphas, betas, eigval, iter_no, settings, retval);
            if (settings.verbosity > 0) {
                std::cout << "Iter "<< iter_no << " eigval error " << retval.eigval_error << "\n";
            }

            if (retval.eigval_converged && ritz ) {
                retval.eigvec_converged = true;
                break;
            }
        }
    }

    // end of iteration: check if we got the right vector
    if (eigvec && settings.verify_eigenvector){
        std::cout << "[Lanczos] Verifying...\n";
        std::fill(u.begin(), u.end(), 0);
        evaluate_add(eigvec->data(), u.data());
        // u -= e * eigenvector
        axpy(u, *eigvec, -eigval);
        retval.eigvec_error = norm(u); 

        std::cout << "[Lanczos] Retval error is " << retval.eigvec_error <<"\n";
    }



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
    Result res;
    res = lanczos_iterate(apply_add, v, alphas, betas, settings);
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
    
    res = lanczos_iterate(apply_add, v, alphas, betas, settings,
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

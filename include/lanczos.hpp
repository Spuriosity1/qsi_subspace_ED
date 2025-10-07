#include "operator.hpp"
#include <Eigen/Eigenvalues>
#include <complex>
#include <random>
#include "timeit.hpp"

extern "C"{
    // hack
#define __STDC_VERSION__ 200000L
#define restrict __restrict__

#include <mrrr.h>
}


template<typename T>
concept Real = std::floating_point<T>;


template<Real S>
S inner(const std::vector<S>& u, const std::vector<S>& v) {

    S result = 0;

    #pragma omp parallel for reduction(+:result) 
    for (std::size_t i = 0; i < u.size(); i++) {
        result += u[i] * v[i];
    }

    return result;
}


// Overload for complex scalars
template<Real T>
std::complex<T> inner(const std::vector<std::complex<T>>& u,
                      const std::vector<std::complex<T>>& v) {
    std::complex<T> result{0, 0};

    #pragma omp declare reduction(+: std::complex<double> : \
    omp_out += omp_in) initializer(omp_priv = {0,0})

    #pragma omp parallel for reduction(+:result)
    for (std::size_t i = 0; i < u.size(); i++) {
        result += std::conj(u[i]) * v[i];
    }

    return result;
}

////////////////////////////////////////
// Real part of inner product

// trivial alias
template<Real S>
double innerReal(const std::vector<S>& u, const std::vector<S>& v) {
    return inner(u,v);
}


template<Real T>
std::complex<T> innerReal(const std::vector<std::complex<T>>& u,
                      const std::vector<std::complex<T>>& v) {
    T result;
#pragma omp parallel for reduction(+:result)
    for (std::size_t i=0; i<u.size(); i++){
        result += u[i].real() * v[i].real();
        result -= u[i].imag() * v[i].imag();
    }
    return result;
}

// Overload for complex scalars
template<Real T>
std::complex<T> norm(const std::vector<std::complex<T>>& u) {
    T result = 0.0;

    #pragma omp parallel for reduction(+:result)
    for (std::size_t i = 0; i < u.size(); i++) {
        result += std::conj(u[i]) * u[i];
    }

    return sqrt(result);
}

template<Real T>
double norm(const std::vector<T>& u) {
    double result = 0.0;

    #pragma omp parallel for reduction(+:result)
    for (std::size_t i = 0; i < u.size(); i++) {
        result += u[i] * u[i];
    }
    return sqrt(result);
}



// performs u <- u + v * c
template<typename T>
void axpy( std::vector<T>& u,
                      const std::vector<T>& v,
                      T c) {

    #pragma omp parallel for
    for (std::size_t i = 0; i < u.size(); i++) {
        u[i] += c * v[i];
    }
}


// performs u <- u + v * c
template<typename T>
void sub( std::vector<T>& u,
                      const std::vector<T>& v,
                      T c) {

    #pragma omp parallel for
    for (std::size_t i = 0; i < u.size(); i++) {
        u[i] -= c * v[i];
    }
}


// does v <- c * u
template<typename T>
void mul( std::vector<T>& v, std::vector<T>& u, T c) {
    #pragma omp parallel for
    for (std::size_t i = 0; i < u.size(); i++) {
        v[i] = u[i] * c;
    }
}


// does in place c * v
template<typename T>
void mul( std::vector<T>& v, T c) {
    #pragma omp parallel for
    for (std::size_t i = 0; i < v.size(); i++) {
        v[i] *= c;
    }
}

template<typename H, typename S>
concept LinearOperator = requires(const H& ham, const S* in, S* out) {
    // must have basis.dim() returning size_t
    { ham.cols() } -> std::convertible_to<std::size_t>;

    // must have an evaluate method that takes two pointers
    { ham.evaluate(in, out) } -> std::same_as<void>;
};


struct LanczosSettings {
    size_t krylov_dim = 10;
    double abs_tol = 1e-8;
    double rel_tol = 1e-8;
    size_t min_iterations = 30;
    size_t max_iterations = 5000;
    size_t convergence_check_interval = 5;
    int verbosity = 0;
    unsigned int x0_seed=10;
    bool calc_eigenvector = false;
};




// I don't think D and E actually get modified here, but the
// C backend doesn't know that
// never mind, they _do_ get overwritten...
inline void tridiagonalise(
    std::vector<double>& D, // the diagonal
    std::vector<double>& E, // the off-diagonal
    double& e, // the eigenvalue storage
    std::vector<double>& v // the eigenvector storage
    ){
    assert(E.size() >= D.size() -1); // silently ignore off diagonal parts beyond len(D)

    // build tridiagonal T and diagonalize
    int n = D.size();
    int m, ldz;
    int il, iu; // index bracket (unused)
    double vl, vu; // value bracket (unused)

//    /* Compute eigenpairs 'il' to 'iu' */
//    il = 0;
//    iu = 1;


    int tryRAC = 1;

    /* Allocate memory */
    // eigenvalues
    auto eigvals     = (double *) malloc( n    * sizeof(double) );
    auto Zsupp = (int *)    malloc( 2*n  * sizeof(int)    );

    m     = n;
    ldz   = n;

    auto eigvecs  = (double *) malloc((size_t) n * m * sizeof(double) );

    // type safety is for fucking NERDS
    /* Use MRRR to compute eigenvalues and -vectors */
    mrrr(
                (char*)("Vectors"), 
                (char*)("All"),
            &n, D.data(), E.data(),
            &vl, &vu,
            &il, &iu,
            &tryRAC,
            &m, eigvals, eigvecs,
            &ldz, 
            Zsupp);
//
//    int mrrr(char *jobz,
//            char *range, 
//            int *n, double *restrict D, double *restrict E,
//            double *vl, double *vu,
//            int *il, int *iu,
//         int *tryrac,
//         int *m, double *W, double *Z, int *ldz,
//         int *Zsupp);

    

    assert(info == 0);

    // find the smallest eigenvalue
    double min_eig = std::numeric_limits<double>::max();
    int min_id = 0;
    for (int i=0; i<m; i++){
        if (eigvals[i] < min_eig){
            min_eig = eigvals[i];
            min_id = i;
        }
    }
    e = min_eig;
    v.resize(m);
    for (int i=0; i<m; i++){
        v[i] = eigvecs[min_id*n + i]; 
        // col-major order I think?
    }


    /* Free allocated memory */
    free(eigvals);
    free(eigvecs);
    free(Zsupp);



//    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
//    for (size_t i = 0; i < m; i++) {
//        T(i, i) = alphas[i];
//        if (i + 1 < m) {
//            T(i, i+1) = betas[i];
//            T(i+1, i) = betas[i];
//        }
//    }
//    Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<_S>> solver(T);
//    e = solver.eigenvalues()(0); // smallest eigenvalue
//    // associated eigenvector 
//    Eigen::VectorXd ritz = solver.eigenvectors().col(0);


}





inline std::ostream& operator<<(std::ostream& os, const LanczosSettings& settings)
{
    os << "Lanczos Algorithm Settings:\n";
    os << "---------------------------\n";
    os << "  Krylov subspace dimension: " << settings.krylov_dim << "\n";
    os << "  Absolute Conv. tolerance:  " << settings.abs_tol << "\n";
    os << "  Relative Conv. tolerance:  " << settings.rel_tol << "\n";
    os << "  Minimum iterations:        " << settings.min_iterations << "\n";
    os << "  Maximum iterations:        " << settings.max_iterations << "\n";
    os << "  Conv. check interval:      " << settings.convergence_check_interval << "\n";
    os << "  Initial vector seed:       " << settings.x0_seed << "\n";
    os << "---------------------------\n";
    os << "  Verbosity level:           " << settings.verbosity << " (0=silent, higher=more output)\n";
    os << "  Calculate eigenvectors:    " << (settings.calc_eigenvector ? "yes" : "no") << "\n";
    os << "---------------------------\n";
    return os;
}

// a stochastically sampled inner product 
// Useful to check convergence
class ApproximateInner {
    ApproximateInner(std::mt19937& rng, size_t dim, size_t n){
        auto dist = std::uniform_int_distribution<size_t>( 0, dim);
        for (size_t i=0; i<n; i++){
            indices_.push_back(dist(rng));
        }
    }

    template<typename T>
    double distance(const std::vector<T>& u, const std::vector<T>& v){
        T res=0;
        for (auto i : indices_){
            res += std::conj(u[i]) * v[i];
        }
        return res;
    }

private:
    std::vector<size_t> indices_;
};



struct LanczosResult {
    bool converged;
    size_t n_iterations;
    double eigval_error;
    double eigvec_error;
};


// Check convergence of the Lanczos iteration
// Returns true if converged, false otherwise
template<RealOrCplx _S>
void check_lanczos_convergence(
    const std::vector<double>& alphas,
    const std::vector<double>& betas,
    double& eigval,
    size_t iteration,
    const LanczosSettings& settings,
    LanczosResult& res)
{
    std::vector<_S> ritz_tmp;
    std::vector tmp_alphas(alphas);
    std::vector tmp_betas(betas);
    double old_eigval = eigval;

    // calculates the new eigval
    tridiagonalise(tmp_alphas, tmp_betas, eigval, ritz_tmp);
    
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

template<RealOrCplx _S>
void set_random_unit(std::vector<_S>& v, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto& x : v) x = dist(rng);

    double nrm = norm(v);
    for (auto& x : v) x /= nrm;
}


// Runs the Lanczos recurrence
template<typename H, RealOrCplx _S>
requires LinearOperator<H, _S>
LanczosResult lanczos_iterate(const H& ham, 
        std::vector<_S>& v, 
    std::vector<double>& alphas,
    std::vector<double>& betas,
        const LanczosSettings& settings,
        const std::vector<double>* ritz = nullptr,   // optional
        std::vector<_S>* eigvec = nullptr        // optional accumulator
        )
{
    // Generating the starting vector
    v.resize(ham.cols());
    std::mt19937 rng(settings.x0_seed);
    set_random_unit(v, rng);

    // v = current v_j (input normalized)
    // u = scratch/previous/next (rotates role each step)
    std::vector<_S> u(ham.cols());
    
    assert( abs(norm(v) - 1) < settings.abs_tol );


    alphas.reserve(settings.krylov_dim);
    betas.reserve(settings.krylov_dim);

    alphas.resize(0);
    betas.resize(0);
    
    LanczosResult retval;


    // Compared to implementation written in 
    // https://slepc.upv.es/documentation/reports/str5.pdf
    // betas[j] = \beta_{j+2}
    // alphas[j] = \alpha_{j+1}

    double beta = 0.0;
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
        TIMEIT("u += Av ", ham.evaluate_add(v.data(), u.data());)

        // α_j = <v_j | u>
        double alpha = innerReal(v, u);

        // u -= α_j * v_j
        sub(u, v, alpha);

        // β_j = ||u||
        beta = norm(u);
        if (beta < settings.abs_tol) break;

        // rotate: new v = v_{j+1},
        std::swap(v, u);
        // v now contains the un-normalised "u" from before. 
        // u contains the old v_{j} for next iteration
        mul(v, 1/beta);

        if (settings.verbosity > 1) {
            std::cout << "Iter "<< j << " a[j]="<<alpha<<" b[j]="<<beta << "\n";
        }

        alphas.push_back(alpha);
        betas.push_back(beta);

         // Convergence test: compute current eigenvalue estimate
        if (j >= settings.min_iterations && j % settings.convergence_check_interval == 0) {
            check_lanczos_convergence<_S>(alphas, betas, eigval, j, settings, retval);
            if (settings.verbosity > 0) {
                std::cout << "Iter "<< j << " eigval error " << retval.eigval_error << "\n";
            }

            if (retval.converged) {
                break;
            }
        }
    }

    return retval;
}





template<typename H, RealOrCplx _S>
requires LinearOperator<H, _S>
LanczosResult eigval0_lanczos(const H& ham, double& eigval, 
        std::vector<_S>& v, 
        const LanczosSettings& settings = LanczosSettings()
        )
{
    if (settings.verbosity >= 0){
        std::cout << settings;
    }

    std::vector<double> alphas;
    std::vector<double> betas;

    std::cout << "[Lanczos] finding lowest eigenvalue\n";
    LanczosResult res = lanczos_iterate(ham, v, alphas, betas, settings);

    std::cout << "[Lanczos] tridiagonalising in Krylov space\n";
    std::vector<_S> ritz;

    std::vector tmp_alphas(alphas);
    std::vector tmp_betas(betas);
    tridiagonalise(tmp_alphas, tmp_betas, eigval, ritz);

    if(!settings.calc_eigenvector) return res;
                                    
    std::cout << "[Lanczos] iterating to determine eigenvector\n";
    // ----------------------------
    // Second pass: reconstruct eigenvector
    // ----------------------------
    std::vector<_S> evector(ham.cols());
    
    lanczos_iterate(ham, v, alphas, betas, settings,
            &ritz, &evector
            );
    std::swap(evector, v);
    return res;
}




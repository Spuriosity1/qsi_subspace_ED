#include <Eigen/Eigenvalues>
#include "common_bits.hpp"

namespace projED {
namespace lanczos {


struct Settings {
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


struct Result {
    bool converged;
    size_t n_iterations;
    double eigval_error;
    double eigvec_error;
};


inline std::ostream& operator<<(std::ostream& os, const Settings& settings)
{
    os << "Lanczos Algorithm Settings:\n";
    os << "---------------------------\n";
    os << "  Krylov subspace dimension: " << settings.krylov_dim << "\n";
    os << "  Absolute Conv. tolerance:  " << settings.abs_tol << "\n";
    os << "  Relative Conv. tolerance:  " << settings.rel_tol << "\n";
    os << "  Minimum iterations:        " << settings.min_iterations << "\n";
    os << "  Maximum iterations:        " << settings.max_iterations << "\n";
    os << "  Initial vector seed:       " << settings.x0_seed << "\n";
    os << "---------------------------\n";
    os << "  Verbosity level:           " << settings.verbosity << " (0=silent, higher=more output)\n";
    os << "  Calculate eigenvectors:    " << (settings.calc_eigenvector ? "yes" : "no") << "\n";
    os << "---------------------------\n";
    return os;
}


inline std::ostream& operator<<(std::ostream& os, const Result& res)
{
        os << "Lanczos Algorithm Result:\n";
    os << "---------------------------\n";
    os << "  Converged:         " << res.converged << "\n";
    os << "  Iterations:        " << res.n_iterations << "\n";
    os << "  Eigenvalue error:  " << res.eigval_error << "\n";
    os << "  Eigenvector error: " << res.eigvec_error << "\n";
    os << "---------------------------\n";
    return os;
}


// real and complex valued eigval0 routines.
// Note that the dimension of the Hilbert space is inferred from v.
Result eigval0(projED::RealApplyFn f, 
        double& eigval, 
        std::vector<double>& v, 
        const Settings& settings_ = Settings()
        );

Result eigval0(projED::ComplexApplyFn f, double& eigval, 
        std::vector<std::complex<double>>& v, 
        const Settings& settings_ = Settings()
        );


}; // end namespace lanczos
}; // end namespacce projED

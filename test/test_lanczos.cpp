#include <cmath>
#include <iostream>
#include <vector>
#include "operator.hpp"
#include "lanczos.hpp"
#include <argparse/argparse.hpp>
#include <random>

using namespace projED;


struct MockBasis { 
    std::size_t dim_;
    size_t dim() const { 
        return dim_;
    } 
};

template<RealOrCplx T>
struct MockHam {
    MockHam(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& _M,
			const size_t dim) : mat(_M), dim_(dim) {}
    size_t cols() const { return dim_; }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
    void evaluate(const double* in, double* out) const {
        Eigen::Map<const Eigen::VectorXd> vin(in, dim_);
        Eigen::Map<Eigen::VectorXd> vout(out, dim_);
        vout.noalias() = mat * vin;
    }

    void evaluate_add(const double* in, double* out) const {
        Eigen::Map<const Eigen::VectorXd> vin(in, dim_);
        Eigen::Map<Eigen::VectorXd> vout(out, dim_);
        vout.noalias() += mat * vin;
    }

    protected:
    size_t dim_;
};

//helper function
//
void generate_random_Herm(Eigen::MatrixXd& M, std::mt19937& rng){
    // Random Hermitian matrix
    std::normal_distribution<double> dist(0.0, 1.0);
    
    auto dim = M.cols();
    assert(M.cols() == M.rows());

    for (int i = 0; i < dim; i++) {
        for (int j = i; j < dim; j++) {
            double val = dist(rng);
            if (i == j)
                M(i,j) = val;          // diagonal
            else
                M(i,j) = M(j,i) = val; // symmetric
        }
    }
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    argparse::ArgumentParser program("lanczos_test");

    program.add_argument("--dim")
        .help("Matrix dimension")
        .scan<'i', int>()
        .default_value(100);

    program.add_argument("--krylov_dim", "-k")
        .help("Krylov space dimension")
        .scan<'i', int>()
        .default_value(30);

    program.add_argument("--max_iterations", "-M")
        .help("Max iterations before giving up")
        .scan<'i', int>()
        .default_value(5000);

    program.add_argument("--min_iterations", "-M")
        .help("Min iterations")
        .scan<'i', int>()
        .default_value(30);

    program.add_argument("--abs_tol", "-a")
        .help("Lanczos eigval atol e.g. -8 = 1e-8")
        .scan<'i', int>()
        .default_value(-8);

    program.add_argument("--rel_tol", "-r")
        .help("Lanczos eigval rtol e.g. -8 = 1e-8")
        .scan<'i', int>()
        .default_value(-8);

    program.add_argument("--seed")
        .help("Seed for the RNG")
        .scan<'i', unsigned int>()
        .default_value(0u);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << program;
        return 1;
    }

    int dim = program.get<int>("--dim");
    unsigned int seed = program.get<unsigned int>("--seed");

    // The random vector
    std::mt19937 rng(seed);

    Eigen::MatrixXd M(dim, dim);
    generate_random_Herm(M, rng);

    MockHam H(M, dim);


    lanczos::Settings settings;
    settings.krylov_dim = program.get<int>("--krylov_dim");
    settings.abs_tol = pow(10, program.get<int>("--abs_tol"));
    settings.rel_tol = pow(10, program.get<int>("--rel_tol"));

    settings.max_iterations = program.get<int>("--max_iterations");
    settings.min_iterations = program.get<int>("--min_iterations");

    settings.verbosity = 3;
    settings.calc_eigenvector = true;

    using coeff_t = double;
    RealApplyFn evadd = [H](const coeff_t* x, coeff_t* y){
        H.evaluate_add(x, y);
    };
    double eigval_lanczos = 0.0;
    // Output vector
    std::vector<double> v0(dim);
    auto res = lanczos::eigval0(evadd, eigval_lanczos, v0, settings);

    // Exact solution with Eigen
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
    double eigval_exact = solver.eigenvalues()(0);
    auto eigenvector_exact = solver.eigenvectors().col(0);

    std::cout << "Lanczos smallest eigenvalue: " << eigval_lanczos << "\n";
    std::cout << "Exact   smallest eigenvalue: " << eigval_exact << "\n";

    auto err_eigval = std::abs(eigval_lanczos - eigval_exact);

    Eigen::Map<Eigen::VectorXd> v0_eigen(v0.data(), v0.size());

    auto err_eigvec = 1-abs(
            eigenvector_exact.dot(v0_eigen) /(
                v0_eigen.norm() * eigenvector_exact.norm() )
            );

    std::cout << "Eigenvalue error: " << err_eigval <<"\n";
    std::cout << "Eigenvector error: " << err_eigvec <<"\n";

    std::cout <<res;
    if (!res.converged) {
        std::cout << "Test failed: Lanczos exceeded maximum iterations\n";
        return 1;
    } else if (err_eigval > 1e-4 ) {
        std::cout << "Test failed: eigval differs too much from exact result\n";
        return 2;
    } else if (err_eigvec > 1e-4 ) {
        std::cout << "Test failed: eigvec differs too much from exact result\n";
        return 3;
    } else { 
        std::cout << "Test passed\n";
        return 0; 
    }
}

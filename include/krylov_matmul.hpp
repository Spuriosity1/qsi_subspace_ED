#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <stdexcept>
#include <cmath>

// Specialisation of KrlovKit.jl
// https://github.com/Jutho/KrylovKit.jl/blob/a39cc4f88f5f7a0935aae52a60372b1d0825773b/src/matrixfun/expintegrator.jl
//
namespace Krylov {

template<typename T>
concept RealOrCplx = std::floating_point<T> ||
                 (requires { typename T::value_type; } &&
                  std::is_same_v<T, std::complex<typename T::value_type>> &&
                  std::floating_point<typename T::value_type>);


struct KrylovSettings{
    int max_iter;
    int krylov_dim;
    double atol = 1e-10;
    double tol = 1e-12; // requested accuracy per unit time
};

void validate_settings(const KrylovSettings& alg){
    if(alg.max_iter < 10) { throw std::runtime_error("max iter must be >= 10"); }
    if(alg.atol <= 0) { throw std::runtime_error("atol must be > 0"); }
    if(alg.tol <= 0) { throw std::runtime_error("tol  must be > 0"); }
    if(alg.krylov_dim < 2) { throw std::runtime_error("krylov_dim must be >= 2"); }

}


// Evaluates 
// exp(tau * Op) * v
// for some dim(v) * M 'tall' matrix v.
// Op *must* be Hermitian.
template <typename OpType, RealOrCplx Scalar>
Eigen::VectorXd _impl_krylov_integrate(const OpType& Op,
                                    std::vector<Eigen::VectorXd>& vecs,
                                    double tau,
                                    const KrylovSettings& alg){
    validate_settings(alg);
    assert(vecs.size() >= 1);
    Eigen::VectorXd u0 = vecs[0];
    Scalar beta_0 = u0.norm();
    Eigen::VectorXd Au0;
    Op.perform_op(u0.data(), Au0.data());
    size_t num_ops = 1;

    size_t krylov_dim = alg.krylov_dim;

    size_t p = vecs.size() -1;

    Eigen::MatrixX<Scalar> HH(krylov_dim + p + 1, krylov_dim + p + 1);

    // timestep params
    double total_err = 0;
    double eta = alg.tol;
    double sgn = tau >= 0 ? 1 : -1;

    // safety factors
    double delta = 1.2;
    double gamma = 0.8;

    // initial vectors
    double tau_0 = 0;

    throw "not implemented";


}



// Computes exp(tau * H) * v using Lanczos method with Krylov subspace of dim m
template <typename OpType>
Eigen::VectorXd krylov_integrate(const OpType& Op,
                                    const Eigen::VectorXd& v,
                                    double tau,
                                    const KrylovSettings& alg)
{
//    using Scalar = typename OpType::Scalar;
    const int n = Op.rows();
    if (n <= 1)
        throw std::invalid_argument("n must be postive >1");
    if (v.size() != n)
        throw std::invalid_argument("Vector size mismatch with operator");
    
    // Store original norm to scale final result
    double v_norm = v.norm();
    if (v_norm < alg.atol) return v;
    
    Eigen::VectorXd v0 = v;
    v /= v_norm;  // Normalize input vector
                
    return v_norm * _impl_krylov_integrate(Op, v, tau, alg);
}





    std::vector<Eigen::VectorXd> V;
    V.reserve(alg.krylov_dim);
    std::vector<double> alpha;
    std::vector<double> beta_vals;
    
    V.push_back(v0);
    Eigen::VectorXd w(n);
    Op.perform_op(v0.data(), w.data());
    double a = v0.dot(w);
    alpha.push_back(a);
    w = w - a * v0;
    double b = w.norm();
    
    
    beta_vals.push_back(b);  // Store first beta value
    
    for (int j = 1; j < alg.krylov_dim; ++j)
    {
        Eigen::VectorXd vj = w / b;
        V.push_back(vj);
        Op.perform_op(vj.data(), w.data());
        w = w - b * V[j - 1];  // Orthogonalize against previous vector
        a = vj.dot(w);
        alpha.push_back(a);
        w = w - a * vj;        // Orthogonalize against current vector
        b = w.norm();
        
        if (b < tol || j == alg.krylov_dim - 1)  // Break if converged or reached max iterations
            break;
        
        beta_vals.push_back(b);
    }
    
    // Build tridiagonal matrix T
    int k = alpha.size();
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(k, k);
    for (int i = 0; i < k; ++i)
        T(i, i) = alpha[i];
    for (int i = 0; i < k - 1; ++i) {
        T(i, i + 1) = beta_vals[i];
        T(i + 1, i) = beta_vals[i];
    }
    
    // Compute exp(-beta * T) * e1
    Eigen::VectorXd e1 = Eigen::VectorXd::Zero(k);
    e1(0) = 1.0;
    
    // Use eigendecomposition for matrix exponential (more robust than .exp())
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed.");

    Eigen::VectorXd eigenvals = solver.eigenvalues();
    Eigen::MatrixXd eigenvecs = solver.eigenvectors();
    
    // Compute exp(-beta * eigenvals)
    Eigen::VectorXd exp_eigenvals(k);
    for (int i = 0; i < k; ++i) {
        exp_eigenvals(i) = std::exp(-tau * eigenvals(i));
    }
    
    // Reconstruct exp(-beta * T) * e1
    Eigen::VectorXd f = eigenvecs * (exp_eigenvals.asDiagonal() * (eigenvecs.transpose() * e1));
    
    // Reconstruct result from Krylov basis, scaled by original norm
    //Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
    v.setZero();
    for (int i = 0; i < k; ++i)
        v += f(i) * V[i];

    // Estimate error from the last Krylov vector and final beta_k
    double error_estimate = 0.0;
    if (k == m && k > 1) {
        error_estimate = std::abs(beta_vals.back() * f(k - 1)) * v_norm;
    }
    
    return error_estimate;


// Computes exp(-beta * H) * v using Lanczos method with Krylov subspace of dim m
template <typename OpType>
Eigen::VectorXd krylov_expv(const OpType& Op,
                                    const Eigen::VectorXd& v,
                                    double beta,
                                    int m = 30,
                                    double tol = 1e-10)
{
    Eigen::VectorXd tmp = v;
    krylov_expv_inplace(Op, tmp, beta, m, tol);
    return tmp;
}

} // end namespace Kyrlov

#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <stdexcept>
#include <cmath>


// Computes exp(-beta * H) * v using Lanczos method with Krylov subspace of dim m
template <typename OpType>
Eigen::VectorXd krylov_expv(const OpType& Op,
                                    const Eigen::VectorXd& v,
                                    double beta,
                                    int m = 30,
                                    double tol = 1e-10)
{
    using Scalar = typename OpType::Scalar;
    const int n = Op.rows();
    if (v.size() != n)
        throw std::invalid_argument("Vector size mismatch with operator");
    
    // Store original norm to scale final result
    double v_norm = v.norm();
    if (v_norm < tol)
        return Eigen::VectorXd::Zero(n);
    
    Eigen::VectorXd v0 = v / v_norm;  // Normalize input vector
    std::vector<Eigen::VectorXd> V;
    std::vector<double> alpha;
    std::vector<double> beta_vals;
    
    V.push_back(v0);
    Eigen::VectorXd w(n);
    Op.perform_op(v0.data(), w.data());
    double a = v0.dot(w);
    alpha.push_back(a);
    w = w - a * v0;
    double b = w.norm();
    
    // Early termination for 1D Krylov space
    if (b < tol) {
        return v_norm * std::exp(-beta * a) * v0;
    }
    
    beta_vals.push_back(b);  // Store first beta value
    
    for (int j = 1; j < m; ++j)
    {
        Eigen::VectorXd vj = w / b;
        V.push_back(vj);
        Op.perform_op(vj.data(), w.data());
        w = w - b * V[j - 1];  // Orthogonalize against previous vector
        a = vj.dot(w);
        alpha.push_back(a);
        w = w - a * vj;        // Orthogonalize against current vector
        b = w.norm();
        
        if (b < tol || j == m - 1)  // Break if converged or reached max iterations
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
    Eigen::VectorXd eigenvals = solver.eigenvalues();
    Eigen::MatrixXd eigenvecs = solver.eigenvectors();
    
    // Compute exp(-beta * eigenvals)
    Eigen::VectorXd exp_eigenvals(k);
    for (int i = 0; i < k; ++i) {
        exp_eigenvals(i) = std::exp(-beta * eigenvals(i));
    }
    
    // Reconstruct exp(-beta * T) * e1
    Eigen::VectorXd f = eigenvecs * (exp_eigenvals.asDiagonal() * (eigenvecs.transpose() * e1));
    
    // Reconstruct result from Krylov basis, scaled by original norm
    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < k; ++i)
        result += f(i) * V[i];
    
    return v_norm * result;
}

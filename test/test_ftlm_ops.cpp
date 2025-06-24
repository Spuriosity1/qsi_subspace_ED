#include "krylov_matmul.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

// Test operator classes
class IdentityOperator {
public:
    int n;
    using Scalar = double;
    
    IdentityOperator(int size) : n(size) {}
    int rows() const { return n; }
    
    void perform_op(const double* v_data, double* result) const {
        for (int i = 0; i < n; ++i) {
            result[i] = v_data[i];
        }
    }
};

class ScalarOperator {
public:
    int n;
    double scalar;
    using Scalar = double;
    
    ScalarOperator(int size, double s) : n(size), scalar(s) {}
    int rows() const { return n; }
    
    void perform_op(const double* v_data, double* result) const {
        for (int i = 0; i < n; ++i) {
            result[i] = scalar * v_data[i];
        }
    }
};

class TridiagonalOperator {
public:
    int n;
    double a, b, c; // sub-diagonal, diagonal, super-diagonal
    using Scalar = double;
    
    TridiagonalOperator(int size, double sub, double diag, double super) 
        : n(size), a(sub), b(diag), c(super) {}
    int rows() const { return n; }
    
    void perform_op(const double* v_data, double* result) const {
        memset(result, 0, n*sizeof(double));
        
        // First row
        result[0] = b * v_data[0];
        if (n > 1) result[0] += c * v_data[1];
        
        // Middle rows
        for (int i = 1; i < n - 1; ++i) {
            result[i] = a * v_data[i-1] + b * v_data[i] + c * v_data[i+1];
        }
        
        // Last row
        if (n > 1) {
            result[n-1] = a * v_data[n-2] + b * v_data[n-1];
        }
    }
};

class MatrixOperator {
public:
    Eigen::MatrixXd mat;
    using Scalar = double;
    
    MatrixOperator(const Eigen::MatrixXd& m) : mat(m) {}
    int rows() const { return mat.rows(); }
    
    void perform_op(const double* v_data, double* result) const {
        memset(result, 0, mat.rows()*sizeof(double));
        for (int i = 0; i < mat.cols(); ++i) {
            for (int j=0; j<mat.rows(); ++j){
                 result[j] += mat(j,i)*v_data[i];
            }
        }
    }
};

// Test utilities
bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

bool vectors_approx_equal(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol = 1e-8) {
    if (a.size() != b.size()) return false;
    return (a - b).norm() < tol;
}

void print_vector(const Eigen::VectorXd& v, const std::string& name) {
    std::cout << name << ": [";
    for (int i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << v(i);
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Test functions
void test_identity_operator() {
    std::cout << "Testing Identity Operator..." << std::endl;
    
    IdentityOperator I(3);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    double beta = 1.0;
    
    Eigen::VectorXd result = krylov_expv(I, v, beta);
    Eigen::VectorXd expected = v * std::exp(-beta); // exp(-I) * v = exp(-1) * v
    
    assert(vectors_approx_equal(result, expected, 1e-6));
    std::cout << "✓ Identity operator test passed" << std::endl;
}

void test_scalar_operator() {
    std::cout << "Testing Scalar Operator..." << std::endl;
    
    double scalar = 2.0;
    ScalarOperator S(3, scalar);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    double beta = 0.5;
    
    Eigen::VectorXd result = krylov_expv(S, v, beta);
    Eigen::VectorXd expected = v * std::exp(-beta * scalar); // exp(-beta * scalar * I) * v
    
    assert(vectors_approx_equal(result, expected, 1e-6));
    std::cout << "✓ Scalar operator test passed" << std::endl;
}

void test_zero_vector() {
    std::cout << "Testing Zero Vector..." << std::endl;
    
    IdentityOperator I(3);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(3);
    double beta = 1.0;
    
    Eigen::VectorXd result = krylov_expv(I, v, beta);
    Eigen::VectorXd expected = Eigen::VectorXd::Zero(3);
    
    assert(vectors_approx_equal(result, expected));
    std::cout << "✓ Zero vector test passed" << std::endl;
}

void test_single_dimension() {
    std::cout << "Testing Single Dimension..." << std::endl;
    
    ScalarOperator S(1, 3.0);
    Eigen::VectorXd v(1);
    v << 2.0;
    double beta = 0.5;
    
    Eigen::VectorXd result = krylov_expv(S, v, beta);
    Eigen::VectorXd expected(1);
    expected << 2.0 * std::exp(-0.5 * 3.0);
    
    assert(vectors_approx_equal(result, expected, 1e-10));
    std::cout << "✓ Single dimension test passed" << std::endl;
}

void test_tridiagonal_matrix() {
    std::cout << "Testing Tridiagonal Matrix..." << std::endl;
    
    // Test a simple 3x3 tridiagonal matrix
    TridiagonalOperator T(3, -1.0, 2.0, -1.0); // [-1, 2, -1] pattern
    Eigen::VectorXd v(3);
    v << 1.0, 0.0, 0.0;
    double beta = 0.1;
    
    // This should converge quickly since it's a small matrix
    Eigen::VectorXd result = krylov_expv(T, v, beta, 10);
    
    // Verify the result is reasonable (non-zero, finite)
    assert(result.norm() > 1e-10);
    assert(std::isfinite(result.norm()));
    std::cout << "✓ Tridiagonal matrix test passed" << std::endl;
}

void test_large_beta() {
    std::cout << "Testing Large Beta..." << std::endl;
    
    ScalarOperator S(3, 1.0);
    Eigen::VectorXd v(3);
    v << 1.0, 1.0, 1.0;
    double beta = 10.0; // Large beta should make result very small
    
    Eigen::VectorXd result = krylov_expv(S, v, beta);
    
    // exp(-10) is very small
    assert(result.norm() < 1e-3);
    assert(result.norm() > 0.0);
    std::cout << "✓ Large beta test passed" << std::endl;
}

void test_small_beta() {
    std::cout << "Testing Small Beta..." << std::endl;
    
    IdentityOperator I(3);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    double beta = 1e-6; // Very small beta
    
    Eigen::VectorXd result = krylov_expv(I, v, beta);
    
    // Should be very close to original vector
    assert(vectors_approx_equal(result, v, 1e-5));
    std::cout << "✓ Small beta test passed" << std::endl;
}

void test_convergence_with_small_krylov_dim() {
    std::cout << "Testing Small Krylov Dimension..." << std::endl;
    
    ScalarOperator S(5, 1.0);
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    double beta = 1.0;
    
    // Should converge in 1 iteration for scalar operator
    Eigen::VectorXd result = krylov_expv(S, v, beta, 2);
    Eigen::VectorXd expected = v * std::exp(-beta);
    
    assert(vectors_approx_equal(result, expected, 1e-6));
    std::cout << "✓ Small Krylov dimension test passed" << std::endl;
}

void test_exact_small_matrix() {
    std::cout << "Testing Exact Small Matrix..." << std::endl;
    
    // Create a 2x2 matrix we can compute exactly
    Eigen::MatrixXd A(2, 2);
    A << 1.0, 0.5,
         0.5, 1.0;
    
    MatrixOperator M(A);
    Eigen::VectorXd v(2);
    v << 1.0, 0.0;
    double beta = 1.0;
    
    Eigen::VectorXd krylov_result = krylov_expv(M, v, beta, 10);
    
    // Compute exact result using Eigen's matrix exponential
    Eigen::MatrixXd exact_exp = (-beta * A).exp();
    Eigen::VectorXd exact_result = exact_exp * v;
    
    assert(vectors_approx_equal(krylov_result, exact_result, 1e-5));
    std::cout << "✓ Exact small matrix test passed" << std::endl;
}

void test_normalization_preservation() {
    std::cout << "Testing Normalization Preservation..." << std::endl;
    
    IdentityOperator I(3);
    Eigen::VectorXd v(3);
    v << 3.0, 4.0, 0.0; // |v| = 5
    double beta = 0.0; // exp(0) = I, so result should be v
    
    Eigen::VectorXd result = krylov_expv(I, v, beta);
    
    assert(vectors_approx_equal(result, v, 1e-10));
    std::cout << "✓ Normalization preservation test passed" << std::endl;
}

void test_symmetric_matrix() {
    std::cout << "Testing Symmetric Matrix..." << std::endl;
    
    // Create a symmetric matrix for robust testing
    Eigen::MatrixXd A(3, 3);
    A << 2.0, -1.0,  0.0,
        -1.0,  2.0, -1.0,
         0.0, -1.0,  2.0;
    
    MatrixOperator M(A);
    Eigen::VectorXd v(3);
    v << 1.0, 1.0, 1.0;
    double beta = 0.5;
    
    Eigen::VectorXd krylov_result = krylov_expv(M, v, beta, 15);
    
    // Compute exact result
    Eigen::MatrixXd exact_exp = (-beta * A).exp();
    Eigen::VectorXd exact_result = exact_exp * v;
    
    assert(vectors_approx_equal(krylov_result, exact_result, 1e-4));
    std::cout << "✓ Symmetric matrix test passed" << std::endl;
}

void test_vector_size_mismatch() {
    std::cout << "Testing Vector Size Mismatch..." << std::endl;
    
    IdentityOperator I(3);
    Eigen::VectorXd v(5); // Wrong size
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    double beta = 1.0;
    
    bool exception_thrown = false;
    try {
        krylov_expv(I, v, beta);
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }
    
    assert(exception_thrown);
    std::cout << "✓ Vector size mismatch test passed" << std::endl;
}

void test_random_orthogonal_matrix(int n=30) {
    std::cout << "Testing Random Orthogonal Matrix..." << std::endl;
    
    // Create a random orthogonal matrix (preserves norms)
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<> dis(0.0, 1.0);
    
    Eigen::MatrixXd Q(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Q(i, j) = dis(gen);
        }
    }
    
    // QR decomposition to get orthogonal matrix
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Q);
    Q = qr.householderQ();
    
    MatrixOperator M(Q);
    Eigen::VectorXd v(n);
//    for (int i = 0; i < n; ++i) {
//        v(i) = dis(gen);
//    }
    v(0)=1;
    double beta = 0.001;
    
    Eigen::VectorXd krylov_result = krylov_expv(M, v, n, beta, 10);
    Eigen::MatrixXd tmp = -beta * Q;
    Eigen::VectorXd exact_result = tmp.exp() * v;

    
    // For orthogonal matrices, the exponential should also preserve structure
    assert(std::isfinite(krylov_result.norm()));
    assert(krylov_result.norm() > 1e-10);
    std::cout << "Exact: "<<exact_result << std::endl;
    std::cout << "Krylov: "<<krylov_result << std::endl;
    assert(vectors_approx_equal(krylov_result, exact_result, 1e-4));
    std::cout << "✓ Random orthogonal matrix test passed" << std::endl;
}

int main() {
    std::cout << "Running Krylov Exponential Evaluator Tests with Eigen\n";
    std::cout << "=====================================================\n" << std::endl;
    
    try {
        test_identity_operator();
        test_scalar_operator();
        test_zero_vector();
        test_single_dimension();
        test_tridiagonal_matrix();
        test_large_beta();
        test_small_beta();
        test_convergence_with_small_krylov_dim();
        test_exact_small_matrix();
        test_normalization_preservation();
        test_symmetric_matrix();
        test_vector_size_mismatch();
        test_random_orthogonal_matrix();
        
        std::cout << "\n=====================================================";
        std::cout << "\n✅ All tests passed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "\n❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}

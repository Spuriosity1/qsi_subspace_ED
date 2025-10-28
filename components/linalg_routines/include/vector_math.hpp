#pragma once
#include <vector>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace projED {

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
T norm(const std::vector<std::complex<T>>& u) {
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



// does in place c * v
template<typename T>
void mul( std::vector<T>& v, T c) {
    #pragma omp parallel for
    for (std::size_t i = 0; i < v.size(); i++) {
        v[i] *= c;
    }
}



static void tridiagonalise(
    std::vector<double>& alphas, // the diagonal
    std::vector<double>& betas, // the off-diagonal
    std::vector<double>& e, // the (smallest real) eigenvalue storage
    std::vector<double>& v, // the eigenvector storage
    size_t n_eigvals=1, // number of eigenvalues
    const char* which="S" // [S] smallest real, [L] largest real 
    ){
    assert(E.size() >= D.size() -1); // silently ignore off diagonal parts beyond len(D)

    int m = alphas.size();

    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    for (size_t i = 0; i < m; i++) {
        T(i, i) = alphas[i];
        if (i + 1 < m) {
            T(i, i+1) = betas[i];
            T(i+1, i) = betas[i];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);
    e = solver.eigenvalues()(0); // smallest eigenvalue
    // associated eigenvector 
    Eigen::VectorXd ritz = solver.eigenvectors().col(0);
}


}

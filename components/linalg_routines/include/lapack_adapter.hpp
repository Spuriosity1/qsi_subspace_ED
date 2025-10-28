#pragma once
#include <vector>
#include <cassert>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <stdexcept>

//extern "C" {
//    // LAPACK routine, see
//    // https://netlib.org/lapack/explore-html/d4/dec/group__stemr_ga71cbf49a0387762afa39582e6abbe466.html#ga71cbf49a0387762afa39582e6abbe466
//void dstemr_(const char *_Nonnull jobz, const char *_Nonnull range,
//             const LAPACK_int_t *_Nonnull n, 
//             double *_Nullable d,
//             double *_Nullable e,
//             const double *_Nonnull vl, const double *_Nonnull vu,
//             const LAPACK_int_t *_Nonnull il, const LAPACK_int_t *_Nonnull iu,
//             LAPACK_int_t *_Nonnull m,
//             double *_Nullable w, double *_Nullable z,
//             const LAPACK_int_t *_Nonnull ldz,
//             const LAPACK_int_t *_Nonnull nzc,
//             LAPACK_int_t *_Nullable isuppz,
//             LAPACK_bool_t *_Nonnull tryrac,
//             double *_Nonnull work, 
//             const LAPACK_int_t *_Nonnull lwork,
//             LAPACK_int_t *_Nullable iwork,
//             const LAPACK_int_t *_Nonnull liwork,
//             LAPACK_int_t *_Nonnull info);
//
//}


namespace projED {


inline void tridiagonalise(
    std::vector<double>& D,       // diagonal
    std::vector<double>& E,       // off-diagonal
    std::vector<double>& e,       // eigenvalues (output)
    std::vector<double>& v,       // eigenvectors (output, column-major)
    size_t n_eigvals = 1,         // number of eigenvalues (ignored if "A")
    const char* which = "S"       // "S"=smallest, "L"=largest, "A"=all
) {
    const size_t n = D.size();
    if (n == 0) return;
    if (E.size() < n - 1 && n > 1)
        throw std::invalid_argument("E must have size n-1");

    // Build the symmetric tridiagonal matrix
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(n, n);
    for (size_t i = 0; i < n; ++i) mat(i, i) = D[i];
    for (size_t i = 0; i < n - 1; ++i) {
        mat(i, i + 1) = E[i];
        mat(i + 1, i) = E[i];
    }

    // Compute all eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mat);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed");

    Eigen::VectorXd evals = solver.eigenvalues();
    Eigen::MatrixXd evecs = solver.eigenvectors();

    // Select desired eigenvalues/eigenvectors
    size_t start = 0;
    size_t count = n;
    if (*which == 'S' || *which == 's') {
        start = 0;
        count = std::min(n_eigvals, n);
    } else if (*which == 'L' || *which == 'l') {
        start = n - std::min(n_eigvals, n);
        count = std::min(n_eigvals, n);
    } else if (*which == 'A' || *which == 'a') {
        start = 0;
        count = n;
    } else {
        throw std::invalid_argument("Invalid value for 'which'");
    }

    e.resize(count);
    v.resize(n * count);

    for (size_t j = 0; j < count; ++j) {
        e[j] = evals(start + j);
        for (size_t i = 0; i < n; ++i) {
            v[i + j * n] = evecs(i, start + j); // column-major
        }
    }
}


inline void tridiagonalise_one(
    std::vector<double>& D,
    std::vector<double>& E,
    double& e_out,
    std::vector<double>& v,
    const char* which = "S"
) {
    std::vector<double> e_v;
    tridiagonalise(D, E, e_v, v, 1, which);
    e_out = e_v[0];
}


} // end namsepace

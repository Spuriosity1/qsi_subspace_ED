#pragma once
#include <vector>
#include <cassert>

namespace projED {

    // the FORTRAN binding
extern "C" {
    // LAPACK routine
    void dstevr_(
        const char* jobz, const char* range, const int* n,
        double* d, double* e, const double* vl, const double* vu,
        const int* il, const int* iu, const double* abstol, int* m,
        double* w, double* z, const int* ldz, int* isuppz,
        double* work, const int* lwork, int* iwork, const int* liwork, int* info
    );
}




inline void tridiagonalise(
        std::vector<double>& D,               // diagonal
        std::vector<double>& E,               // off-diagonal
        std::vector<double>& e,               // eigenvalues (output)
        std::vector<double>& v,               // eigenvectors (output)
        size_t n_eigvals = 1,                 // number of eigenvalues
        const char* which = "S"               // "S" = smallest, "L" = largest
    ) {
        const int n = static_cast<int>(D.size());
        if (n-1<0 || E.size() < static_cast<size_t>(n - 1))
            throw std::invalid_argument("E must have size n-1");

        if (n == 0)
            return;

        // LAPACK uses Fortran-style ints
        int info, m;
        char jobz = 'V';
        char range = 'I';

        // Select index range of eigenvalues
        int il = 1;
        int iu = static_cast<int>(n_eigvals);
        if (*which == 'L' || *which == 'l') {
            il = n - static_cast<int>(n_eigvals) + 1;
            iu = n;
        }

        double vl = 0.0, vu = 0.0;  // not used for RANGE='I'
        double abstol = 0.0;        // use default tolerance

        // Query optimal workspace sizes
        int lwork = -1, liwork = -1;
        double wkopt;
        int iwkopt;
        std::vector<int> isuppz(2 * std::max(1, n));
        dstevr_(&jobz, &range, &n,
                D.data(), E.data(),
                &vl, &vu, &il, &iu, &abstol, &m,
                nullptr, nullptr, &n, isuppz.data(),
                &wkopt, &lwork, &iwkopt, &liwork, &info);
        if (info != 0)
            throw std::runtime_error("DSTEVR workspace query failed");

        lwork = static_cast<int>(wkopt);
        liwork = iwkopt;

        // Allocate workspace
        std::vector<double> work(lwork);
        std::vector<int> iwork(liwork);

        // Allocate output
        e.resize(m);
        v.resize(n * m);

        // Compute
        dstevr_(&jobz, &range, &n,
                D.data(), E.data(),
                &vl, &vu, &il, &iu, &abstol, &m,
                e.data(), v.data(), &n, isuppz.data(),
                work.data(), &lwork, iwork.data(), &liwork, &info);

        if (info != 0)
            throw std::runtime_error("DSTEVR computation failed");

        // Shrink outputs to actual number of eigenpairs found
        e.resize(m);
        v.resize(static_cast<size_t>(n) * m);
    }


inline void tridiagonalise_one(
        std::vector<double>& D,               // diagonal
        std::vector<double>& E,               // off-diagonal
        double& e,               // eigenvalues (output)
        std::vector<double>& v,               // eigenvectors (output)
        const char* which = "S"               // "S" = smallest, "L" = largest
        ) {
    std::vector<double> e_v(1);
    e_v[0] = e;
    tridiagonalise(D, E, e_v, v, 1, which);
    e = e_v[0];
}



} // end namsepace

#pragma once
#include <vector>
#include <cassert>

#ifdef USE_APPLE_ACCELERATE
#include <vecLib/vecLib.h>
#endif

typedef long int LAPACK_int_t;
typedef long int LAPACK_bool_t;


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

    // the FORTRAN binding



inline void tridiagonalise_mrrr(
        std::vector<double>& D,               // diagonal
        std::vector<double>& E,               // off-diagonal
        std::vector<double>& e,               // eigenvalues (output)
        std::vector<double>& v,               // eigenvectors (output)
        size_t n_eigvals = 1,                 // number of eigenvalues (ignored when 'all' passed)
        const char* which = "S"               // "S" = smallest, "L" = largest, "A" = all
    ) {
        const LAPACK_int_t n = static_cast<LAPACK_int_t>(D.size());
        if (n-1<0 || E.size() < static_cast<size_t>(n - 1))
            throw std::invalid_argument("E must have size n-1");

        if (n == 0)
            return;

        // LAPACK uses Fortran-style ints
        LAPACK_int_t info, m;
        char jobz = 'V';
        char range = 'I';

        if (*which == 'A' || *which == 'a'){
            n_eigvals = n;
            range = 'A';
        }


        // Select index range of eigenvalues
        LAPACK_int_t il = 1;
        LAPACK_int_t iu = static_cast<LAPACK_int_t>(n_eigvals);
        if (*which == 'L' || *which == 'l') {
            il = n - static_cast<LAPACK_int_t>(n_eigvals) + 1;
            iu = n;
        } 

        LAPACK_int_t nzv = n_eigvals; 

        
        LAPACK_int_t tryRAC =1;

        double vl = 0.0, vu = 0.0;  // not used for RANGE='I', 'A'
     
        LAPACK_int_t ldz = n; // leading zeros of calculated eigenvector array
        LAPACK_int_t lwork = -1; // number of places in work arrays
        LAPACK_int_t liwork = -1; // dimension of iwork
        std::vector<LAPACK_int_t> isuppz(2 * std::max(static_cast<LAPACK_int_t>(1), n));
        std::vector<double> work(1);
        std::vector<LAPACK_int_t> iwork(1);

        // Query to get correct work size (controlled by -1 in ldz)
        dstemr_(&jobz, &range, &n,
                D.data(), // IN, OUT the diagonal (length n)
                E.data(), // IN, OUT the off diagonal (length max(1,n-1))
                &vl, &vu, // lower, upper eigval bounds (not referecned if range='A' or 'I')
                &il, &iu, // indices of eigenvalues (ascending order)
                &m, // OUT total number of eigvals found. Guaranteed to be IU - IL+1 here
                nullptr, // OUT eigenvalue array
                nullptr, // OUT eigenvector array (col-major)
                &ldz, // OUT leading dimension of the array v
                &nzv, // NZV, number of eigenvectors to be held in Z
                isuppz.data(), // ISUPPZ is INTEGER array, dimension ( 2*max(1,M) )
                &tryRAC,
                work.data(),
                &lwork,
                iwork.data(),
                &liwork,
                &info
               );

        if (info != 0)
            throw std::runtime_error("DSTEMR workspace query failed");

        lwork = work[0];
        liwork = iwork[0];

        // Allocate workspace
        work.resize(lwork);
        iwork.resize(liwork);

        // Allocate output
        e.resize(m);
        v.resize(n * m);

        // Compute
        dstemr_(&jobz, &range, &n,
                D.data(), // IN, OUT the diagonal (length n)
                E.data(), // IN, OUT the off diagonal (length max(1,n-1))
                &vl, &vu, // lower, upper eigval bounds (not referecned if range='A' or 'I')
                &il, &iu, // indices of eigenvalues (ascending order)
                &m, // OUT total number of eigvals found. Guaranteed to be IU - IL+1 here
                e.data(), // OUT eigenvalue array
                v.data(), // OUT eigenvector array (col-major)
                &ldz, // OUT leading dimension of the array v
                &nzv, // NZV, number of eigenvectors to be held in Z
                isuppz.data(), // ISUPPZ is INTEGER array, dimension ( 2*max(1,M) )
                &tryRAC,
                work.data(),
                &lwork,
                iwork.data(),
                &liwork,
                &info
               );

        if (info != 0)
            throw std::runtime_error("DSTEVR computation failed");
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

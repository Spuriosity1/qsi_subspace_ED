#pragma once
#include <vector>
#include <complex>


#ifdef USE_APPLE_ACCELERATE
#include <version>
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace projED {

typedef std::complex<double> cplx_t;

inline double inner(const std::vector<double>& u, const std::vector<double>& v) {
    return cblas_ddot(u.size(), u.data(), 1, v.data(), 1);
}

// computes conj(u) dot v
inline cplx_t inner(const std::vector<cplx_t>& u, const std::vector<cplx_t>& v) {
    cplx_t res;
    cblas_zdotc_sub(u.size(), u.data(), 1, v.data(), 1, &res);
    return res;
}


// Real part of inner product conj(u) * v

inline double innerReal(const std::vector<double>& u, const std::vector<double>& v) {
    return cblas_ddot(u.size(), u.data(), 1, v.data(), 1);
}
inline double innerReal(const std::vector<cplx_t>& u, const std::vector<cplx_t>& v) {
    double res = 0;
    const double* u_data = reinterpret_cast<const double*>(u.data());
    const double* v_data = reinterpret_cast<const double*>(v.data());
    auto N = u.size();

    res += cblas_ddot(N, u_data, 2, v_data, 2);
    res += cblas_ddot(N, u_data+1, 2, v_data+1, 2);
    return res;
}

inline double norm(const std::vector<double>& u) {
    return sqrt(inner(u,u));
}
inline double norm(const std::vector<cplx_t>& u) {
    return sqrt(innerReal(u,u));
}

// performs u <- u + v * c
inline void axpy( std::vector<double>& u,
                      const std::vector<double>& v,
                      double c) {
    cblas_daxpy(u.size(), c, v.data(), 1, u.data(), 1);
}

// performs u <- u + v * c
inline void axpy( std::vector<cplx_t>& u,
                      const std::vector<cplx_t>& v,
                      double c) {
    double* u_data = reinterpret_cast<double*>(u.data());
    const double* v_data = reinterpret_cast<const double*>(v.data());
    auto N = u.size()*2;
    cblas_daxpy(N, c, v_data, 1, u_data, 1);
}


// does in place c * v
inline void mul( std::vector<double>& v, double c) {
    cblas_dscal(v.size(), c, v.data(), 1);
}
inline void mul( std::vector<cplx_t>& v, double c){
    double* v_data = reinterpret_cast<double*>(v.data());
    cblas_dscal(v.size()*2, c, v_data, 1);
}


}


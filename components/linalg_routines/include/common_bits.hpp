#pragma once
#include <cstdint>
#include <complex>
#include <random>
#include <functional>
#include <complex>


#ifdef DONT_USE_BLAS
#include "blas_fallback/vector_math.hpp"
#else
#include "blas_adapter.hpp"
#include "lapack_adapter.hpp"
#endif

namespace projED {

    // Function pointer types for matrix-vector operations
    // Signature: void apply(const Scalar* in, Scalar* out, void* user_data)
    // Computes: out += A * in
    using RealApplyFn = std::function<void(const double* in, double* out)>;
    using ComplexApplyFn = std::function<void(const std::complex<double>* in, 
            std::complex<double>* out)>;


template<typename _S>
void set_random_unit(std::vector<_S>& v, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto& x : v) x = dist(rng);

    double nrm = norm(v);
    for (auto& x : v) x /= nrm;
}

};



#pragma once
#include "bits.hpp"
#include "blas_adapter.hpp"
#include <mpi.h>
#include <random>

namespace projED {



template<RealOrCplx _S>
void set_random_unit_mpi(std::vector<_S>& local_v, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto& x : local_v) x = dist(rng);
    
    // Compute local norm squared
    double local_norm_sq = innerReal(local_v, local_v);
    
    // Reduce to get global norm squared
    double global_norm_sq = 0.0;
    MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Normalize
    double nrm = std::sqrt(global_norm_sq);
    for (auto& x : local_v) x /= nrm;
}



}

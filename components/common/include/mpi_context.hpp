#pragma once
#include <cassert>
#include <cstdio>
#include <functional>
#include <mpi.h>
#include <complex>
#include <fstream>
#include <random>
#include <unordered_set>
#include "bittools.hpp"
#include <stdexcept>

// MPI datatype helper
template<typename T>
MPI_Datatype get_mpi_type();

template<> inline MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }
template<> inline MPI_Datatype get_mpi_type<unsigned long long>() { return MPI_UNSIGNED_LONG_LONG; }
template<> inline MPI_Datatype get_mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }
template<> inline MPI_Datatype get_mpi_type<unsigned>() { return MPI_UNSIGNED; }
template<> inline MPI_Datatype get_mpi_type<int long long>() { return MPI_LONG_LONG; }
template<> inline MPI_Datatype get_mpi_type<int long>() { return MPI_LONG; }
template<> inline MPI_Datatype get_mpi_type<int>() { return MPI_INT; }
template<> inline MPI_Datatype get_mpi_type<std::complex<double>>() { return MPI_C_DOUBLE_COMPLEX; }
template<> inline MPI_Datatype get_mpi_type<std::complex<float>>() { return MPI_C_FLOAT_COMPLEX; }
template<> inline MPI_Datatype get_mpi_type<Uint128>() {
//inline MPI_Datatype get_mpi_type_uint128() {
    static MPI_Datatype dtype = MPI_DATATYPE_NULL;
    if (dtype == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(2, MPI_UNSIGNED_LONG_LONG, &dtype);
        MPI_Type_commit(&dtype);
    }
    return dtype;
}



struct MPIHashContext {
    using state_t = Uint128;
    using idx_t = int64_t;
    MPIHashContext() {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        char fname[100];
        time_t now = time(nullptr);
        struct tm* utc_time = gmtime(&now);
        char timestamp[20];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H-%M-%SZ", utc_time);
        snprintf(fname, 100, "log_%s_n%d_r%d.log", timestamp, world_size, my_rank);
        log.open(fname);
    }

    // destructor
    ~MPIHashContext(){
        log.close();
    }

    // Copy constructor
    MPIHashContext(const MPIHashContext&) = delete;
    
    // Copy assignment operator
    MPIHashContext& operator=(const MPIHashContext&) = delete;
    
    // Move constructor
    MPIHashContext(MPIHashContext&& other) noexcept
        : world_size(other.world_size),
          my_rank(other.my_rank),
          log(std::move(other.log)) {
    }
    
    // Move assignment operator
    MPIHashContext& operator=(MPIHashContext&& other) noexcept {
        if (this != &other) {
            log.close();
            world_size = other.world_size;
            my_rank = other.my_rank;
            log = std::move(other.log);
        }
        return *this;
    }

    idx_t rank_of_state(state_t psi) const noexcept {
        uint64_t x = psi.uint64[0] ^ psi.uint64[1]; // fold
        x *= 0x9E3779B97F4A7C15ull;           // golden ratio mix
        return x % world_size;
    }

    int world_size;
    int my_rank;

    std::ofstream log;

protected:
    idx_t local_dimension;

};




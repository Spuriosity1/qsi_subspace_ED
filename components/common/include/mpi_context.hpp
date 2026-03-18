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

// Returns resident set size in bytes from /proc/self/status (Linux).
// Returns 0 on platforms that don't support it (e.g. macOS).
inline size_t rss_bytes() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            size_t kb = std::stoull(line.substr(6));
            return kb * 1024;
        }
    }
    return 0;
}

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
        char timestamp[21];
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
        // splitmix64 finalizer: bijective 64-bit→64-bit with full avalanche.
        // Each output bit depends on all input bits, so consecutive sorted
        // ice-rule states (which differ by only a few bits) map to
        // completely uncorrelated ranks.  Applied to each 64-bit half
        // independently before XOR-combining to use the full 128-bit state.
        // The additive offset on the upper half avoids h=0^0 when uint64[1]=0.
        auto mix = [](uint64_t x) -> uint64_t {
            x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
            x ^= x >> 27; x *= 0x94D049BB133111EBULL;
            return x ^ (x >> 31);
        };
        uint64_t h = mix(psi.uint64[0]) ^ mix(psi.uint64[1] + 0x9E3779B97F4A7C15ULL);
        return (idx_t)(h % (uint64_t)world_size);
    }

    int world_size;
    int my_rank;

    std::ofstream log;

protected:
    idx_t local_dimension;

};




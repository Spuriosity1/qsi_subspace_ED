#pragma once
#include <mpi.h>
#include <complex>
#include "bittools.hpp"

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


template <typename state_t, typename idx_t>
struct MPIContext {
    MPIContext(){
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        idx_partition.resize(world_size+1);
        state_partition.resize(world_size+1);
    }

    int world_size;
    int my_rank;
    
    // sorted parallel arrays, both of length num_nodes + 1
    // rank 'n' handles states in interval [ state_partition[n], state_partition[n+1])
    std::vector<state_t> state_partition;
    std::vector<idx_t> idx_partition;

    // naively divides into index sectors
    void build_idx_partition(int64_t n_basis_states){
        if (my_rank == 0){
            std::cout<<"Building index partion with "<<n_basis_states<<" states\n";
            if (world_size > n_basis_states) {
                throw std::runtime_error("Too many nodes for too small a basis!");
            }
        }

        int64_t n_per_rank = n_basis_states / world_size;
        for (int i=0; i<world_size; i++){
            idx_partition[i] = i * n_per_rank;
            if(idx_partition[i] < 0){
                throw std::logic_error("integer underflow in idx_partition");
            }
        }
        idx_partition[world_size] = n_basis_states;
    }


    idx_t block_size(size_t r){
        return idx_partition[r+1] - idx_partition[r];
    }

    idx_t local_start_index() const {
        return idx_partition[my_rank];
    }

    idx_t local_block_size() const {
        return idx_partition[my_rank+1] - idx_partition[my_rank];
    }

    idx_t global_basis_dim() const {
        return idx_partition[world_size];
    }

    // returns the node on which a specified psi can be found
    size_t rank_of_state(state_t psi) const;
    // returns the node on which a specified global index can be found
    size_t rank_of_idx(idx_t J) const;

};




template <typename state_t, typename idx_t>
size_t MPIContext<state_t, idx_t>::rank_of_state(state_t psi) const {
    // linear search, all states should fit in cache unless # nodes is very large
    // IMPORTANT: never checks state_partition[world_size] itself, which may
    // overflow in the 128 site cache
    for (int n=0; n<world_size; n++){
        if (psi < state_partition[n+1]){
            return n;
        }
    }
    // this should almost never happen (possible if we serch for psi nonexistent -- in this case we can safely hand it off to any old node)
    return world_size-1;
}

// returns the rank on which a specified psi can be found
template <typename state_t, typename idx_t>
size_t MPIContext<state_t, idx_t>::rank_of_idx(idx_t J) const {
    // linear search, all states should fit in cache unless # nodes is very large
    for (int n=0; n<world_size; n++){
        if (J < idx_partition[n+1]){
            return n;
        }
    }

    // this should almost never happen (possible if we serch for psi nonexistent -- in this case we can safely hand it off to any old node)
    return world_size-1;
}



class RankLogger {
    const int& rank;
public:
    RankLogger(const int& r) : rank(r) {}

    template<typename T>
    auto& operator<<(const T& value) {
        std::cout << "[rank " << rank << "] " << value;
return std::cout;
    }

    // support std::endl and other manipulators
    auto& operator<<(std::ostream& (*manip)(std::ostream&)) {
        std::cout << manip;
return std::cout;
    }
};

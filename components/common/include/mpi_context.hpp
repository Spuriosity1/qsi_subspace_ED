#pragma once
#include <mpi.h>
#include <complex>
#include "bittools.hpp"

// MPI datatype helper
template<typename T>
MPI_Datatype get_mpi_type();

template<> inline MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }
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
    std::vector<ZBasisBase::state_t> state_partition;
    std::vector<ZBasisBase::idx_t> idx_partition;

    // naively divides into index sectors
    void build_idx_partition(int n_basis_states){
        std::cout<<"Building index partion with "<<n_basis_states<<"states\n";
        if (world_size > n_basis_states) {
            throw std::runtime_error("Too many nodes for too small a basis!");
        }
        hsize_t n_per_rank = n_basis_states / world_size;
        for (int i=0; i<world_size; i++){
            idx_partition[i] = i * n_per_rank;
        }
        idx_partition[world_size] = n_basis_states;
    }


    ZBasisBase::idx_t block_size(size_t r){
        return idx_partition[r+1] - idx_partition[r];
    }

    ZBasisBase::idx_t local_start_index() const {
        return idx_partition[my_rank];
    }

    ZBasisBase::idx_t local_block_size() const {
        return idx_partition[my_rank+1] - idx_partition[my_rank];
    }

    ZBasisBase::idx_t global_basis_dim() const {
        return idx_partition[world_size];
    }

    // returns the node on which a specified psi can be found
    size_t rank_of_state(ZBasisBase::state_t psi) const;
    // returns the node on which a specified global index can be found
    size_t rank_of_idx(ZBasisBase::idx_t J) const;


};

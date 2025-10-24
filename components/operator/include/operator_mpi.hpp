#pragma once
#include "operator.hpp"
#include <mpi.h>


// MPI datatype helper
template<typename T>
MPI_Datatype get_mpi_type();

template<> inline MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }
template<> inline MPI_Datatype get_mpi_type<std::complex<double>>() { return MPI_C_DOUBLE_COMPLEX; }
template<> inline MPI_Datatype get_mpi_type<std::complex<float>>() { return MPI_C_FLOAT_COMPLEX; }

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
    // node 'n' handles states in interval [ state_partition[n], state_partition[n+1])
    std::vector<ZBasisBase::state_t> state_partition;
    std::vector<ZBasisBase::idx_t> idx_partition;

    // naively divides into index sectors
    void build_idx_partition(size_t n_basis_states){
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

    constexpr ZBasisBase::idx_t local_start_index(){
        return idx_partition[my_rank];
    }

    constexpr ZBasisBase::idx_t local_block_size(){
        return idx_partition[my_rank+1] - idx_partition[my_rank];
    }

    constexpr ZBasisBase::idx_t global_basis_dim(){
        return idx_partition[world_size];
    }

    // returns the node on which a specified psi can be found
    size_t node_of_state(ZBasisBase::state_t psi) const;
    // returns the node on which a specified global index can be found
    size_t node_of_idx(ZBasisBase::idx_t J) const;


};


struct MPI_ZBasisBST : public ZBasisBST 
{
	 MPIContext load_from_file(const fs::path& bfile, const std::string& dataset="basis");
};


inline MPI_Datatype get_mpi_type_uint128() {
    static MPI_Datatype dtype = MPI_DATATYPE_NULL;
    if (dtype == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(2, MPI_UNSIGNED_LONG_LONG, &dtype);
        MPI_Type_commit(&dtype);
    }
    return dtype;
}


template<RealOrCplx coeff_t, Basis B>
struct MPILazyOpSum {
    using Scalar = coeff_t;
    explicit MPILazyOpSum(
            const B& local_basis_, const SymbolicOpSum<coeff_t>& ops_,
            MPIContext& context_
            ) : basis(local_basis_), ops(ops_), ctx(context_) {
    }

    MPILazyOpSum operator=(const MPILazyOpSum& other) = delete;

	// Core evaluator 
    // Applies y = A x (sets y=0 first)
	void evaluate(const coeff_t* x, coeff_t* y) const
    {
		std::fill(y, y + basis.dim(), coeff_t(0));
        this->evaluate_add(x, y);
	}

    // Does y += A*x, where y[i] and x[i] are both indexed from the start of the local block
	void evaluate_add(const coeff_t* x, coeff_t* y) const; 
protected:
    void evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const;
    void evaluate_add_off_diag(const coeff_t* x, coeff_t* y) const;


    MPIContext& ctx;
	const B& basis;
	const SymbolicOpSum<coeff_t> ops;
private:

    void inplace_bucket_sort(std::vector<ZBasisBase::state_t>& states,
        std::vector<coeff_t>& c,
        std::vector<int>& bucket_sizes,
        std::vector<int>& bucket_starts
        ) const;

//    std::vector<coeff_t> _scratch_coeff;
//    std::vector<ZBasisBase::state_t> _scratch_states;
};


template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::evaluate_add(const coeff_t* x, coeff_t* y) const {
    evaluate_add_diagonal(x, y);
    evaluate_add_off_diag(x, y);
}



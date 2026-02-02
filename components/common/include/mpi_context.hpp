#pragma once
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



template <typename idx_t>
struct MPIContext {
    MPIContext() {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        idx_partition.resize(world_size+1);

        char fname[100];
        time_t now = time(nullptr);
        struct tm* utc_time = gmtime(&now);
        char timestamp[20];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H%M%SZ", utc_time);
        snprintf(fname, 100, "log_r%d_%s.log", my_rank, timestamp);
        log.open(fname);
    }

    // destructor
    ~MPIContext(){
        log.close();
    }

    // Copy constructor
    MPIContext(const MPIContext&) = delete;
    
    // Copy assignment operator
    MPIContext& operator=(const MPIContext&) = delete;
    
    // Move constructor
    MPIContext(MPIContext&& other) noexcept
        : world_size(other.world_size),
          my_rank(other.my_rank),
          idx_partition(std::move(other.idx_partition)),
          log(std::move(other.log)) {
    }
    
    // Move assignment operator
    MPIContext& operator=(MPIContext&& other) noexcept {
        if (this != &other) {
            log.close();
            world_size = other.world_size;
            my_rank = other.my_rank;
            idx_partition = std::move(other.idx_partition);
            log = std::move(other.log);
        }
        return *this;
    }


    int world_size;
    int my_rank;

    // sorted parallel arrays, both of length num_nodes + 1
    // rank 'n' handles states in interval [ state_partition[n], state_partition[n+1])
    std::vector<idx_t> idx_partition;
    std::ofstream log;

    // Partitions the indices, roughly equal number on each rank
    void partition_indices_equal(int64_t n_basis_states){
        log<<"Building index partion with "<<n_basis_states<<" states\n";
        if (this->world_size > n_basis_states) {
            throw std::runtime_error("Too many nodes for too small a basis!");
        }
        

        int64_t n_per_rank = n_basis_states / this->world_size;
        for (int r=0; r<this->world_size; r++){
            this->idx_partition[r] = r * n_per_rank;
            if(this->idx_partition[r] < 0){
                throw std::logic_error("integer underflow in idx_partition");
            }
        }
        this->idx_partition[this->world_size] = n_basis_states;
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

    // returns the node on which a specified global index can be found
    size_t rank_of_idx(idx_t J) const;

};



// returns the rank on which a specified psi can be found
template <typename idx_t>
size_t MPIContext<idx_t>::rank_of_idx(idx_t J) const {
    // linear search, all states should fit in cache unless # nodes is very large
    for (int n=0; n<world_size; n++){
        if (J < idx_partition[n+1]){
            return n;
        }
    }

    // this should almost never happen (possible if we serch for psi nonexistent -- in this case we can safely hand it off to any old node)
    return world_size-1;
}



template < typename idx_t>
struct SparseMPIContext : public MPIContext<idx_t> {
    using state_t = Uint128;

    SparseMPIContext() : MPIContext<idx_t>() {
        state_partition.resize(this->world_size+1);
        assert(state_partition.size() == this->idx_partition.size());
    }

    // returns the node on which a specified psi can be found
    size_t rank_of_state_slow(const state_t& psi) const;
    size_t rank_of_state(const state_t& psi) const;
    

    void populate_state_terminals(int64_t n_basis_states, 
            const std::function<state_t(uint64_t)>& read_state);

    std::vector<state_t> state_partition;

private:
    int n_bits;
};



template < typename idx_t>
void SparseMPIContext< idx_t>::populate_state_terminals(int64_t n_basis_states, 
        const std::function<state_t(uint64_t)>& read_state)
{
    assert(n_basis_states > this->world_size);
    state_partition.resize(this->world_size+1);

    if (this->my_rank == 0){
        for (int i=0; i<this->world_size; i++){
            state_partition[i] = read_state(this->idx_partition[i]);
        }
        state_partition[this->world_size] = ~Uint128(0);
    }

    // sync state
    MPI_Bcast(state_partition.data(), state_partition.size(), get_mpi_type<Uint128>(), 0, MPI_COMM_WORLD);
    MPI_Bcast(this->idx_partition.data(), this->idx_partition.size(), get_mpi_type<idx_t>(),
            0, MPI_COMM_WORLD);
}


template <typename idx_t>
size_t SparseMPIContext<idx_t>::rank_of_state(const state_t& psi) const {
    // Binary search to find which rank owns this state
    // We're looking for the largest n where state_partition[n] <= psi
    // Equivalently: the smallest n where psi < state_partition[n+1]
    
    int left = 0;
    int right = this->world_size - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        // Check if psi belongs to partition mid
        if (psi < state_partition[mid + 1]) {
            // psi might be in partition mid or earlier
            right = mid;
        } else {
            // psi is definitely after partition mid
            left = mid + 1;
        }
    }
    
    // At this point, left == right, and this is our answer
    // Verify it's valid (optional defensive check)
    if (left < this->world_size && psi < state_partition[left + 1]) {
        return static_cast<size_t>(left);
    }
    
    // Fallback (should be unreachable for valid states)
    return static_cast<size_t>(this->world_size - 1);
}



template < typename idx_t>
auto& operator<<(std::ostream& os, const SparseMPIContext< idx_t>& ctx){
    os<<"Partition scheme:\n index\t state\n";
    for (size_t i=0; i<ctx.idx_partition.size(); i++){
        os<<ctx.idx_partition[i]<<"\t";
        printHex(os, ctx.state_partition[i])<<"\n";
    }
//    os<<"Rank lookup hashmap: "<<ctx.n_hashes()<<" entries";
    return os;
}
        



template < typename idx_t>
size_t SparseMPIContext< idx_t>::rank_of_state_slow(const state_t& psi) const {
    // linear search, all states should fit in cache unless # nodes is very large
    // IMPORTANT: never checks state_partition[world_size] itself, which may
    // overflow in the 128 site cache
    for (int n=0; n<this->world_size; n++){
        if (psi < state_partition[n+1]){
            return n;
        }
    }
    // this should almost never happen (possible if we serch for psi nonexistent -- in this case we can safely hand it off to any old node)
    // Fallback (should be unreachable for valid states)
    return static_cast<size_t>(this->world_size - 1);
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

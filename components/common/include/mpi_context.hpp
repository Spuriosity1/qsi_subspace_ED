#pragma once
#include <functional>
#include <mpi.h>
#include <complex>
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
    MPIContext(){
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        idx_partition.resize(world_size+1);
    }

    int world_size;
    int my_rank;
    
    // sorted parallel arrays, both of length num_nodes + 1
    // rank 'n' handles states in interval [ state_partition[n], state_partition[n+1])
    std::vector<idx_t> idx_partition;

    // Partitions the indices, roughly equal number on each rank
    void partition_indices_equal(int64_t n_basis_states){
        if (this->my_rank == 0){
            std::cout<<"Building index partion with "<<n_basis_states<<" states\n";
            if (this->world_size > n_basis_states) {
                throw std::runtime_error("Too many nodes for too small a basis!");
            }
        }

        int64_t n_per_rank = n_basis_states / this->world_size;
        for (int i=0; i<this->world_size; i++){
            this->idx_partition[i] = i * n_per_rank;
            if(this->idx_partition[i] < 0){
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
    size_t rank_of_state_slow(state_t psi) const;
    size_t rank_of_state(state_t psi) const;
    

    void partition_basis(int64_t n_basis_states, 
            const std::function<state_t(uint64_t)>& read_state);

    std::vector<state_t> state_partition;

    uint64_t hash_state(const state_t& psi) const{
        [[ likely ]] if (n_bits < 64) {
            // can deduce partitioning from top half alone
            return psi.uint64[1] & bit_mask.uint64[1];
        } else {
            return ((psi & bit_mask) >> (128-n_bits)).uint64[0];
        }
    }

    void insert_hashes(const std::vector<uint64_t>&hash_buffer, 
            int64_t displ, int64_t count, int rank){
        for (int i=0; i<count; i++){
            rank_index[hash_buffer[displ + i]] = rank;
        }
    }

private:
    void estimate_optimal_mask(int64_t n_basis_states, 
        const std::function<state_t(uint64_t)>& read_state);


    std::unordered_map<uint64_t, int> rank_index;
    Uint128 bit_mask;
    int n_bits;
};


template < typename idx_t>
void SparseMPIContext< idx_t>::estimate_optimal_mask(int64_t n_basis_states, 
        const std::function<state_t(uint64_t)>& read_state)
{


    // Downsample a random subset of the states
    std::mt19937_64 rng(100);
    auto dist = std::uniform_int_distribution<int64_t>(0,n_basis_states-1);

    const size_t sample_size = std::min(int64_t(1 << 16), n_basis_states);
    const size_t target_size = this->world_size*2;

    std::vector<Uint128> basis_sample(sample_size);

    for (size_t i=0; i<sample_size; i++){
        basis_sample[i] = read_state(dist(rng));
    }

    std::unordered_set<Uint128> seen;

    this->bit_mask = 0;
    n_bits = 0;

    while(n_bits < 128 && seen.size() < target_size ){
        bit_mask >>= 1;
        n_bits++;
        bit_mask |= Uint128{1ull<<63,0};
        seen.clear();

        for (const auto& b : basis_sample) {
            seen.insert(b & bit_mask);
        }
    }

    if (n_bits >= 128){
        throw std::logic_error("Impossible mask state: indexing is broken"); 
        // early exit MUST have been triggered, or else all states are identical
    } 


    std::cout<<"[ rank "<<this->my_rank<<" ] Bit mask " << bit_mask << 
        " (top "<< n_bits <<" bits\n";
}


template < typename idx_t>
void SparseMPIContext< idx_t>::partition_basis(int64_t n_basis_states, 
        const std::function<state_t(uint64_t)>& read_state)
{
    assert(n_basis_states > this->world_size);

    estimate_optimal_mask(n_basis_states, read_state);
    // the job: find terminals that correspond to these 
    // rank 'n' handles states in interval [ state_partition[n], state_partition[n+1])
    //
    // idx_partition must be populated such that 
    // read_state(idx_partition[n]-1) & mask != read_state(idx_partition[n])

    // populate idx_partition with a guess
    this->partition_indices_equal(n_basis_states);
  // Binary search for mask boundaries
    for (int n = 1; n < this->world_size; n++) {
        int64_t initial_guess = this->idx_partition[n];
        state_t target_mask = read_state(initial_guess) & bit_mask;
        
        // Binary search for the first index where (state & mask) == target_mask
        // Search in range [0, initial_guess]
        int64_t left = 0;
        int64_t right = initial_guess;
        int64_t result = initial_guess;
        
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            state_t mid_state = read_state(mid);
            state_t mid_mask = mid_state & bit_mask;
            
            if (mid_mask == target_mask) {
                // Found a match, but search left for the first occurrence
                result = mid;
                right = mid - 1;
            } else if (mid_mask < target_mask) {
                // target_mask is to the right
                left = mid + 1;
            } else {
                // target_mask is to the left
                right = mid - 1;
            }
        }
        
        this->idx_partition[n] = result;
        this->state_partition[n] = read_state(result);
    }
    
    this->state_partition[0] = read_state(0);
    this->state_partition[this->world_size] = ~Uint128(0);



//    // tweak: shift idx_partition by binary search until we hit a mask boundary
//    for (int n=1; n<this->world_size; n++){
//        auto& J = this->idx_partition[n];
//
//        state_t lo = read_state(J-1);
//        state_t hi = read_state(J);
//        while( (lo&bit_mask) == (hi&bit_mask) ){
//            J--;
//            std::swap(lo, hi);
//            lo = read_state(J-1);
//        }
//        this->state_partition[n] = hi;
//    }
//    this->state_partition[0] = read_state(0);
//    // one past the end
//    // !!! will break if we ever see 0xffffffffffffffffffffffffffffffff
//    this->state_partition[this->world_size] = ~Uint128(0);
}


template <typename idx_t>
size_t SparseMPIContext<idx_t>::rank_of_state(state_t psi) const {
    return rank_index.at(hash_state(psi));
}

/*
    // naively divides into index sectors
    void partition_basis_old(int64_t n_basis_states, 
            std::function<state_t(uint64_t)> read_state){

        for (int r = 0; r < this->world_size; ++r) {              
            assert(this->idx_partition[r] < n_basis_states);
            state_partition[r] = read_state(this->idx_partition[r]);
        }

        // one past last: last_state + 1
        Uint128 last = read_state(n_basis_states - 1);
        ++last.uint128;
        state_partition[this->world_size] = last;
    }
*/


template < typename idx_t>
auto& operator<<(std::ostream& os, const SparseMPIContext< idx_t>& ctx){
    os<<"Partition scheme:\n index\t state\n";
    for (size_t i=0; i<ctx.idx_partition.size(); i++){
        os<<ctx.idx_partition[i]<<"\t";
        printHex(std::cout, ctx.state_partition[i])<<"\n";
    }
    return os;
}
        



template < typename idx_t>
size_t SparseMPIContext< idx_t>::rank_of_state_slow(state_t psi) const {
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

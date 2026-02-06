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
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H-%M-%SZ", utc_time);
        snprintf(fname, 100, "log_%s_n%d_r%d.log", timestamp, world_size, my_rank);
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

    std::vector<std::vector<std::pair<int, size_t>>> get_rebalance_plan(const std::vector<int>& curr_work);

private:
    int n_bits;
};



/**
 * Input: curr_work, a world_size-sized vector of the relative 'hardness' of each rank
 * note -> does not need to be normalised.
 *
 * Output: A vector of transfer specifications such that
 *
 * partition[source_rank] = [ (r_d0, n0), (r_d1, n1), (r_d2, n2), ...]
 * where r_d0, r_d1, r_d2 are ascending sequential and sum(nj) = local_block_size
 */
template <typename idx_t>
std::vector<std::vector<std::pair<int, size_t>>> 
SparseMPIContext<idx_t>::get_rebalance_plan(const std::vector<int>& rank_hardness_score){
    // Calculate target distribution based on hardness scores
    size_t total_hardness = std::accumulate(rank_hardness_score.begin(), rank_hardness_score.end(), 0ull);
    size_t total_records = this->global_basis_dim();
    
    std::vector<size_t> target_distribution(this->world_size);
    size_t assigned = 0;
    
    // Proportionally distribute records based on hardness
    for (int i = 0; i < this->world_size - 1; i++) {
        target_distribution[i] = (rank_hardness_score[i] * total_records) / total_hardness;
        assigned += target_distribution[i];
    }
    // Give remainder to last rank to ensure exact total
    target_distribution[this->world_size - 1] = total_records - assigned;
    
    // curr_work now represents the target distribution (how many records each rank should have)
    // But currently each rank has block_size(rank) records
    
    std::vector<std::vector<std::pair<int, size_t>>> partition(this->world_size);
    
    // Two-pointer algorithm to redistribute records
    int source_rank = 0;           // Current source rank we're taking records from
    int dest_rank = 0;             // Current destination rank we're assigning records to
    size_t source_remaining = this->block_size(0);      // Records remaining in current source
    size_t dest_remaining = target_distribution[0];     // Records needed by current destination
    
    while (source_rank < this->world_size && dest_rank < this->world_size) {
        // Transfer the minimum of what's available and what's needed
        size_t transfer_amount = std::min(source_remaining, dest_remaining);
        
        // Add this transfer to the partition
        partition[source_rank].push_back({dest_rank, transfer_amount});
        
        // Update remaining amounts
        source_remaining -= transfer_amount;
        dest_remaining -= transfer_amount;
        
        // Move to next source if current one is exhausted
        if (source_remaining == 0) {
            source_rank++;
            if (source_rank < this->world_size) {
                source_remaining = this->block_size(source_rank);
            }
        }
        
        // Move to next destination if current one is full
        if (dest_remaining == 0) {
            dest_rank++;
            if (dest_rank < this->world_size) {
                dest_remaining = target_distribution[dest_rank];
            }
        }
    }
    
#ifndef NDEBUG
    // Verify that each rank's partition sums to block_size
    for (int rank = 0; rank < this->world_size; rank++) {
        idx_t acc = 0;
        for (auto& [r, x] : partition[rank]) {
            acc += x;
        }
        assert(acc == this->block_size(rank) && "Partition sum must equal block_size");
    }
    
    // Additional verification: check that destinations are sequential
    for (int rank = 0; rank < this->world_size; rank++) {
        for (size_t i = 1; i < partition[rank].size(); i++) {
            assert(partition[rank][i].first >= partition[rank][i-1].first && 
                   "Destination ranks must be ascending sequential");
        }
    }
#endif

    if (this->my_rank ==0){
        std::cout<<"[Main] Global partition plan\n";
        for (int r=0; r<this->world_size; r++){
            std::cout<<"[source rank "<<r<<"]\n";
            for (const auto& [dest_rank, n] : partition[r]){
                std::cout << "("<<n<<") -> "<<dest_rank<<"\n";
            }
        }
    }
    
    return partition;
}



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
    size_t J;

    const __uint128_t* arr = reinterpret_cast<const __uint128_t*>(state_partition.data());

    static const int64_t CACHE_SIZE=32;

    while (right - left > CACHE_SIZE){
        size_t mid = (left + right) /2;
        if (arr[mid] < psi.uint128) left = mid + 1;
        else right =mid;
    }

    for (J=left; J<right; ++J){
        // arr is strict-ascending
        // first arr[J+1] > psi is the winner
        if (arr[J+1]>psi.uint128) return J;
    }



    return J;
    
//    while (left < right) {
//        int mid = left + (right - left) / 2;
//        
//        // Check if psi belongs to partition mid
//        if (psi < state_partition[mid + 1]) {
//            // psi might be in partition mid or earlier
//            right = mid;
//        } else {
//            // psi is definitely after partition mid
//            left = mid + 1;
//        }
//    }
//    
//    // At this point, left == right, and this is our answer
//    // Verify it's valid (optional defensive check)
//    if (left < this->world_size && psi < state_partition[left + 1]) {
//        return static_cast<size_t>(left);
//    }
    
//    // Fallback (should be unreachable for valid states)
//    return static_cast<size_t>(this->world_size - 1);
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

#pragma once
#include "bittools.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include "tetra_graph_io.hpp"
#include <array>
#include <cstdio>
#include <thread>
#include <queue>
#include <stack>
#include <vector>
#include <mutex>


struct vtree_node_t {
	Uint128 state_thus_far;
	unsigned curr_spin;
	unsigned num_spinon_pairs;
	// curr_spin is the bit ID of the rightmost unknown spin
	// i.e. (1<<curr_spin) & state_thus_far is guaranteed to be 0
};


typedef std::array<int, 4> global_sz_sector_t;

struct lat_container {
	lat_container(const lattice& _lat, unsigned num_spinon_pairs)
		: num_spinon_pairs(num_spinon_pairs), lat(_lat) {
			//auto natoms = data.at("atoms").size();
			auto natoms = _lat.spins.size();
			masks.resize(natoms+1);
			for (size_t i = 0; i < natoms+1; i++) {
				masks[i] = make_mask(i);
			}
		}

        // state is only initialised up to (but not including) bit 1<<idx
	// returns possible states of state&(1<<idx)

	// return values:
	// 0b00 -> no spin state valid
	// 0b01 -> spin down (0) state valid
	// 0b10 -> spin up (1) state valid
	// 0b11 -> both up and down valid
	char possible_spin_states(const vtree_node_t& curr) const;
	//char possible_spin_states(const Uint128& state, unsigned idx) const ;

	const unsigned num_spinon_pairs;
	protected:
	//global_sz_sector_t global_sz_sector;
	std::vector<Uint128> masks; // bitmasks filled by make_mask

	template <typename Container>
	void fork_state_impl(Container& to_examine, vtree_node_t curr); 

	void fork_state(std::stack<vtree_node_t>& to_examine);
	void fork_state(std::queue<vtree_node_t>& to_examine);
	const lattice& lat;
};



struct pyro_vtree : public lat_container {
	pyro_vtree(const lattice& lat, unsigned num_spinon_pairs) :
		lat_container(lat, num_spinon_pairs) {
			is_sorted = false;
		}

	void build_state_tree();
	void sort();
	// Applies bittools::permute to all elements of the basis
	void permute_spins(const std::vector<size_t>& perm);

	void write_basis_csv(const std::string &outfilename); 
    void write_basis_hdf5(const std::string& outfile);
protected:
	void save_state(const Uint128& state) {
			state_list.push_back(state);
	}
	// Repository of ice states for perusal
	std::vector<Uint128> state_list;

	bool is_sorted;

	// auxiliary variable for printing
	unsigned counter = 0;
};



// Thread-safe work-stealing queue for load balancing
class WorkStealingQueue {
private:
    std::deque<vtree_node_t> deque_;
    mutable std::mutex mutex_;
    
public:
    // Default constructor
    WorkStealingQueue() = default;
    
    // Move constructor
    WorkStealingQueue(WorkStealingQueue&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mutex_);
        deque_ = std::move(other.deque_);
    }
    
    // Move assignment
    WorkStealingQueue& operator=(WorkStealingQueue&& other) noexcept {
        if (this != &other) {
            std::lock(mutex_, other.mutex_);
            std::lock_guard<std::mutex> lock1(mutex_, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.mutex_, std::adopt_lock);
            deque_ = std::move(other.deque_);
        }
        return *this;
    }
    
    // Delete copy constructor and assignment (non-copyable due to mutex)
    WorkStealingQueue(const WorkStealingQueue&) = delete;
    WorkStealingQueue& operator=(const WorkStealingQueue&) = delete;
    void push_back(const vtree_node_t& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        deque_.push_back(item);
    }
    
    void push_batch_back(const std::vector<vtree_node_t>& items) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& item : items) {
            deque_.push_back(item);
        }
    }
    
    bool try_pop_back(vtree_node_t& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (deque_.empty()) {
            return false;
        }
        item = deque_.back();
        deque_.pop_back();
        return true;
    }
    
    bool try_steal_front(vtree_node_t& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (deque_.empty()) {
            return false;
        }
        item = deque_.front();
        deque_.pop_front();
        return true;
    }
    
    size_t try_steal_batch_front(std::vector<vtree_node_t>& items, size_t max_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = std::min(max_count, deque_.size() / 2); // steal half
        items.clear();
        items.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            items.push_back(deque_.front());
            deque_.pop_front();
        }
        return count;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return deque_.size();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return deque_.empty();
    }
};


struct pyro_vtree_parallel : public lat_container {
    pyro_vtree_parallel(const lattice &lat, unsigned num_spinon_pairs, 
                       unsigned n_threads = std::thread::hardware_concurrency())
        : lat_container(lat, num_spinon_pairs), n_threads(n_threads) {
        is_sorted = false;
        if (n_threads == 0) n_threads = 1;
        
        // Initialize thread-local storage
        state_set.resize(n_threads);
        job_stacks.resize(n_threads);
        work_stealing_queues.resize(n_threads);
        work_stealing_queues.resize(n_threads);
        for (unsigned i = 0; i < n_threads; ++i) {
            work_stealing_queues[i] = std::make_unique<WorkStealingQueue>();
        }
        counters.resize(n_threads, 0);
        threads.reserve(n_threads);
        std::cout<<"Parallel mode | " << n_threads << " threads\n";
    }
    
    void build_state_tree();
    void sort();
    void permute_spins(const std::vector<size_t>& perm);
    void write_basis_csv(const std::string& outfilename);
    void write_basis_hdf5(const std::string& outfile);
    
protected:
    void build_state_dfs(std::stack<vtree_node_t> &node_stack, unsigned thread_id,
                        unsigned long max_stack_size = (1ul << 40));
    void build_state_bfs(std::queue<vtree_node_t>& node_stack, 
                        unsigned long max_queue_len);
    
    // Enhanced DFS with work stealing
    void build_state_dfs_work_stealing(unsigned thread_id, 
                                     std::atomic<bool>& all_done,
                                     unsigned long max_stack_size = (1ul << 40));
    
    // Work stealing helper functions
    bool try_steal_work(unsigned thread_id);
    void share_work_if_needed(unsigned thread_id, size_t threshold = 100);
    bool has_work_available(unsigned exclude_thread_id) const;
    
    unsigned n_threads;
    bool is_sorted;
    
    size_t n_states() const {
        size_t acc = 0;
        for (const auto& v : state_set) {
            acc += v.size();
        }
        return acc;
    }
    
    // Thread-local storage for cache-friendliness
    std::vector<std::vector<Uint128>> state_set;          // first index is thread ID
    std::vector<std::thread> threads;
    std::vector<std::stack<vtree_node_t>> job_stacks;     // thread-local work stacks
    
    // Work-stealing infrastructure
    std::vector<std::unique_ptr<WorkStealingQueue>> work_stealing_queues;  // per-thread work-stealing queues
    
    // Debug and monitoring
    std::vector<unsigned> counters;
    
    // Constants for work stealing
    static constexpr size_t WORK_STEAL_THRESHOLD = 100;
    static constexpr size_t WORK_STEAL_BATCH_SIZE = 20;
    static constexpr size_t MIN_WORK_TO_SHARE = 50;
};





/*
struct pyro_vtree_parallel : public lat_container {
	pyro_vtree_parallel(const lattice &lat, unsigned num_spinon_pairs, 
			unsigned n_threads = 1)
		: lat_container(lat, num_spinon_pairs), n_threads(n_threads) {
		is_sorted = false;
		}

	void build_state_tree();
	void sort();

	// Applies bittools::permute to all elements of the basis
	void permute_spins(const std::vector<size_t>& perm);

	void write_basis_csv(const std::string& outfilename);
	void write_basis_hdf5(const std::string& outfile);


protected:
	void _build_state_dfs(std::stack<vtree_node_t> &node_stack, unsigned thread_id,
			unsigned long max_stack_size = (1ul << 40));
	void _build_state_bfs(std::queue<vtree_node_t>& node_stack, 
		unsigned long max_queue_len);
	unsigned n_threads;

	bool is_sorted;

	size_t n_states() const {
		size_t acc=0;
		for (auto v : state_set){
			acc += v.size();
		}
		return acc;
	}

	// first index is the thread ID
	std::vector<std::vector<Uint128>> state_set;
	std::vector<std::thread> threads;
	std::vector<std::stack<vtree_node_t>> job_stacks;

	// auxiliary, for debug only
	std::vector<unsigned> counters = {0};
};
*/






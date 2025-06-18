#include "pyro_tree.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <thread>
#include "bittools.hpp"
#include "vanity.hpp"
#include "basis_io.hpp"

#include <hdf5.h>
#include <cinttypes>
#include <barrier>
#include <future>


// LOGIC
char lat_container::possible_spin_states(const vtree_node_t& curr) const {
	// state is only initialised up to (but not including) bit 1<<idx
	// returns possible states of state&(1<<idx)
	const Uint128& state = curr.state_thus_far;
	unsigned idx=curr.curr_spin;


	// return values:
	// 0b00 -> no spin state valid
	// 0b01 -> spin down (0) state valid
	// 0b10 -> spin up (1) state valid
	// 0b11 -> both up and down valid
	char res=0b11;

	Uint128 state_new = state; // new spin is already a 0
#ifdef DEBUG
	assert( !readbit(state, idx) );
	assert( idx < lat.spins.size() );
#endif
	//const auto known_mask = make_mask(idx);
	const auto& known_mask = this->masks[idx];

	// iterate over the two possible states of state&(1<<idx)
	for (__uint128_t updown=0; updown<2; updown++){
		if (updown == 1){
			or_bit(state_new, idx);
		}

		auto t = lat.spins[idx].tetra_neighbours[0];
			// calculate the partial tetra charges
			// NOTE: state_new is all zeros for bits > idx
			int Q = popcnt_u128( state_new & t->bitmask );
			// we know the state of all previous bits, and the one we just set
			int num_known_spins = popcnt_u128( t->bitmask & known_mask )+1;

			int num_spins = t->member_spin_ids.size();
			int num_unknown_spins = num_spins - num_known_spins;

			if (Q + num_unknown_spins < t->min_spins_up || Q > t->max_spins_up){
				// Q is inconsistent with an ice rule
				res &= ~(1<<updown);
				continue; // no point checking the other tetra
			}


		t = lat.spins[idx].tetra_neighbours[1];
			// calculate the partial tetra charges
			// NOTE: state_new is all zeros for bits > idx
			Q = popcnt_u128( state_new & t->bitmask );
			// we know the state of all previous bits, and the one we just set
			num_known_spins = popcnt_u128( t->bitmask & known_mask )+1;

			num_spins = t->member_spin_ids.size();
			num_unknown_spins = num_spins - num_known_spins;

			if (Q + num_unknown_spins < t->min_spins_up || Q > t->max_spins_up){
				// Q is inconsistent with an ice rule
				res &= ~(1<<updown);
			}
	}
	return res;
}



// Attempts to generate the two next configurations and add them to the queue
template <typename Container>
void lat_container::fork_state_impl(Container& to_examine, vtree_node_t curr) {
    char poss_states = this->possible_spin_states(curr);
	bool may_create_pair = (curr.num_spinon_pairs < this->num_spinon_pairs);
    if (poss_states & 0b01) { // 0 is allowed
        auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1, curr.num_spinon_pairs});
        to_examine.push(tmp);
    } else if (may_create_pair) {
        auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1, curr.num_spinon_pairs+1});
        to_examine.push(tmp);
	}

	if (poss_states & 0b10) { // 1 is allowed
        auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1, curr.num_spinon_pairs});
        or_bit(tmp.state_thus_far, curr.curr_spin);
        to_examine.push(tmp);
    } else if (may_create_pair) {
        auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1, curr.num_spinon_pairs+1});
        or_bit(tmp.state_thus_far, curr.curr_spin);
        to_examine.push(tmp);
	}
}

void lat_container::fork_state(cust_stack& to_examine) {
    auto curr = to_examine.top();
	to_examine.pop();
    fork_state_impl(to_examine, curr);
}


void lat_container::fork_state(std::queue<vtree_node_t>& to_examine) {
    auto& curr = to_examine.front();
    fork_state_impl(to_examine, curr);
	to_examine.pop();
}


void pyro_vtree::build_state_tree(){
	cust_stack to_examine;
	// seed the root node
	to_examine.push(vtree_node_t({0,0,0}));

	while(!to_examine.empty()){
#if VERBOSITY > 1
		counter++;
		if (counter%100 == 0){
			printf("stack size %lu\n", to_examine.size());
		}
#endif
#if VERBOSITY > 2
		printf("State %016llx; spin_idx %d, queue size %lu\n", curr.state_thus_far.uint128, curr.curr_spin, to_examine.size());
#endif
		if (to_examine.top().curr_spin == lat.spins.size()){
			state_list.push_back(to_examine.top().state_thus_far);
			to_examine.pop();
		} else {
			fork_state(to_examine);
		}
	}
}

void pyro_vtree::sort(){
	if (this->is_sorted) return;
	std::sort(state_list.begin(), state_list.end());
	this->is_sorted = true;
}


// Simple k-way merge using a min-heap (sequential, but efficient)
template <typename T>
std::vector<T> k_way_merge(const std::vector<std::vector<T>>& chunks) {
    using pair = std::pair<T, std::pair<size_t, size_t>>;  // value, {chunk_idx, idx_in_chunk}
    std::priority_queue<pair, std::vector<pair>, std::greater<>> min_heap;

    // Init heap with first element of each chunk
    for (size_t i = 0; i < chunks.size(); ++i) {
        if (!chunks[i].empty()) {
            min_heap.emplace(chunks[i][0], std::make_pair(i, 0));
        }
    }

    std::vector<T> result;
    while (!min_heap.empty()) {
        auto [val, idx] = min_heap.top(); min_heap.pop();
        auto [chunk_idx, elem_idx] = idx;
        result.push_back(val);
        if (elem_idx + 1 < chunks[chunk_idx].size()) {
            min_heap.emplace(chunks[chunk_idx][elem_idx + 1], std::make_pair(chunk_idx, elem_idx + 1));
        }
    }
    return result;
}

// Memory-optimized k-way merge for Uint128 (memory IO bottlenecked)
template <typename T>
std::vector<T> k_way_merge_optimized(std::vector<std::vector<T>>& chunks) {
    // Remove empty chunks and count total elements
    size_t total_size = 0;
    auto write_pos = chunks.begin();
    for (auto read_pos = chunks.begin(); read_pos != chunks.end(); ++read_pos) {
        if (!read_pos->empty()) {
            total_size += read_pos->size();
            if (write_pos != read_pos) {
                *write_pos = std::move(*read_pos);
            }
            ++write_pos;
        }
    }
    chunks.erase(write_pos, chunks.end());
    
    if (chunks.empty()) return {};
    if (chunks.size() == 1) return std::move(chunks[0]);
    
    // Pre-allocate result - critical for memory IO performance
    std::vector<T> result;
    result.reserve(total_size);
    
    // Simple array-based min-heap for minimal memory overhead
    // Store: {chunk_index, element_index} - value lookup is chunks[chunk_idx][elem_idx]
    struct HeapNode {
        uint32_t chunk_idx;
        uint32_t elem_idx;
    };
    
    std::vector<HeapNode> heap;
    heap.reserve(chunks.size());
    
    // Initialize with first element from each chunk
    for (uint32_t i = 0; i < chunks.size(); ++i) {
        heap.push_back({i, 0});
    }
    
    // Comparator that dereferences to actual values
    auto compare = [&chunks](const HeapNode& a, const HeapNode& b) {
        return chunks[a.chunk_idx][a.elem_idx] > chunks[b.chunk_idx][b.elem_idx];
    };
    
    std::make_heap(heap.begin(), heap.end(), compare);
    
    // Main merge loop - optimized for memory bandwidth
    while (!heap.empty()) {
        // Get minimum element info
        HeapNode min_node = heap.front();
        std::pop_heap(heap.begin(), heap.end(), compare);
        heap.pop_back();
        
        // Move the actual value (avoid extra copy for 128-bit values)
        result.emplace_back(std::move(chunks[min_node.chunk_idx][min_node.elem_idx]));
        
        // Advance to next element in the same chunk
        if (++min_node.elem_idx < chunks[min_node.chunk_idx].size()) {
            heap.push_back(min_node);
            std::push_heap(heap.begin(), heap.end(), compare);
        }
    }
    
    return result;
}

void pyro_vtree_parallel::sort() {
    if (this->is_sorted) return;
    
    // Quick check: if we only have one chunk, just sort it
    if (state_set.size() == 1) {
        std::sort(state_set[0].begin(), state_set[0].end());
        this->is_sorted = true;
        return;
    }
    
    // Remove empty chunks first to avoid unnecessary work
    state_set.erase(
        std::remove_if(state_set.begin(), state_set.end(),
                      [](const auto& chunk) { return chunk.empty(); }),
        state_set.end()
    );
    
    if (state_set.empty()) {
        this->is_sorted = true;
        return;
    }
        
    // Sort chunks in parallel with better thread management
    if (state_set.size() <= n_threads) {
        // Each chunk gets its own thread
        std::vector<std::future<void>> futures;
        futures.reserve(state_set.size());
        
        for (auto& chunk : state_set) {
            futures.push_back(std::async(std::launch::async, [&chunk]() {
                if (!chunk.empty()) {
                    std::sort(chunk.begin(), chunk.end());
                }
            }));
        }
        
        // Wait for all sorting to complete
        for (auto& future : futures) {
            future.wait();
        }
    } else {
        // More chunks than threads - use thread pool approach
        std::atomic<size_t> chunk_index{0};
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        
        for (size_t i = 0; i < n_threads; ++i) {
            threads.emplace_back([&]() {
                size_t idx;
                while ((idx = chunk_index.fetch_add(1)) < state_set.size()) {
                    if (!state_set[idx].empty()) {
                        std::sort(state_set[idx].begin(), state_set[idx].end());
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    // Merge sorted chunks efficiently
    auto merged_result = k_way_merge_optimized(state_set);
    
    // Replace state_set with merged result efficiently
    state_set.clear();
    state_set.emplace_back(std::move(merged_result));
    
    // Ensure we have the expected structure
    assert(std::is_sorted(state_set[0].begin(), state_set[0].end()));
    this->is_sorted = true;
}



void pyro_vtree_parallel::
_build_state_bfs(std::queue<vtree_node_t>& node_stack, 
		unsigned long max_queue_len){
	if (state_set.size() == 0){ state_set.resize(1); }
	while (!node_stack.empty() && node_stack.size() < max_queue_len){
		if (node_stack.front().curr_spin == lat.spins.size()){
			state_set[0].push_back(node_stack.front().state_thus_far);
			node_stack.pop();
		} else {
			fork_state(node_stack);
		}
	}
}

void pyro_vtree_parallel::
_build_state_dfs(cust_stack& node_stack, 
		unsigned thread_id, 
		unsigned long max_queue_len){
	while (!node_stack.empty() && node_stack.size() < max_queue_len){
#if VERBOSITY > 1
		counters[thread_id]++;
		if (counters[thread_id]%100 == 0){
			printf("Thread %u stack size %lu\n", thread_id, node_stack.size());
		}
#endif
		if (node_stack.top().curr_spin == lat.spins.size()){
			state_set[thread_id].push_back(node_stack.top().state_thus_far);
			node_stack.pop();
		} else {
			fork_state(node_stack);
		}
	}
}

// the complicated parallel code

template <typename StackT>
void pyro_vtree_parallel::rebalance_stacks(std::vector<StackT>& stacks) {

    std::vector<vtree_node_t> all_jobs;

    for (auto& stack : stacks) {
        printf("%4lu | ", stack.size());
        while (!stack.empty()) {
            all_jobs.push_back(stack.top());
            stack.pop();
        }
    }

    printf("\n");

    size_t i = 0;
    for (auto& job : all_jobs) {
        stacks[i++ % stacks.size()].push(job);
    }

}

template <typename T>
bool any_full(const std::vector<T>& job_stacks){
    for (auto& s : job_stacks){
        if (!s.empty()) return true;
    }
    return false;
}

void pyro_vtree_parallel::build_state_tree(){
	// strategy: fork nodes until we exceed the thread pool	
	// BFS to keep the layer of all threads roughly the same
	std::queue<vtree_node_t> starting_nodes;
	starting_nodes.push(vtree_node_t({0,0,0}));
	_build_state_bfs(starting_nodes, n_threads*INITIAL_DEPTH_FACTOR);

	n_threads = std::min(n_threads, static_cast<unsigned>(starting_nodes.size()));
	printf("Set up execution with %u threads\n", n_threads);

	// now farm out to different threads
	state_set.resize(n_threads);
	job_stacks.resize(n_threads);

	std::cout<<starting_nodes.size()<<" starting states.\n";
    unsigned thread_id=0;
    while(!starting_nodes.empty()){
		job_stacks[thread_id].push(starting_nodes.front());
		starting_nodes.pop();
        thread_id = (thread_id+1) % n_threads;
    }
    
#ifdef DEBUG 
    const int CHECKIN_INTERVAL=5;
#else 
    const int CHECKIN_INTERVAL=5000000;
#endif

    

    std::atomic<bool> work_to_do{true};


    auto on_completion = [this, &work_to_do]() noexcept
    { 
        rebalance_stacks(job_stacks);
        work_to_do = any_full(job_stacks);
    };
    std::barrier sync_point(n_threads, on_completion);

    for (unsigned tid = 0; tid < n_threads; ++tid) {
        threads.emplace_back([this, tid, &sync_point, &work_to_do]() {
            auto& local_stack = job_stacks[tid];
            while(work_to_do) {
                size_t counter;
                for (counter=0; 
                    !local_stack.empty() && counter < CHECKIN_INTERVAL;
                    counter++) 
                {
                    if (local_stack.top().curr_spin == lat.spins.size()) {
                        state_set[tid].push_back(local_stack.top().state_thus_far);
                        local_stack.pop();
                    } else {
                        fork_state(local_stack);
                    }
                }

//                std::cout << "[" << tid << "] processed " << counter <<"\n";
                sync_point.arrive_and_wait();
            } 
        });
    };

/*
    // Synchronization variables
    std::atomic<unsigned> threads_at_barrier{0};
    std::atomic<bool> work_complete{false};
    std::mutex barrier_mutex;
    std::condition_variable barrier_cv;
    
    for (unsigned tid = 0; tid < n_threads; ++tid) {
        threads.emplace_back([this, tid, &threads_at_barrier, &work_complete, &barrier_mutex, &barrier_cv]() {
            auto& local_stack = job_stacks[tid];
            //const int CHECKIN_INTERVAL=500000;
            const int CHECKIN_INTERVAL=5;
            
            while (!work_complete.load()) {
                // Phase 1: Process work
                int work_done = 0;
                while (!local_stack.empty() && work_done < CHECKIN_INTERVAL && !work_complete.load()) {
                    if (local_stack.top().curr_spin == lat.spins.size()) {
                        state_set[tid].push_back(local_stack.top().state_thus_far);
                        local_stack.pop();
                    } else {
                        fork_state(local_stack);
                    }
                    work_done++;
                }
                
                // Early exit if work is complete
                if (work_complete.load()) break;
                
               // std::cout << "Thread " << tid << " processed " << work_done 
                 //        << " items, stack size now: " << local_stack.size() << std::endl;
                
                // Phase 2: Synchronization barrier
                {
                    std::unique_lock<std::mutex> lock(barrier_mutex);
                    
                    unsigned arrived = ++threads_at_barrier;
                   // std::cout << "Thread " << tid << " at barrier (" << arrived << "/" << n_threads << ")" << std::endl;
                    
                    if (arrived == n_threads) {
                        // Last thread - check for completion and rebalance
                        bool any_work = false;
                        for (const auto& stack : job_stacks) {
                            if (!stack.empty()) {
                                any_work = true;
                                break;
                            }
                        }
                        
                        if (!any_work) {
                            std::cout << "Thread " << tid << " detected completion" << std::endl;
                            work_complete.store(true);
                        } else {
                            std::cout << "Thread " << tid << " rebalancing stacks" << std::endl;
                            rebalance_stacks(job_stacks);
                        }
                        
                        // Reset barrier and wake all threads
                        threads_at_barrier.store(0);
                        barrier_cv.notify_all();
                        
                    } else {
                        // Wait for all threads to reach barrier
                        barrier_cv.wait(lock, [&threads_at_barrier] {
                            return threads_at_barrier.load() == 0;
                        });
                    }
                }
                
                // Check completion status after barrier
                if (work_complete.load()) {
                    std::cout << "Thread " << tid << " exiting" << std::endl;
                    break;
                }
            }
        });
    }
    */
    
    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();
    }

	// make sure the queue was cleared
	for (auto& q : job_stacks){
		if (!q.empty()){
			throw std::logic_error("Queue not cleared, likely exceeded iteration limit");
		}
	}

}


// IO

void pyro_vtree::write_basis_csv(const std::string &outfilename) {
	this->sort();
	basis_io::write_basis_csv(state_list, outfilename);
}


void pyro_vtree::permute_spins(const std::vector<size_t>& perm) {
	for (auto& b : this->state_list) {
		b = permute(b, perm);
	}
}


void pyro_vtree_parallel::permute_spins(const std::vector<size_t>& perm) {
    std::vector<std::thread> threads;
    for (auto& l : state_set){
        threads.emplace_back([&l, perm]() {
            for (auto& b : l) {
                b = permute(b, perm);
            }
        });
    }
	// Wait for all threads to finish
	for (auto& t : threads){
		t.join();
	}
}

void pyro_vtree_parallel::write_basis_csv(const std::string& outfilename)
{
	this->sort();
	for (size_t i=1; i<state_set.size(); i++){
		if(state_set[i].size() != 0){
			throw std::logic_error("Error in write_basis_csv - basis was not sorted properly");
		}
	}
	basis_io::write_basis_csv(state_set[0], outfilename);
}

void pyro_vtree::write_basis_hdf5(const std::string& outfilename){
	this->sort();
	basis_io::write_basis_hdf5(this->state_list, outfilename);
}


void pyro_vtree_parallel::write_basis_hdf5(const std::string& outfilename){
	this->sort();
	basis_io::write_basis_hdf5(this->state_set[0], outfilename);
}





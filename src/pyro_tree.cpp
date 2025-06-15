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
#include "admin.hpp"


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

void lat_container::fork_state(std::stack<vtree_node_t>& to_examine) {
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
	std::stack<vtree_node_t> to_examine;
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


void pyro_vtree_parallel::sort(){	
	if (this->is_sorted) return;
	// step 1: move everything into state_set[0]
	auto& state_list = state_set[0];
	for (size_t i=1; i<state_set.size(); i++){
          state_list.insert(state_list.end(), state_set[i].begin(),
                            state_set[i].end());
		  // delete the old vector
		  state_set[i].clear();
		  state_set[i].shrink_to_fit();
	}
	// sort as normal
	std::sort(state_list.begin(), state_list.end());	
	this->is_sorted = true;
}

void pyro_vtree_parallel::
build_state_bfs(std::queue<vtree_node_t>& node_stack, 
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
build_state_dfs(std::stack<vtree_node_t>& node_stack, 
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



void pyro_vtree_parallel::build_state_tree() {
    // Initialize root work - seed first thread
    job_stacks[0].push(vtree_node_t({0, 0, 0}));
    
    std::atomic<bool> all_done{false};
    
    // Launch worker threads
    for (unsigned i = 0; i < n_threads; ++i) {
        threads.emplace_back([this, i, &all_done]() {
            build_state_dfs_work_stealing(i, all_done);
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    threads.clear(); // Clear for potential reuse
}

void pyro_vtree_parallel::build_state_dfs_work_stealing(unsigned thread_id, 
                                                       std::atomic<bool>& all_done,
                                                       unsigned long max_stack_size) {
    auto& local_stack = job_stacks[thread_id];
    auto& local_states = state_set[thread_id];
    auto& counter = counters[thread_id];
    
    while (!all_done.load(std::memory_order_relaxed)) {
        // Process local work first
        while (!local_stack.empty() && !all_done.load(std::memory_order_relaxed)) {
            auto curr = local_stack.top();
            local_stack.pop();
            
            counter++;
            
#if VERBOSITY > 2
            if (counter % 1000 == 0) {
                printf("Thread %u: processed %u nodes, stack size %lu\n", 
                       thread_id, counter, local_stack.size());
            }
#endif
            
            if (curr.curr_spin == lat.spins.size()) {
                // Found complete state - store in thread-local storage
                local_states.push_back(curr.state_thus_far);
            } else {
                // Generate child states using the existing logic
                char poss_states = this->possible_spin_states(curr);
                bool may_create_pair = (curr.num_spinon_pairs < this->num_spinon_pairs);
                
                // Generate spin-down state
                if (poss_states & 0b01) {
                    auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1, curr.num_spinon_pairs});
                    local_stack.push(tmp);
                } else if (may_create_pair) {
                    auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1, curr.num_spinon_pairs + 1});
                    local_stack.push(tmp);
                }
                
                // Generate spin-up state
                if (poss_states & 0b10) {
                    auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1, curr.num_spinon_pairs});
                    or_bit(tmp.state_thus_far, curr.curr_spin);
                    local_stack.push(tmp);
                } else if (may_create_pair) {
                    auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1, curr.num_spinon_pairs + 1});
                    or_bit(tmp.state_thus_far, curr.curr_spin);
                    local_stack.push(tmp);
                }
                
                // Check if we should limit stack size to prevent memory issues
                if (local_stack.size() > max_stack_size) {
                    // Share work if stack gets too large
                    share_work_if_needed(thread_id, max_stack_size / 2);
                }
            }
            
            // Periodically share work for load balancing
            if (local_stack.size() > WORK_STEAL_THRESHOLD) {
                share_work_if_needed(thread_id, WORK_STEAL_THRESHOLD);
            }
        }
        
        // Local work is done, try to steal work from other threads
        if (!try_steal_work(thread_id)) {
            // No work found, check if all threads are done
            if (!has_work_available(thread_id)) {
                all_done.store(true, std::memory_order_relaxed);
                break;
            }
            
            // Brief pause before trying again
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

bool pyro_vtree_parallel::try_steal_work(unsigned thread_id) {
    std::vector<vtree_node_t> stolen_work;
    
    // Try to steal from other threads' work-stealing queues
    for (unsigned i = 0; i < n_threads; ++i) {
        if (i == thread_id) continue;
        
        // Try to steal a batch of work
        size_t stolen_count = work_stealing_queues[i]->try_steal_batch_front(stolen_work, WORK_STEAL_BATCH_SIZE);
        if (stolen_count > 0) {
            // Add stolen work to local stack
            for (const auto& work : stolen_work) {
                job_stacks[thread_id].push(work);
            }
            return true;
        }
        
        // If batch stealing failed, try to steal single item
        vtree_node_t single_work;
        if (work_stealing_queues[i]->try_steal_front(single_work)) {
            job_stacks[thread_id].push(single_work);
            return true;
        }
    }
    
    // Also try to steal directly from other threads' stacks (more aggressive)
    for (unsigned i = 0; i < n_threads; ++i) {
        if (i == thread_id) continue;
        
        // This is a more invasive form of work stealing - use sparingly
        if (job_stacks[i].size() > MIN_WORK_TO_SHARE * 2) {
            // Try to steal some work items by temporarily accessing other stack
            // Note: This requires careful synchronization in a real implementation
            // For now, we'll rely on the work-stealing queues primarily
        }
    }
    
    return false;
}

void pyro_vtree_parallel::share_work_if_needed(unsigned thread_id, size_t threshold) {
    auto& local_stack = job_stacks[thread_id];
    
    if (local_stack.size() <= threshold) {
        return;
    }
    
    // Calculate how much work to share
    size_t work_to_share = std::min(local_stack.size() / 2, WORK_STEAL_BATCH_SIZE);
    
    if (work_to_share < MIN_WORK_TO_SHARE / 4) {
        return; // Not worth sharing such a small amount
    }
    
    // Move work from stack to work-stealing queue
    std::vector<vtree_node_t> work_items;
    work_items.reserve(work_to_share);
    
    for (size_t i = 0; i < work_to_share && !local_stack.empty(); ++i) {
        work_items.push_back(local_stack.top());
        local_stack.pop();
    }
    
    work_stealing_queues[thread_id]->push_batch_back(work_items);
}

bool pyro_vtree_parallel::has_work_available(unsigned exclude_thread_id) const {
    // Check if any thread has work available
    for (unsigned i = 0; i < n_threads; ++i) {
        if (i == exclude_thread_id) continue;
        
        if (!job_stacks[i].empty() || !work_stealing_queues[i]->empty()) {
            return true;
        }
    }
    return false;
}


/*
void pyro_vtree_parallel::
build_state_tree(){
	// strategy: fork nodes until we exceed the thread pool	
	// BFS to keep the layer of all threads roughly the same
	std::queue<vtree_node_t> starting_nodes;
	starting_nodes.push(vtree_node_t({0,0,0}));
	build_state_bfs(starting_nodes, n_threads);
	assert(starting_nodes.size() <= n_threads);
	n_threads = starting_nodes.size();
	printf("Set up execution with %u threads\n", n_threads);

	// now farm out to different threads
	state_set.resize(n_threads);
	job_stacks.resize(n_threads);
	counters.resize(n_threads);

	for (unsigned thread_id=0; thread_id<n_threads; thread_id++){
		job_stacks[thread_id].push(starting_nodes.front());
		starting_nodes.pop();
		printf("Thread %u state 0x%llx\n", thread_id, job_stacks[thread_id].top().state_thus_far.uint64[0]);
		threads.push_back(std::thread([this, thread_id]() {
					_build_state_dfs(job_stacks[thread_id], thread_id); }));
	}

	// Wait for all threads to finish
	for (auto& t : threads){
		t.join();
	}
	// make sure the queue was cleared
	for (auto& q : job_stacks){
		if (!q.empty()){
			throw std::logic_error("Queue not cleared, likely exceeded iteration limit");
		}
	}

}

*/


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
	for (auto& state_list : this->state_set) {
		for (auto& b : state_list) {
			b = permute(b, perm);
		}
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





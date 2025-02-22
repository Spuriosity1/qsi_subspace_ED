#include "pyro_tree.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include "bittools.hpp"
#include "vanity.hpp"





// LOGIC
char lat_container::possible_spin_states(const Uint128& state, unsigned idx) const {
	// state is only initialised up to (but not including) bit 1<<idx
	// returns possible states of state&(1<<idx)

	// return values:
	// 0b00 -> no spin state valid
	// 0b01 -> spin down (0) state valid
	// 0b10 -> spin up (1) state valid
	// 0b11 -> both up and down valid
	char res=0b11;

	Uint128 state_new = state; // new spin is already a 0
#ifndef NDEBUG
	assert( !readbit(state, idx) );
	assert( idx < lat.spins.size() );
#endif
	const auto known_mask = make_mask(idx);

	for (__uint128_t updown=0; updown<2; updown++){
		if (updown == 1){
			or_bit(state_new, idx);
		}

		for (auto t : lat.spins[idx].tetra_neighbours){
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
				break; // no point checking anything else
			}
		}

	}
	return res;
}


// Attempts to generate the two next configurations and add them to the queue
template <typename Container>
void lat_container::fork_state_impl(Container& to_examine, vtree_node_t curr) {
    char poss_states = this->possible_spin_states(curr.state_thus_far, curr.curr_spin);
    if (poss_states & 0b01) { // 0 is allowed
        auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1});
        to_examine.push(tmp);
    }
    if (poss_states & 0b10) { // 1 is allowed
        auto tmp = vtree_node_t({curr.state_thus_far, curr.curr_spin + 1});
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
	to_examine.push(vtree_node_t({0,0}));

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
			states_2I2O.push_back(to_examine.top().state_thus_far);
			to_examine.pop();
		} else {
			fork_state(to_examine);
		}
	}
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
_build_state_dfs(std::stack<vtree_node_t>& node_stack, 
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

void pyro_vtree_parallel::
build_state_tree(){
	// strategy: fork nodes until we exceed the thread pool	
	// BFS to keep the layer of all threads roughly the same
	std::queue<vtree_node_t> starting_nodes;
	starting_nodes.push(vtree_node_t({0,0}));
	_build_state_bfs(starting_nodes, n_threads);
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


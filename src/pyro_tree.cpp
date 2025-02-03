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
			int max_spins_up = (num_spins +1) /2;
			int min_spins_up = (num_spins   ) /2;
			int num_unknown_spins = num_spins - num_known_spins;

			if (Q + num_unknown_spins < min_spins_up || Q > max_spins_up){
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
/*

/// old, inefficient implementation


void base_pyro_tree::print_state_tree() const {
	// print breadthwise
	std::queue<node_t*> next;
	next.push(root);

	while(next.size() > 0){
		node_t* curr = next.front();
		next.pop();
		_print_node(curr);
		if(curr->leaf[0]) next.push(curr->leaf[0]);
		if(curr->leaf[1]) next.push(curr->leaf[1]);
	}
}

void base_pyro_tree::_print_node(node_t* n) const
{
	printf("%16p |",static_cast<void*>(n));
	for (unsigned i=0; i<n->spin_idx; i++){ printf(" "); }
	printf("0x%016llx%016llx --> ", n->state_thus_far.uint64[1],n->state_thus_far.uint64[0]);
	printf("0: %16p ", static_cast<void*>(n->leaf[0]));
	printf("1: %16p \n", static_cast<void*>(n->leaf[1]));
}


base_pyro_tree::~base_pyro_tree(){
	// delete the tree
	if (this->root == nullptr){
		return; // tree already deleted!
	}
	std::stack<node_t*> to_delete;
	to_delete.push(root);
	while (!to_delete.empty()){
		node_t* curr = to_delete.top();
		to_delete.pop();

		for (int i=0; i<2; i++){
			if(curr->leaf[i]){ to_delete.push(curr->leaf[i]); }
		}
		delete curr;
	}

}

void pyro_tree::build_state_tree(){	
	// start the recursion
	_build_state_tree(root);
}


void base_pyro_tree::_fork_node(node_t* node){
	char options = possible_spin_states(node->state_thus_far, node->spin_idx);
#if VERBOSITY > 2
	printf("%p | index %d poss_states %d\n", static_cast<const void*>(node), node->spin_idx,options);
#endif
	for (int i=0; i<2; i++){
		auto& n = node->leaf[i];
		if (options & (1<<i)) {
			n = new node_t;
			n->spin_idx = node->spin_idx+1;
			n->state_thus_far = node->state_thus_far;
			or_bit(n->state_thus_far, node->spin_idx, i);			
		}
	}
}

void pyro_tree::_build_state_tree(node_t* node){
	if (node->spin_idx == lat.spins.size() ){
		// end of the line, we have a working ice state (no children)
		states_2I2O.push_back(node->state_thus_far);
		return;
	}
	_fork_node(node);
	for (int i=0; i<2; i++){
		if (node->leaf[i]){
			_build_state_tree(node->leaf[i]);
		}
	}
}


void parallel_pyro_tree::_build_state_tree(node_t* node, unsigned tid){
	if (node->spin_idx == lat.spins.size() ){
		// end of the line, we have a working ice state (no children)
		state_set[tid].push_back(node->state_thus_far);
		return;
	}
	_fork_node(node);
	for (int i=0; i<2; i++){
		if (node->leaf[i]){
			_build_state_tree(node->leaf[i], tid);
		}
	}
}

void parallel_pyro_tree::build_state_tree(){
	std::queue<node_t*> starting_nodes;
	starting_nodes.push(root);
	while (starting_nodes.size() < n_threads){
		// fork these nodes until
		node_t* n = starting_nodes.front();
		starting_nodes.pop();
		this->_fork_node(n);
		for (int i=0; i<2; i++){
			if (n->leaf[i]) {
				starting_nodes.push(n->leaf[i]);
			}
		}
	}
	unsigned curr_thread_idx = 0;
	while (!starting_nodes.empty()){
		node_t* curr = starting_nodes.front();
		starting_nodes.pop();
		threads.push_back(
				std::thread([this, curr, curr_thread_idx] {
					this->_build_state_tree(curr, curr_thread_idx);
					}
				));
		curr_thread_idx++;
	}
	for (auto& t : threads){
		t.join();
	}
	assert(curr_thread_idx == n_threads);
}

*/

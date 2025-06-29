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

template<typename T>
void remove_empty(std::vector<std::vector<T>>& chunks){
    chunks.erase(
            std::remove_if(chunks.begin(), chunks.end(),
                [](const auto& chunk) { return chunk.empty(); }),
            chunks.end()
            );

}


// Zero-allocation in-place merge without any heap
template <typename T>
void inplace_merge_chunks(std::vector<std::vector<T>>& chunks) {
    if (chunks.size() <= 1) return;

    for (auto& c : chunks){
        std::cout<<"[merge] "<<c.size()<<"\n";
    }
    
    remove_empty(chunks); 
    if (chunks.size() <= 1) return;

// Merge chunks pairwise in-place - no heap allocation
    while (chunks.size() > 1) {
        std::vector<std::vector<T>> next_round;
        next_round.reserve((chunks.size() + 1) / 2);
        
        // Merge pairs of chunks
        for (size_t i = 0; i + 1 < chunks.size(); i += 2) {
            auto& left = chunks[i];
            auto& right = chunks[i + 1];
            
            if (right.empty()) {
                if (!left.empty()) {
                    next_round.push_back(std::move(left));
                }
                continue;
            }
            if (left.empty()) {
                next_round.push_back(std::move(right));
                continue;
            }
            
            // Reserve space for merged result
            size_t left_size = left.size();
            left.reserve(left_size + right.size());
            
            // Move elements from right to end of left
            left.insert(left.end(),
                       std::make_move_iterator(right.begin()),
                       std::make_move_iterator(right.end()));
            
            // In-place merge within the single vector
            std::inplace_merge(left.begin(),
                             left.begin() + left_size,
                             left.end());
            
            next_round.push_back(std::move(left));
        }
        
        // Handle odd chunk if exists
        if (chunks.size() % 2 == 1) {
            auto& last = chunks.back();
            if (!last.empty()) {
                next_round.push_back(std::move(last));
            }
        }
        
        chunks = std::move(next_round);
    }
    
}

int cmp_uint128( const void* a, const void*b){
    return *static_cast<const Uint128*>(a) < *static_cast<const Uint128*>(b);
}

void pyro_vtree_parallel::sort() {
    if (this->is_sorted) return;
    
    // Remove empty chunks first
    state_set.erase(
        std::remove_if(state_set.begin(), state_set.end(),
                      [](const auto& chunk) { return chunk.empty(); }),
        state_set.end()
    );
    
    if (state_set.empty()) {
        this->is_sorted = true;
        return;
    }
    
    // If only one chunk, sort it directly
    if (state_set.size() == 1) {
        std::sort(state_set[0].begin(), state_set[0].end());
        this->is_sorted = true;
        return;
    }
    
    // Sort each chunk individually (this is the only parallel part remaining)
    // Sacrifice some parallelism for guaranteed in-place operation
    const size_t num_threads = std::min(state_set.size(), 
                                       static_cast<size_t>(std::thread::hardware_concurrency()));
    
    if (num_threads > 1 && state_set.size() > 1) {
        // Parallel sort of individual chunks
        std::vector<std::future<void>> futures;
        futures.reserve(state_set.size());
        
        for (auto& chunk : state_set) {
            futures.push_back(std::async(std::launch::async, [&chunk]() {
                if (!chunk.empty()) {
                    //std::qsort(chunk.data(), chunk.size(), sizeof(Uint128), cmp_uint128);
                    std::sort(chunk.begin(), chunk.end());
                }
            }));
        }
        
        // Wait for all sorting to complete
        for (auto& future : futures) {
            future.wait();
        }
    } else {
        // Sequential sort
        for (auto& chunk : state_set) {
            if (!chunk.empty()) {
                //std::qsort(chunk.data(), chunk.size(), sizeof(Uint128), cmp_uint128);
                std::sort(chunk.begin(), chunk.end());
            }
        }
    }
    
    // Merge chunks in-place (sequential but memory-efficient)
    inplace_merge_chunks(state_set);
    
    // Ensure we have the expected structure
    assert(state_set.size() == 1);
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
    const int CHECKIN_INTERVAL=50000000;
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
    for (size_t i=0; i<state_set.size(); i++){
        std::cout << "[w] chunk ["<<i<<"] size "<<state_set[i].size() <<"\n";
    }
	basis_io::write_basis_hdf5(this->state_set[0], outfilename);
}





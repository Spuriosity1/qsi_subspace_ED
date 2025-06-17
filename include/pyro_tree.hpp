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


template<typename T>
struct vstack {
    std::vector<T> m_contents;
    T top() const { return m_contents.back(); }
    T pop() { 
        T top = m_contents.back();
        m_contents.pop_back();
        return top;
    }
    bool empty() const { return m_contents.empty(); }
    size_t size() const { return m_contents.size(); }
    void push(const T& x) { m_contents.push_back(x); }
};


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

    // using cust_stack = vstack<vtree_node_t>;
    using cust_stack = std::stack<vtree_node_t>;
	void fork_state(cust_stack& to_examine);
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
	void _build_state_dfs(cust_stack &node_stack, unsigned thread_id,
			unsigned long max_stack_size = (1ul << 40));
	void _build_state_bfs(std::queue<vtree_node_t>& node_stack, 
		unsigned long max_queue_len);
	unsigned n_threads;


    template <typename StackT>
    void rebalance_stacks(std::vector<StackT>& stacks);

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
	std::vector<cust_stack> job_stacks;

    static constexpr unsigned INITIAL_DEPTH_FACTOR = 5;
};






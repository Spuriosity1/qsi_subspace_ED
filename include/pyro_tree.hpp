#pragma once
#include "bittools.hpp"
#include "json.hpp"
#include "tetra_graph_io.hpp"
#include <thread>
#include <queue>
#include <stack>
#include <vector>

struct node_t {
	unsigned spin_idx; // aka layer, 	
	Uint128 state_thus_far; // only up to (1>>spin_idx) have been set

	node_t* leaf[2]={nullptr, nullptr};

};


struct vtree_node_t {
	Uint128 state_thus_far;
	unsigned curr_spin;
	// curr_spin is the bit ID of the rightmost unknown spin
	// i.e. (1<<curr_spin) & state_thus_far is guaranteed to be 0
};

struct lat_container {
	lat_container(const nlohmann::json& data) : 
		lat(data) {}


	// state is only initialised up to (but not including) bit 1<<idx
	// returns possible states of state&(1<<idx)

	// return values:
	// 0b00 -> no spin state valid
	// 0b01 -> spin down (0) state valid
	// 0b10 -> spin up (1) state valid
	// 0b11 -> both up and down valid
	char possible_spin_states(const Uint128& state, unsigned idx) const ;

	protected:

	template <typename Container>
	void fork_state_impl(Container& to_examine, vtree_node_t curr); 

	void fork_state(std::stack<vtree_node_t>& to_examine);
	void fork_state(std::queue<vtree_node_t>& to_examine);
	lattice lat;


};

struct pyro_vtree : public lat_container {
	pyro_vtree(const nlohmann::json&data) :
		lat_container(data) {
		}

	void build_state_tree();
	void write_basis_file(FILE* outfile){
		for (auto b : states_2I2O){
			write_line(outfile, b);
		}
	}
	protected:
	void save_state(const Uint128& state) {
			states_2I2O.push_back(state);
	}
	// Repository of ice states for perusal
	std::vector<Uint128> states_2I2O;


	// auxiliary variable for printing
	unsigned counter = 0;
};

struct pyro_vtree_parallel : public lat_container {
	pyro_vtree_parallel(const nlohmann::json &data, unsigned n_threads = 1)
		: lat_container(data), n_threads(n_threads) {}

	void build_state_tree();
	void write_basis_file(FILE *outfile) {
		for (auto states_2I2O : state_set) {
			for (auto b : states_2I2O) {
				write_line(outfile, b);
			}
		}
	}

protected:
	void _build_state_dfs(std::stack<vtree_node_t> &node_stack, unsigned thread_id,
			unsigned long max_stack_size = (1ul << 40));
	void _build_state_bfs(std::queue<vtree_node_t>& node_stack, 
		unsigned long max_queue_len);
	unsigned n_threads;

	// first index is the thread ID
	std::vector<std::vector<Uint128>> state_set;
	std::vector<std::thread> threads;
	std::vector<std::stack<vtree_node_t>> job_stacks;

	// auxiliary, for debug only
	std::vector<unsigned> counters = {0};
};

// older implementation
struct base_pyro_tree : public lat_container {
	base_pyro_tree(const nlohmann::json& data) :
		lat_container(data)
	{
		this->root = new node_t;
		root->spin_idx=0;
		root->state_thus_far.uint128=0;
	}
	base_pyro_tree(base_pyro_tree& other) = delete;
	base_pyro_tree(base_pyro_tree&& other) = delete;

	~base_pyro_tree();


	void print_state_tree() const ;


	protected:
	void _print_node(node_t* n) const;

	// checks if two valid states can be made and makes children if possible
	void _fork_node(node_t* n);

	node_t* root=nullptr;

};

struct pyro_tree : public base_pyro_tree {
	pyro_tree(const nlohmann::json& data) :
		base_pyro_tree(data)
	{
	}

	void build_state_tree();
	void write_basis_file(FILE* outfile){
		for (auto b : states_2I2O){
			std::fprintf(outfile, "0x%016llx%016llx\n", b.uint64[1],b.uint64[0]);
		}
	}
protected:
	void _build_state_tree(node_t* node);

	// Repository of ice states for perusal
	std::vector<Uint128> states_2I2O;
};

struct parallel_pyro_tree : public base_pyro_tree {
	parallel_pyro_tree(const nlohmann::json& data, unsigned n_threads=1) :
		base_pyro_tree(data),
		n_threads(n_threads)
	{
		state_set.resize(n_threads);
	}

	void build_state_tree();
	void write_basis_file(FILE* outfile){
		for (auto states_2I2O : state_set){
			for (auto b : states_2I2O){
				std::fprintf(outfile, "0x%016llx%016llx\n", b.uint64[1],b.uint64[0]);
			}
		}
	}
protected:
	void _build_state_tree(node_t* node, unsigned tid);
	// first index is the thread ID
	std::vector<std::vector<Uint128>> state_set;
	std::vector<std::thread> threads;
	const unsigned n_threads;

};




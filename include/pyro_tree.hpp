#pragma once
#include "bittools.hpp"
#include "tetra_graph_io.hpp"
#include <algorithm>

struct node_t {
	int spin_idx; // aka layer, 	
	Uint128 state_thus_far; // only up to (1>>spin_idx) have been set

	node_t* leaf[2]={nullptr, nullptr};

};


struct pyro_tree : public lattice {
	pyro_tree(const nlohmann::json& data) : 
		lattice(data)
	{
	}


	// state is only initialised up to (but not including) bit 1<<idx
	// returns possible states of state&(1<<idx)

	// return values:
	// 0b00 -> no spin state valid
	// 0b01 -> spin down (0) state valid
	// 0b10 -> spin up (1) state valid
	// 0b11 -> both up and down valid
	char possible_spin_states(const Uint128& state, int idx) const ;


	void build_state_tree();
	std::vector<Uint128> get_states(){
		return states_2I2O;
	}

	void print_state_tree() const ;


	~pyro_tree();
	private:
	void _print_node(node_t* n) const;
	void _build_state_tree(node_t* node);

	// Repository of ice states for perusal
	std::vector<Uint128> states_2I2O;
	node_t* root=nullptr;
/*
	// auxiliary variables to track evaluation
	std::vector<unsigned> tree_size;
	const unsigned print_frequency=500; // frequency of printing updates
	unsigned counter=print_frequency-1;
	*/
};

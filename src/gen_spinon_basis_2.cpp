#include "tetra_graph_io.hpp"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sys/cdefs.h>
#include <stdint.h>
#include <vector>
#include "bittools.hpp"


using namespace std;
using json=nlohmann::json;



struct spin_t;

struct tetra_node {
	tetra_node(std::vector<int> spin_ids){
		this->bitmask = 0;
		for (auto i : spin_ids){
			this->bitmask |= (1 << i);
		}
		num_spins = spin_ids.size();
	}
	
	Uint128 bitmask;
	int num_spins; // almost always 4
};

struct spin_t{
	std::vector<tetra_node*> tetra_neighbours;
	// almost always 2 members, possibly fewer
	
};

struct node_t {
	int spin_idx; // aka layer, 	
	Uint128 state_thus_far; // only up to (1>>spin_idx) have been set

	node_t* leaf[2]={nullptr, nullptr};

};


class lattice {
	public:
	lattice(const tetra_graph& graph){
		spins.resize(graph.num_spins());
		for (auto t : graph.tetra_no){
			std::vector<int> member_spins;
			for (auto [spin_id, other_t] :  t->links){
				assert(spin_id < graph.num_spins());
				member_spins.push_back(spin_id);
			}
			tetra_nodes.push_back(tetra_node(member_spins));
			for (auto [spin_id, other_t] :  t->links){
				spins[spin_id].tetra_neighbours.push_back(&tetra_nodes.back());
			}
		}
	}



	// return values:
	// 0b00 -> no spin state valid
	// 0b01 -> spin down (0) state valid
	// 0b10 -> spin up (1) state valid
	// 0b11 -> both up and down valid
	//
	//
	char possible_spin_states(const Uint128& state, int idx){
		// state is only initialised up to (but not including) bit 1<<idx
		// returns possible states of state&(1<<idx)
		char res = 0b11;

		Uint128 state_new = state; // new spin is already a 0
		assert((state_new & (1 << idx) ) == 0);
		
		for (int updown=0; updown<2; updown++){
			state_new |= updown << idx;

			for (auto t : spins[idx].tetra_neighbours){
				// calculate the partial tetra charges
				int Q = popcnt_u128(  (state_new & t->bitmask) >> idx);
				int num_known_spins = popcnt_u128(t->bitmask >> idx);

				if (abs(Q-2) > t->num_spins-num_known_spins){
					// Q is inconsistent with an ice rule
					res &= ~(1<<updown);
					break; // no point checking anything else
				}
			}

		}
		return res;
	}

	void build_state_tree(){
		node_t* root = new node_t;
		_build_state_tree(root);
	}

	std::vector<Uint128> get_states(){
		return states_2I2O;
	}

	private:


	void _build_state_tree(node_t* node){
		if (node->spin_idx == spins.size() -1){
			// end of the line, we have a working ice state
			states_2I2O.push_back(node->state_thus_far);
			return;
		}

		char options = possible_spin_states(node->state_thus_far, node->spin_idx+1);
		for (int i=0; i<2; i++){
			auto& n = node->leaf[i];
			if (options & (1<<i)) {
				n = new node_t;
				n->spin_idx = node->spin_idx+1;
				n->state_thus_far = node->state_thus_far;
				n->state_thus_far |= (i << n->spin_idx);
				_build_state_tree(n);
			}
		}
	}

	std::vector<spin_t> spins;
	std::vector<tetra_node> tetra_nodes;
	std::vector<Uint128> states_2I2O;

};

int main(int argc, char *argv[]) {
	if (argc < 2) {
		cout << "USAGE: " << argv[0] << "<latfile: json>";
	}

	std::string infilename(argv[1]);
	string outfilename=infilename.substr(0,infilename.find_last_of('.'))+".csv";

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	pyro_connectivity_data pyro(data);
	tetra_graph graph(pyro);

	lattice L(graph);

	L.build_state_tree();
	
	FILE* outfile = std::fopen(outfilename.c_str(), "w");
	for (Uint128 b : L.get_states()){
		std::fprintf(outfile, "0x%llx%llx\n", 
				static_cast<uint64_t>(b>>64),
				static_cast<uint64_t>(b));	
	}
	std::fclose(outfile);


	return 0;
}

#include <bitset>
#include <nlohmann/json_fwd.hpp>
#include <queue>
#include <fstream>
#include <iostream>
#include <map>
#include <strings.h>
#include <utility>
#include <vector>
#include <set>
#include "tetra_graph_io.hpp"
#include "bittools.hpp"

using namespace std;
using json=nlohmann::json;

template<typename T>
inline void setbits(__int128 &x, T &spin_idx) {
	for (auto J : spin_idx) {
		x |= (1 << J);
	}
}


template<typename T>
inline void setbits(__int128 &x, const T &spin_idx, bitset<64> bits) {
	int i=0;

	for (auto J : spin_idx) {
		x |= (bits[i] << J);
		i++;
	}
}

const static std::array<int, 6> states_2I2O = {0b0011, 0b0101, 0b1001,
	0b1100, 0b1010, 0b0110};


struct tetra_tree_node {
	int idx;
	tetra_tree_node* prev = nullptr;
	// pairs of the form spin_idx, tetra_tree_node*
	// these are the free spins
	std::vector<std::pair<int, tetra_tree_node *>> leaves; 
														   
														   
	__int128 constraint_mask; // a mask such that statedict[state & constraint_mask] gives all accessible states
	std::map<__int128, std::vector<__int128>> statedict;
};

struct tetra_tree {
	tetra_tree(const tetra_graph& g, int root_idx=0) {
		root = new tetra_tree_node;
		root->idx = root_idx;

		// keep track of which spins we have covered
		std::set<int> g_owned_spin_ids;


		std::queue<std::pair<tetra*, tetra_tree_node*>> next;
		next.push(std::make_pair(g.tetra_no[0], root));
		
		index.resize(g.tetra_no.size());
		while (next.size() > 0){
			auto [curr_tet, curr_node] = next.front();
			next.pop();
			for (const auto [si, s_tetra] : curr_tet->links){
				if (g_owned_spin_ids.insert(si).first == g_owned_spin_ids.end()){
					// unsuccessful insertion. si already owned by another tetra
					continue;
				}
				// si has not yet been traversed, we own this spin now
				tetra_tree_node* tn = new tetra_tree_node;
				tn->idx = s_tetra->idx;
				tn->prev = curr_node;

				curr_node->leaves.push_back(std::make_pair(si, tn));
				next.push(std::make_pair(s_tetra,tn));
				index[tn->idx] = tn;
			}
		}
		assert(g_owned_spin_ids.size() < g.num_spins());
		set_lookup_tables(g);


	}

	inline std::vector<int> free_spin_ids(){
		// Returns a set of all the free spins
		std::vector<int> res;
		for (auto t: this->index){
			for (auto [si, _] : t->leaves){
				res.push_back(si);
			}
		}
		return res;
	}


	void enumerate_ice_states(std::vector<__int128>& out){

		// loose idea of the algorithm:
		// 1. starting from the root (least constrained), choose a free state
		std::queue<tetra_tree_node*> next;
		__int128 curr_state;
		next.push(this->root);
		auto free_spins = pruned_tree();



	}

private:
	void set_lookup_tables(const tetra_graph& g){
		// set the state-lookup tables
		for (int i=0; i<index.size(); i++){
			tetra_tree_node* node = index[i];
			tetra* tetra = g.tetra_no[i];

			std::vector<int> free_spins;
			std::set<int> constr_spins;
			for (auto [spini, ti] : tetra->links){
				// all spins connected 
				constr_spins.insert(spini);
			}
			for (auto [spini, nodei] : node->leaves){
				constr_spins.erase(spini);
				free_spins.push_back(spini);
			}
			setbits(node->constraint_mask,constr_spins);
			// now, node's constraint mask is correctly set.
			for (int i=0; i<(1<<tetra->links.size()); i++){
				// understood as i = 0     |  0 0 0
				//                   free  | constr
				int free_substate = i >> constr_spins.size();
				int constr_substate = i ^ (free_substate << constr_spins.size());
				__int128 free_state=0;
				__int128 constr_state=0;
				setbits(free_state, free_spins, free_substate);
				setbits(constr_state, constr_spins, constr_substate);
				if (node->statedict.find(constr_state) == node->statedict.end()){
					node->statedict[constr_state] = {};
				}
				node->statedict[constr_state].push_back(free_state);
			}	
		}
	}
	std::vector<tetra_tree_node*> index;
	tetra_tree_node* root;
};

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "USAGE: " << argv[0] << "<latfile: json>";
  }
  ifstream ifs(argv[1]);
  json data = json::parse(ifs);
  pyro_connectivity_data pyro(data);
  tetra_graph graph(pyro);

  // run BFS
  tetra_tree tree(graph);



  return 0;
}

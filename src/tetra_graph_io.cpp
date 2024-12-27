#include "tetra_graph_io.hpp"
#include <sstream>
#include <set>

using json = nlohmann::json;
using namespace std;


pyro_connectivity_data::pyro_connectivity_data(const json &data) {
	for (const auto &b : data["bonds"]) {
		bond_graph.push_back(std::make_pair(b["from_idx"], b["to_idx"]));
	}
	for (const auto &t : data["tetrahedra"]) {
		int tetra_sl = t["sl"];
		tetras[tetra_sl].push_back(t["member_spin_idx"]);
	}
	for (const auto &h : data["rings"]) {
		hexagons.push_back(h);
	}
}


//std::pair<int,int>
int pyro_connectivity_data::find_tetra(int tetra_sl, int spin_idx) const {
	/* returns index pair (tetra_idx, spin_sl) such that
	 * tetras[tetra_sl][tetra_idx][spin_sl] == spin_idx
	 *
	 */
	assert( (tetra_sl&1) == tetra_sl );
	// linear search bc we aren't NERDS
	auto these_tetras = tetras[tetra_sl];
	for (int tetra_idx=0; tetra_idx<these_tetras.size(); tetra_idx++){
		auto t = these_tetras[tetra_idx];
		for (int spin_sl=0; spin_sl<t.size(); spin_sl++){	
			if (spin_idx == t[spin_sl]){
				return tetra_idx;
				//return std::make_pair(tetra_idx, spin_sl);
			}
		}
	}
	std::stringstream s;
	s << "Spin index "<<spin_idx <<"is not in tetra list";
	throw std::runtime_error(s.str());
}



// a doubly linked list of tetras
// Responsible for fast index access ONLY
tetra_graph::tetra_graph(
		const pyro_connectivity_data& connspec
		) 
{

	//allocate all the objects and populate NN list
	for (int tsl=0; tsl<2; tsl++){
		auto tlist = connspec.tetras[tsl];
		for (const auto& member_list : tlist){
			tetra* t = new tetra;
			t->sl = tsl;

			for (int spinid : member_list){	
				int neighbour_idx = connspec.find_tetra(t->sl^1, spinid);
				tetra* tn = tetra_no[neighbour_idx];
				t->links.push_back(std::make_pair(
							spinid, tn));
			}

			t->idx = tetra_no.size();
			tetra_no.push_back(t);
		}
	}

}

int tetra_graph::num_spins() const {
	std::set<int> spin_ids;
	for (const auto t : tetra_no){
		for (auto& l : t->links){
			spin_ids.insert(l.first);
		}
	}
	return spin_ids.size();
}



std::vector<tetra*> tetra_graph::tetras_containing(int spin){
	std::vector<tetra*> finds;
	for (auto t : this->tetra_no){
		for (auto [si, _] : t->links){
			if (si == spin){
				finds.push_back(t);
			}
		}
	}
	return finds;
}



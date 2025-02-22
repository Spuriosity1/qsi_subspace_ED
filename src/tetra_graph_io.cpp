#include "tetra_graph_io.hpp"
#include <set>

using json = nlohmann::json;
using namespace std;


lattice::lattice(const json &data) {
	/*
	for (const auto &b : data["bonds"]) {
//		bond_graph.push_back(std::make_pair(b["from_idx"], b["to_idx"]));
		unique_spinids.insert(static_cast<int>(b["from_idx"]));
		unique_spinids.insert(static_cast<int>(b["to_idx"]));
	}
	*/

	std::set<int> unique_spinids;
	
	for (const auto &t : data["tetrahedra"]) {
		auto ti = tetra(t["member_spin_idx"]);
		unique_spinids.insert(ti.member_spin_ids.begin(), ti.member_spin_ids.end());
		this->tetras.push_back(std::move(ti));
	}
	for (const auto &h : data["rings"]) {
		auto hi = spin_set(h["member_spin_idx"]);
		unique_spinids.insert(hi.member_spin_ids.begin(), hi.member_spin_ids.end());
		this->rings.push_back(std::move(hi));
	}

	spins.resize(unique_spinids.size());	
	_register_spins();
}

void lattice::_register_spins() {
	// cross link with tetras and rings
	for (size_t ti=0; ti<tetras.size(); ti++){
		for (auto& si : tetras[ti].member_spin_ids){
			spins[si].tetra_neighbours.push_back(&tetras[ti]);
		}
	}
	for (size_t hi=0; hi < rings.size(); hi++){
		for (auto& si : rings[hi].member_spin_ids){
			spins[si].rings_containing.push_back(&rings[hi]);
		}
	}
}



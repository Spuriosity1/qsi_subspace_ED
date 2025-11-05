#include "tetra_graph_io.hpp"
#include <cmath>
#include <set>
#include <unordered_set>

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
    for(const auto& sid : unique_spinids){
        std::string sl = data["atoms"][sid]["sl"];
        spins[sid].sl = stoi(sl);
    }
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




void lattice::_permute_spins(const std::vector<size_t>& perm){
	if (perm.size() != spins.size()){
		throw std::out_of_range("Permutation applied does not match # spins");
	}
	std::vector<spin> tmp_spins(spins.size());
	std::set<size_t> permuted_ids;

	for (size_t i=0; i< spins.size(); i++){
		if (permuted_ids.insert(perm[i]).second == false){
			throw std::out_of_range("Perm is not a permutation");
		}
		tmp_spins[i] = spins[perm[i]];
	}
	spins = tmp_spins; // pointers inside are still valid
	
}

// helper function, applies i-> perm[i] for all ints
std::vector<int> perm_v(const std::vector<int>& v, const std::vector<size_t>& perm){
	std::vector<int> tmp(v.size());
	for (size_t i=0; i<v.size(); i++){
		tmp[i] = perm[v[i]];
	}
	return tmp;
}


void lattice::apply_permutation(const std::vector<size_t>& perm){
	auto iperm = invperm(perm);
	_permute_spins(perm); // reorder the spins themselves
	for (auto& t : tetras){
		for (int& sid : t.member_spin_ids){
			sid = iperm[sid];
		}
		t.recompute_bitmask();
	}
	for (auto& r : rings){
		for (int& sid : r.member_spin_ids){
			sid = iperm[sid];
		}
		r.recompute_bitmask();
	}
}

// Greedy algorithm -- we try to select an orderign of spins such that a()
std::vector<size_t> lattice::greedy_spin_ordering(int initial_id) const {
    const int N = spins.size();
    std::vector<bool> selected(N, false);
    std::vector<size_t> ordering;

    ordering.push_back(initial_id);
    selected[initial_id] = true;

	std::unordered_set<const tetra*> covered_tetras(
        spins[initial_id].tetra_neighbours.begin(), 
        spins[initial_id].tetra_neighbours.end()
    );

	for (int iter = 1; iter < N; ++iter) {
        int best_spin = -1;
        int best_overlap = -1;

        for (int i = 0; i < N; ++i) {
            if (selected[i]) continue;

            int overlap = 0;
            for (const auto* t : spins[i].tetra_neighbours) {
                if (covered_tetras.count(t)) ++overlap;
            }

            if (overlap > best_overlap) {
                best_overlap = overlap;
                best_spin = i;
            }
        }

        if (best_spin == -1) {
            // This can happen if disconnected: pick any remaining
            for (int i = 0; i < N; ++i) {
                if (!selected[i]) {
                    best_spin = i;
                    break;
                }
            }
        }

        selected[best_spin] = true;
        ordering.push_back(best_spin);

        for (const auto* t : spins[best_spin].tetra_neighbours) {
            covered_tetras.insert(t);
        }
    }

    return ordering;
}


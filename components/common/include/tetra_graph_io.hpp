#pragma once
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <set>
#include <array>
#include "bittools.hpp"

struct spin;
struct spin_set; 
struct tetra;

struct spin {
    int sl;
	std::vector<tetra*> tetra_neighbours;
	std::vector<spin_set*> rings_containing;
};

struct spin_set {
	spin_set(const std::vector<int>& spin_ids) : 
		member_spin_ids(spin_ids)
	{
		recompute_bitmask();
	}

	void recompute_bitmask(){
		this->bitmask.uint128 = 0;
		for (auto i : member_spin_ids){
			or_bit(this->bitmask, i);
		}
	}

	std::vector<int> member_spin_ids;
	Uint128 bitmask;
};

// represects specifically a tetra-like thing
struct tetra : public spin_set {
	tetra(const std::vector<int>& spin_ids, int min_spins_up, int max_spins_up) : 
		spin_set(spin_ids), min_spins_up(min_spins_up), max_spins_up(max_spins_up) {}

	// default no-spinon rules
	tetra(const std::vector<int>& spin_ids) : 
		spin_set(spin_ids){
		max_spins_up = (this->member_spin_ids.size() +1) /2;
		min_spins_up = (this->member_spin_ids.size() ) /2;
		}

	int min_spins_up;
	int max_spins_up;
};


struct lattice {
	lattice(const nlohmann::json &data); 
	
	std::vector<spin> spins;
	std::vector<tetra> tetras;
	std::vector<spin_set> rings;

	// reshusffles the indices such that 
	// spins[old_id] = spins[perm[old_id]]
	void apply_permutation(const std::vector<size_t>& perm);

	std::vector<size_t> greedy_spin_ordering(int initial_id=0) const;

	private:
	void _register_spins();
	void _permute_spins(const std::vector<size_t>& perm);
	
};

// Returns a vector of 'n' spin ids that connect untrelated spins
std::vector<std::vector<int>> find_defect_motifs(const nlohmann::json& jdata, int n);


namespace tetra_graph_io {

inline lattice read_latfile_json(const std::string& file){
	std::ifstream ifs(file);
	nlohmann::json data = nlohmann::json::parse(ifs);
	ifs.close();
	return lattice(file);
}

};



template<typename T>
requires std::convertible_to<T, size_t>
inline std::vector<T> invperm(const std::vector<T>&perm){
	std::vector<T> iperm(perm.size());
	for (size_t i=0; i<perm.size(); i++){
		iperm[perm[i]] = i;
	}
	return iperm;
}






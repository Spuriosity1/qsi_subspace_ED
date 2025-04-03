#pragma once
#include <fstream>
#include <json.hpp>
#include <vector>
#include <array>
#include "bittools.hpp"

struct spin;
struct spin_set; 
struct tetra;

struct spin {
	std::vector<tetra*> tetra_neighbours;
	std::vector<spin_set*> rings_containing;
};

struct spin_set {
	spin_set(const std::vector<int>& spin_ids) : 
		member_spin_ids(spin_ids)
	{
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
	
	private:
	void _register_spins();
};


namespace tetra_graph_io {

inline lattice read_latfile_json(const std::string& file){
	std::ifstream ifs(file);
	nlohmann::json data = nlohmann::json::parse(ifs);
	ifs.close();
	return lattice(file);
}

};



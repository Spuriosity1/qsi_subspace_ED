#pragma once
#include <json.hpp>
#include <vector>
#include <array>
#include "bittools.hpp"

struct spin;
struct spin_set; 

struct spin {
	std::vector<spin_set*> tetra_neighbours;
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



struct lattice {
	lattice(const nlohmann::json &data); 
	
	std::vector<spin> spins;
	std::vector<spin_set> tetras;
	std::vector<spin_set> rings;
	
	private:
	void _register_spins();
};



#pragma once
#include <nlohmann/json.hpp>
#include <vector>
#include <array>

struct pyro_connectivity_data {
	pyro_connectivity_data(const nlohmann::json &data); 

	std::vector<std::pair<int, int>> bond_graph; // each entry is a spin
	std::array<std::vector<std::vector<int>>, 2> tetras;
	std::vector<std::vector<int>> hexagons;

	//std::pair<int,int>
	int find_tetra(int tetra_sl, int spin_idx) const;
	int num_spins() const;
};

struct tetra;

struct tetra {
	int sl; // 0 or 1, up or down sublattice
	int idx; // index for tetra_no
	// the four (or fewer) links associated
	// format (spinid, pointer to neighbour)
	std::vector<std::pair<int, tetra*>> links;
};


// a doubly linked list of tetras
// Responsible for fast index access ONLY
struct tetra_graph {
	tetra_graph(
			const pyro_connectivity_data& connspec
			);

	int num_spins() const;

	std::vector<tetra*> tetra_no; // the index
	std::vector<tetra*> tetras_containing(int spin);

};


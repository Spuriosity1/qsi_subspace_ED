#pragma once
#include "tetra_graph_io.hpp"
#include<string>
#include <vector>
#include <random>

inline std::string as_basis_file(const std::string& input_jsonfile, const std::string& ext=".basis" ){
	return input_jsonfile.substr(0,input_jsonfile.find_last_of('.'))+ext;
}


inline std::vector<size_t> get_permutation(
		const std::string& choice, const lattice& lat){

	std::vector<size_t> perm;	
	for (size_t i=0; i<lat.spins.size(); i++){
		perm.push_back(i);
	}

	std::random_device rd;
	std::mt19937 rng(rd());
	if (choice == "greedy"){
		perm = lat.greedy_spin_ordering(0);
	} else if (choice == "random") {
		std::shuffle(perm.begin(), perm.end(), rng);
	}
	return perm;
}

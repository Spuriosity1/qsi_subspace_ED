#pragma once
#include "tetra_graph_io.hpp"
#include <hdf5.h>
#include<string>
#include<sstream>
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


// Helper function to split a string by a delimiter
inline std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        tokens.push_back(item);
    }
    return tokens;
}

// Extracts key-value pairs of the form key=value from the filename
inline std::unordered_map<std::string, std::string> parse_parameters(const std::string& path) {
    std::unordered_map<std::string, std::string> result;

    // Find the start of the parameter section
    size_t start = path.find_last_of('/');
    std::string filename = (start != std::string::npos) ? path.substr(start + 1) : path;

    // Split on '%'
    std::vector<std::string> parts = split(filename, '%');
    for (const auto& part : parts) {
        size_t eq_pos = part.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = part.substr(0, eq_pos);
            std::string value = part.substr(eq_pos + 1);
            result[key] = value;
        }
    }

    return result;
}



#pragma once
#include "bittools.hpp"
#include <string>
#include <vector>
#include <stdexcept>
#include <cinttypes>

#ifndef DONT_USE_HDF5
#include "basis_io_h5.hpp" 
#endif

namespace basis_io {



inline auto write_line(FILE* of, const Uint128& b){
    return std::fprintf(of, "0x%016" PRIx64 "%016" PRIx64 "\n", b.uint64[1], b.uint64[0]);
}

inline bool read_line(FILE *infile, Uint128& b) {
	char buffer[40];  // Enough to hold "0x" + 32 hex digits + null terminator
	if (!std::fgets(buffer, sizeof(buffer), infile)) {
		return false;  // Return false on failure (e.g., EOF)
	}
	
	return std::sscanf(buffer, "0x%016" PRIx64 "%016" PRIx64 , &b.uint64[1], &b.uint64[0]) == 2;
}

inline void write_basis_csv(const std::vector<Uint128>& state_list, 
		const std::string &outfilename) {
	FILE *outfile = std::fopen((outfilename + ".csv").c_str(), "w");
	for (auto b : state_list) {
	  basis_io::write_line(outfile, b);
	}

	std::fclose(outfile);
}


inline std::vector<Uint128> read_basis_csv(const std::string &infilename) {
	FILE *infile = std::fopen((infilename).c_str(), "r");
	if (!infile) {
		throw std::runtime_error("Failed to open file: " + infilename + ".csv");
	}
	std::vector<Uint128> state_list;
	Uint128 b;
	while (read_line(infile, b)) {
		state_list.emplace_back(b);
	}

	std::fclose(infile);
	return state_list;
}

};



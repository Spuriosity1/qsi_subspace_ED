#pragma once
#include "admin.hpp"
#include "bittools.hpp"
#include "pyro_tree.hpp"
#include <hdf5.h>
#include <string>
#include <vector>

namespace basis_io {

	inline auto write_line(FILE* of, const Uint128& b){
		return std::fprintf(of, "0x%016llx%016llx\n", b.uint64[1],b.uint64[0]);
	}

	inline bool read_line(FILE *infile, Uint128& b) {
		char buffer[40];  // Enough to hold "0x" + 32 hex digits + null terminator
		if (!std::fgets(buffer, sizeof(buffer), infile)) {
			return false;  // Return false on failure (e.g., EOF)
		}
		
		return std::sscanf(buffer, "0x%016llx%016llx", &b.uint64[1], &b.uint64[0]) == 2;
	}

};

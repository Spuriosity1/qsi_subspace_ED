#pragma once

#include "bittools.hpp"
#include <exception>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>
#include <basis_io.hpp>
#include <filesystem> // C++17
					  
namespace fs = std::filesystem;

typedef Uint128 comp_basis_state_t; // type which stores the computational basis state
typedef uint64_t idx_t;  // the type to use for the indices themselves




class state_not_found_error : public std::exception
{
	char state_as_c_str[128];
	public:
	state_not_found_error(const comp_basis_state_t& state_not_found){
		snprintf(state_as_c_str, sizeof(state_as_c_str), "State not found: 0x%8llx%8llx\n", 
				state_not_found.uint64[1], state_not_found.uint64[0]);
	}
	const char * what() const noexcept override {
		 return state_as_c_str;
	}
};



// Wrapper for a std::vector implementing indexable set semantics
// Represents a set of Sz product states spanning some subspace
struct ZBasis {
	ZBasis(){}

	// returns the index of a particular basis state
	idx_t idx_of_state(const comp_basis_state_t& state) const {
		auto it = state_to_index.find(state);

		return it->second;
	}
	inline comp_basis_state_t state_at_idx(idx_t idx) const {
		return states[idx];
	}

	void load_from_file(const fs::path& bfile){
		if (bfile.extension() == ".h5"){
			states = basis_io::read_basis_hdf5(bfile); 
		} else if (bfile.extension() == ".csv"){
			basis_io::read_basis_csv(bfile); 
		} else {
			throw std::runtime_error(
					"Bad basis format: file must end with .csv or .h5");
		}
	}

protected:
	std::vector<comp_basis_state_t> states;
	std::unordered_map<comp_basis_state_t, idx_t, Uint128Hash, Uint128Eq> state_to_index;
};



// Operators are instantiated relative to some basis, they keep 
// a reference to it
//
struct Operator {
	Operator(const ZBasis& _op_basis) : 
		op_basis(_op_basis) {};
protected:
	const ZBasis& op_basis;

};


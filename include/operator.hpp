#pragma once

#include "bittools.hpp"
#include <exception>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
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

	size_t ndim() const {
		return states.size();
	}

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
			states = basis_io::read_basis_csv(bfile); 
		} else {
			throw std::runtime_error(
					"Bad basis format: file must end with .csv or .h5");
		}

		for (int i=0; i<states.size(); i++){
			state_to_index[states[i]] = i;
		}
	}
	
protected:
	friend struct PMROperator;
	std::vector<comp_basis_state_t> states;
	std::unordered_map<comp_basis_state_t, idx_t, Uint128Hash, Uint128Eq> state_to_index;
};



// Operators are instantiated relative to some basis, they keep 
// a reference to it.
// Implements an Eigen-compatible operator*.
// Permuationa Matrix Representation Operator -- represents a generalised permutation.
// Representation is internally 
// const * Z Z Z Z X X X X + + + - - - -
// S- and S+ are impleneted as "abort masks" followed by an X.
struct PMROperator {
	using Index = Eigen::Index;

	PMROperator(const ZBasis& _basis) : 
		basis(_basis) {};

	// Required typedefs, constants, and method:
	typedef double Scalar;
	typedef double RealScalar;
	typedef int StorageIndex;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic,
		IsRowMajor = false
	};

	Index rows() const { return basis.ndim(); }
	Index cols() const { return basis.ndim(); }

	template <typename Rhs>
	Eigen::Product<PMROperator, Rhs, Eigen::AliasFreeProduct>
	operator*(const Eigen::MatrixBase<Rhs> &x) const {
		x.derived();
	}
	
	// Apply this operator to an input vector `in` and store result in `out`
	template <typename Dest>
	void applyTo(const Eigen::VectorXd& in, Dest& out) const {
		out.setZero();
		for (std::size_t i = 0; i < basis.states.size(); ++i) {
			const comp_basis_state_t& state = basis.states[i];

			const auto s = state.uint128;
			const auto d = down_mask.uint128;
			const auto u = up_mask.uint128;

			if ( (s & d) != 0 ) continue;
			if ( (s & u) != u ) continue;




			double phase = 1.0;
			if (auto it = basis.state_to_index.find(state); it != basis.state_to_index.end()) {
				out[it->second] += phase * in[i];
			}
		}
	}

protected:
	const ZBasis& basis;
	comp_basis_state_t X_mask;
	comp_basis_state_t Z_mask;
	comp_basis_state_t down_mask; // init_state & down_mask must be zero
	comp_basis_state_t up_mask; // init_state & up_mask must be == up_mask
};

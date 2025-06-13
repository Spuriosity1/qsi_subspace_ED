#pragma once

#include "bittools.hpp"
#include <exception>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef __APPLE__ // patch broken NEON optimization
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_NEON
#endif

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <basis_io.hpp>
#include <filesystem> // C++17
					  
namespace fs = std::filesystem;

typedef Uint128 comp_basis_state_t; // type which stores the computational basis state
typedef uint64_t idx_t;  // the type to use for the indices themselves

template<typename T>
concept ScalarLike = std::floating_point<T> ||
                 (requires { typename T::value_type; } &&
                  std::is_same_v<T, std::complex<typename T::value_type>> &&
                  std::floating_point<typename T::value_type>);


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


class bad_operator_spec : public std::exception
{
	char state_as_c_str[128];
	public:

	bad_operator_spec(const std::vector<char>& opstring, const std::vector<int>& spin_ids){ 
		if (opstring.size() != spin_ids.size()){
		snprintf(state_as_c_str, sizeof(state_as_c_str), 
				"Mangled opstring spec: vectors hve differing lengths \
opstring: %zu, spin_ids: %zu", opstring.size(), spin_ids.size());
		} else {	
			std::stringstream s;
			for (size_t i=0; i<opstring.size(); i++){
				s << spin_ids[i] <<opstring[i] <<" ";
			}
			snprintf(state_as_c_str, sizeof(state_as_c_str), "%s", s.str().c_str());
		}
	}
	const char * what() const noexcept override {
		 return state_as_c_str;
	}
};



// Wrapper for a std::vector implementing indexable set semantics
// Represents a set of Sz product states spanning some subspace
struct ZBasis {
	ZBasis(){}

	size_t dim() const {
		return states.size();
	}

	// returns the index of a particular basis state
	idx_t idx_of_state(const comp_basis_state_t& state) const {
		auto it = state_to_index.find(state);
		if (it == state_to_index.end()){
			throw state_not_found_error(state);
		}
		return it->second;
	}
	inline comp_basis_state_t operator[](idx_t idx) const {
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

		for (idx_t i=0; i<states.size(); i++){
			state_to_index[states[i]] = i;
		}
	}
	
protected:
	std::vector<comp_basis_state_t> states;
	std::unordered_map<comp_basis_state_t, idx_t, Uint128Hash, Uint128Eq> state_to_index;
};


inline int highest_set_bit(comp_basis_state_t x) {
    if (x == 0) return -1;

    if (x.uint64[1] != 0) {
        return 64 + 63 - __builtin_clzll(x.uint64[1]);
    } else {
        return 63 - __builtin_clzll(x.uint64[0]);
    }
}

// Operators are instantiated relative to some basis, they keep 
// a reference to it.
// Implements an Eigen-compatible operator*.
// Permuationa Matrix Representation Operator -- represents a generalised permutation.
// Representation is internally 
// const * Z Z Z Z X X X X + + + - - - -
// S- and S+ are impleneted as "abort masks" followed by an X.
struct SymbolicPMROperator {
	SymbolicPMROperator(const std::vector<char>& opstring, 
			const std::vector<int>& spin_ids)  {
			if ( opstring.size() != spin_ids.size() ){
				throw bad_operator_spec(opstring, spin_ids);
			}
			for (size_t i=0; i<opstring.size(); i++){
				auto J = spin_ids[i];
				switch (opstring[i]) {
					case 'x':
					case 'X':
						or_bit(X_mask, J);
						break;
					case 'z':
					case 'Z':
						or_bit(Z_mask, J);
						break;
					case '+':
						or_bit(X_mask, J);
						or_bit(down_mask, J);
						break;
					case '-':
						or_bit(X_mask, J);
						or_bit(up_mask, J);
						break;
					default:
						throw bad_operator_spec(opstring, spin_ids);
				}
			}
		}

    // returns sign of only possibly-nonzero entry, modifies J to its index
    int applyIndex(const ZBasis& basis, idx_t& J) const {
        comp_basis_state_t state = basis[J];

        const auto s = state.uint128;
        const auto d = down_mask.uint128;
        const auto u = up_mask.uint128;

        if ( (s & d) != 0 ) return 0;
        if ( (s & u) != u ) return 0;
        state ^= X_mask;
        int sign = 1 - 2 * (popcnt_u128(state & Z_mask) % 2);
        
        J= basis.idx_of_state(state);
        return sign;
    }
    
	
	// Apply this operator to an input vector `in` and store result in `out`
	template <typename Orig, typename Dest>
	void apply(const ZBasis& basis, const Orig& in, Dest& out) const {	
		for (idx_t i = 0; i < basis.dim(); ++i) {
            /*
			comp_basis_state_t state = basis[i];

			const auto s = state.uint128;
			const auto d = down_mask.uint128;
			const auto u = up_mask.uint128;

			if ( (s & d) != 0 ) continue;
			if ( (s & u) != u ) continue;
			state ^= X_mask;
			int sign = 1 - 2 * (popcnt_u128(state & Z_mask) % 2);

			out [ basis.idx_of_state(state) ] = sign * in[i];
            */

            idx_t J = i;
            auto c = applyIndex(basis, J) * in[i];
            out[J] += c;
		}
	}

	idx_t highest_set_bit() const {
		return ::highest_set_bit(X_mask | Z_mask | down_mask | up_mask);
	}


protected:
	comp_basis_state_t X_mask = 0;
	comp_basis_state_t Z_mask = 0;
	comp_basis_state_t down_mask = 0; // init_state & down_mask must be zero
	comp_basis_state_t up_mask = 0; // init_state & up_mask must be == up_mask
};


template<ScalarLike coeff_t>
struct SymbolicOpSum {
	using Op = SymbolicPMROperator;

	void add_term(coeff_t c, const Op& op) {
		terms.emplace_back(c, op); // copies are fine
	}

	void operator+=(const Op& op) {
		terms.emplace_back(1.0,op);
	}

	std::vector<std::pair<coeff_t, Op >> terms;
};

// define a natural algebra of these boys
template <ScalarLike coeff_t>
SymbolicOpSum<coeff_t> operator*(coeff_t c, SymbolicPMROperator op){
	SymbolicOpSum<coeff_t> s;
	s.add_term(c, op);
	return s;
}

template <ScalarLike coeff_t>
SymbolicOpSum<coeff_t> operator*(SymbolicPMROperator op, coeff_t c){
	SymbolicOpSum<coeff_t> s;
	s.add_term(c, op);
	return s;
}


template <ScalarLike coeff_t>
struct LazyOpSum {
	using Scalar = coeff_t;
	explicit LazyOpSum(
			const ZBasis& basis_, const SymbolicOpSum<coeff_t>& ops_
			) : basis(basis_), ops(ops_) 
	{
		// allocate the temporary storage
		tmp = new coeff_t[basis.dim()]; 
	}

	LazyOpSum(const LazyOpSum& other) : basis(other.basis), ops(other.ops){
		tmp = new coeff_t[basis.dim()];
	}

	~LazyOpSum() {
		delete[] tmp;
	}

	// Core evaluator 
	void evaluate(const coeff_t* x, coeff_t* y) const {
		std::fill(y, y + basis.dim(), coeff_t(0));
		for (const auto& [c, op_ptr] : ops.terms) {
			std::fill(tmp, tmp + basis.dim(), coeff_t(0));
			op_ptr.apply(basis, x, tmp);
			for (idx_t i = 0; i < basis.dim(); ++i)
				y[i] += c * tmp[i];
		}
	}


	// Eigen-compatible wrapper, evaluate y = this * x
	template <typename In, typename Out>
	void applyTo(const In& x, Out& y) const {
		Eigen::Map<const Eigen::VectorXd> x_vec(x.data(), basis.dim());
		Eigen::Map<Eigen::VectorXd> y_vec(y.data(), basis.dim());
		evaluate(x_vec.data(), y_vec.data());
	}
	
	// Eigen glue
	Eigen::Index rows() const { return basis.dim(); }
	Eigen::Index cols() const { return basis.dim(); }

	
	// Enable Eigen’s operator*
	template <typename Rhs>
	friend Eigen::Product<LazyOpSum, Rhs, Eigen::AliasFreeProduct>
	operator*(const LazyOpSum& lhs, const Eigen::MatrixBase<Rhs>& rhs) {
		return Eigen::Product<LazyOpSum, Rhs, Eigen::AliasFreeProduct>(lhs, rhs.derived());
	}	

	// Dense matrix matrialiser
	Eigen::Matrix<coeff_t, Eigen::Dynamic, Eigen::Dynamic>
	toDenseMatrix() const {
		Eigen::Index N = basis.dim();
		Eigen::Matrix<coeff_t, Eigen::Dynamic, Eigen::Dynamic> M(N, N);

		Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
		Eigen::VectorXd y(N);
		for (Eigen::Index j = 0; j < N; ++j) {
			x[j] = coeff_t(1);
			applyTo(x, y);
			M.col(j) = y;
			x[j] = coeff_t(0);
		}
		return M;
	}

	// Sparse matrix materialiser
	Eigen::SparseMatrix<coeff_t>
	toSparseMatrix(coeff_t tol = 1e-14) const {
		Eigen::Index N = basis.dim();
		std::vector<Eigen::Triplet<coeff_t>> triplets;

		// Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
		// Eigen::VectorXd y(N);

		for (Eigen::Index j = 0; j < N; ++j) {
            /*
			x[j] = coeff_t(1);
			applyTo(x, y);
			for (Eigen::Index i = 0; i < N; ++i) {
				if (std::abs(y[i]) > tol)
					triplets.emplace_back(i, j, y[i]);
			}
			x[j] = coeff_t(0);
            */
            for (auto& [c, op] : ops.terms){
                idx_t J = j;
                coeff_t res = c * op.applyIndex(basis, J);
                if (std::abs(res) > tol )
                    triplets.emplace_back(J, j, res);
            }

		}
		Eigen::SparseMatrix<coeff_t> S(N, N);
		S.setFromTriplets(triplets.begin(), triplets.end());
		return S;
	}

	private:
	coeff_t* tmp; // temp storage
	const ZBasis& basis;
	const SymbolicOpSum<coeff_t>& ops;
};

template <typename coeff_t>
class LazyOpSumProd {
public:
	using Scalar = coeff_t;

	LazyOpSumProd(const LazyOpSum<coeff_t>& op_)
	    : op(op_), xdim(op_.rows()) {}

	// Spectra expects raw pointers
	void perform_op(const coeff_t* x_in, coeff_t* y_out) const {
		op.evaluate(x_in, y_out);
	}

	Eigen::Index rows() const { return xdim; }
	Eigen::Index cols() const { return xdim; }

private:
	const LazyOpSum<coeff_t>& op;
	Eigen::Index xdim;
};


/*
namespace Eigen {
	template<typename coeff_t, typename Rhs>
	struct Product<LazyOpSum<coeff_t>, Rhs, AliasFreeProduct> :
		public Matrix<coeff_t, Dynamic, 1>
	{
		Product(const LazyOpSum<coeff_t>& op, const Rhs& rhs)
			: Matrix<coeff_t, Dynamic, 1>(op.rows())
		{
			op.applyTo(rhs, *this);
		}
	};
}
*/



#pragma once

#include "bittools.hpp"
#include <exception>
#include <iostream>
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


template<typename T>
concept ScalarLike = std::floating_point<T> ||
                 (requires { typename T::value_type; } &&
                  std::is_same_v<T, std::complex<typename T::value_type>> &&
                  std::floating_point<typename T::value_type>);


template<typename comp_basis_state_t>
class state_not_found_error : public std::exception
{
	char state_as_c_str[128];
	public:
	state_not_found_error(const comp_basis_state_t& state_not_found){
		snprintf(state_as_c_str, sizeof(state_as_c_str), "State not found: 0x%" PRIx64 "%" PRIx64 "\n", 
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

	using state_t = Uint128; // type which stores the computational basis state
	using idx_t = uint64_t;  // the type to use for the indices themselves


	ZBasis(){}

	size_t dim() const {
		return states.size();
	}

	// returns the index of a particular basis state
	idx_t idx_of_state(const state_t& state) const {
		auto it = state_to_index.find(state);
		if (it == state_to_index.end()){
			throw state_not_found_error(state);
		}
		return it->second;
	}
	inline state_t operator[](idx_t idx) const {
		return states[idx];
	}

	size_t insert_states(const std::vector<state_t>& others,
			std::vector<state_t>& new_states){
		new_states.resize(0);
		size_t n_insertions = 0;
		for (auto& s : others){
			// skip if we know about it already
			if (state_to_index.find(s) != state_to_index.end()) continue;
			state_to_index[s] = states.size();
			states.push_back(s);
			new_states.push_back(s);
			n_insertions++;
		}
		return n_insertions;
	}

	void load_from_file(const fs::path& bfile, const std::string& dataset="basis"){
        std::cerr << "Loading basis from file " << bfile <<"\n";
        if (bfile.stem().extension() == ".partitioned"){
            assert(bfile.extension() == ".h5");
            states = basis_io::read_basis_hdf5(bfile, dataset.c_str());
        } else if (bfile.extension() == ".h5"){
            assert(dataset=="basis");
			states = basis_io::read_basis_hdf5(bfile); 
		} else if (bfile.extension() == ".csv"){
            assert(dataset=="basis");
			states = basis_io::read_basis_csv(bfile); 
		} else {
			throw std::runtime_error(
					"Bad basis format: file must end with .csv or .h5");
		}

		for (idx_t i=0; i<states.size(); i++){
			state_to_index[states[i]] = i;
		}
	}
	
    // pls dont modify me without permission :o
	std::vector<state_t> states;
	std::unordered_map<state_t, idx_t, Uint128Hash, Uint128Eq> state_to_index;
};


inline int highest_set_bit(ZBasis::state_t x) {
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

	SymbolicPMROperator()  {
    }

	SymbolicPMROperator(const std::vector<char>& opstring, 
			const std::vector<int>& spin_ids)  {
        _construct(opstring, spin_ids);
    }

       SymbolicPMROperator(const std::string& spec) {
           auto [opstr, sgn] = parseSpec(spec);
           _construct(opstr, sgn);
       }

    // Mutates the uint128 state into Z X ([S+][S-] | [S-] [S+]) *state
    // and returns overall sign
	inline int applyState(ZBasis::state_t& state) const {
        const auto s = state.uint128;
        const auto d = down_mask.uint128;
        const auto u = up_mask.uint128;

        if ( (s & d) != 0 ) return 0;
        if ( (s & u) != u ) return 0;
        // X mask makes sense: 
        state ^= X_mask;

        // Explanation: There is a factor of -1 for every spin DOWN (i.e. 0) 
        // in state & Z_mask, i.e. every 1 in (~state) & Z_mask. 
        // if there are an even number, we have overall +1. 
        // If odd, there is overall -1.
        //
        //   popcnt_u128((~state) & Z_mask) - spin dn in Z mask 
        //
        return sign * (1 - 2 * (popcnt_u128((~state) & Z_mask) % 2) );
	}

    // returns sign of only possibly-nonzero entry, modifies J to its index
    int applyIndex(const ZBasis& basis, ZBasis::idx_t& J) const {
		ZBasis::state_t state = basis[J];

		int _sign = applyState(state);

        auto it = basis.state_to_index.find(state);
        if (it == basis.state_to_index.end()) {
            return 0;  // state not found in basis
        }
        J = it->second;
        return _sign;
//        J= basis.idx_of_state(state);
//        return _sign;
    }
    
	
	// Apply this operator to an input vector `in` and store result in `out`
	template <typename Orig, typename Dest>
	void apply(const ZBasis& basis, const Orig& in, Dest& out) const {	
		for (ZBasis::idx_t i = 0; i < basis.dim(); ++i) {
			ZBasis::idx_t J = i;
            auto c = applyIndex(basis, J) * in[i];
            out[J] += c;
		}
	}

	ZBasis::idx_t highest_set_bit() const {
		return ::highest_set_bit(X_mask | Z_mask | down_mask | up_mask);
	}


    inline bool operator==(const SymbolicPMROperator& other) const {
        return X_mask == other.X_mask &&
            Z_mask == other.Z_mask &&
            up_mask == other.up_mask &&
            down_mask == other.down_mask &&
            sign == other.sign;
    }
            

    SymbolicPMROperator operator*=(int a){
        sign *= a;
        return *this;
    }

    SymbolicPMROperator operator*=(const SymbolicPMROperator& b){
        // t->Z t->X t->P+ t->P- * Z X P+ P-
        // anticommute with Z
        sign *= (1 - 2 * (popcnt_u128(X_mask & b.Z_mask) % 2) ); 
        // t->Z  * Z * t->X t->P+ t->P- * X P+ P-
        // flip
        Z_mask ^= b.Z_mask;
        X_mask ^= b.X_mask;
        // sort out the mess with the +- 
        auto require_up = down_mask & b.X_mask;
        require_up |= up_mask & (~b.X_mask);
        auto require_down = up_mask & b.X_mask;
        require_down |= down_mask & (~b.X_mask);
        up_mask = b.up_mask | require_up;
        down_mask = b.down_mask | require_down;
        if ((up_mask & down_mask) != 0)
            sign =0; // we killed the operator
        return *this;
    }

    SymbolicPMROperator operator*(const SymbolicPMROperator& b){
        auto c = *this;
        c *= b;
        return c;
    }

    int get_sign() const noexcept {
        return sign;
    }

protected:
	ZBasis::state_t X_mask = 0;
	ZBasis::state_t Z_mask = 0;
	ZBasis::state_t down_mask = 0; // init_state & down_mask must be zero
	ZBasis::state_t up_mask = 0; // init_state & up_mask must be == up_mask
    int sign=1;
private:
    static std::pair<std::vector<char>, std::vector<int>> parseSpec(const std::string& spec) {
        std::vector<char> ops;
        std::vector<int> ids;

        std::istringstream ss(spec);
        std::string token;
        while (ss >> token) {
            if (token.size() < 2)
                throw std::invalid_argument("Invalid token in operator string: " + token);

            size_t i = 0;
            int idx = 0;
            while (i < token.size() - 1 && std::isdigit(token[i])) {
                idx = idx * 10 + (token[i] - '0');
                ++i;
            }

            char op = token[i];
            if (op != 'Z' && op !='z' && op != 'X' && op != 'x' && op != '+' && op != '-')
                throw std::invalid_argument("Invalid op '" + std::string(1, op) + "' in token: " + token);

            ops.push_back(op);
            ids.push_back(idx);
        }

        return {ops, ids};
    }

	void _construct(const std::vector<char>& opstring, 
			const std::vector<int>& spin_ids)  {

			if ( opstring.size() != spin_ids.size() ){
				throw bad_operator_spec(opstring, spin_ids);
			}
			for (int i=opstring.size()-1; i>=0; i--){
				auto J = spin_ids[i];
				switch (opstring[i]) {
					case 'x':
					case 'X':
						xor_bit(X_mask, J);
                        if (readbit(Z_mask, J)) sign *= -1;
                    break;
					case 'z':
					case 'Z':
						xor_bit(Z_mask, J);
						break;
					case '+':
                        // ++ situation
                        if (readbit(down_mask&X_mask, J)){
                            sign=0; break;
                        }
                        if (readbit(Z_mask, J))
                            sign *= -1;

						xor_bit(X_mask, J);
                        // if +-, don't reset the mask
                        if (!readbit(X_mask, J))
                            or_bit(up_mask, J);
                        else
                            or_bit(down_mask, J);
						break;
					case '-':
                        // -- situation
                        if (readbit(up_mask&X_mask, J)){
                            sign=0; break; 
                        }
                        if (readbit(Z_mask, J))
                            sign *= -1;

						xor_bit(X_mask, J);
                        // if -+, don't reset the mask
                        if (!readbit(X_mask, J))
                            or_bit(down_mask, J);
                        else
                            or_bit(up_mask, J);
						break;
					default:
						throw bad_operator_spec(opstring, spin_ids);
				}
			}
		}
};

inline SymbolicPMROperator operator*(int m, const SymbolicPMROperator& other){
    auto r = other;
    r *= m;
    return r;
}


template<ScalarLike coeff_t>
struct SymbolicOpSum {

	using Op = SymbolicPMROperator;


    SymbolicOpSum(){
    }

    SymbolicOpSum(const Op& o){
        terms.emplace_back(1.0,o);
    }

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



    LazyOpSum operator=(const LazyOpSum& other) = delete;

    LazyOpSum(const LazyOpSum& other) : basis(other.basis), ops(other.ops) {
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
			for (ZBasis::idx_t i = 0; i < basis.dim(); ++i)
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
            for (auto& [c, op] : ops.terms){
				ZBasis::idx_t J = j;
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
	const SymbolicOpSum<coeff_t> ops;
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




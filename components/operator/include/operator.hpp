#pragma once

#include "bittools.hpp"
#include <complex>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "bits.hpp"

#ifdef __APPLE__ // patch broken NEON optimization
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_NEON
#endif

//#include <pthash.hpp>

#include <basis_io.hpp>
#include <filesystem> // C++17
#include <unordered_map>
#include "operator.hpp"
#include <algorithm>

					  
namespace fs = std::filesystem;


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

//
//struct UInt128map {
//	using state_t = Uint128; // type which stores the computational basis state
//	using idx_t = uint64_t;  // the type to use for the indices themselves
//                            
//    // strategy: Use a configurable num. of MSB to place concrete brackets
//    UInt128map(){};
//    
//    void initialise(const std::vector<state_t>& states, int n_spins, int num_radix=10);
//
//protected:
//    int n_spins; // the index of the highest set bit in states
//                 // e.g. 100100 -> 5, as in (psi >> 5) & 0x1
//    int n_radix;
//    
//    // Given psi, define a topl-level index a = (psi.uin64[1] & hi_mask) >> hi_shift
//    // bounds[] gives 
//    // such that psi (if present) is in [states[a], states[b]).
//    std::vector<idx_t> bounds;
//
//    int hi_shift;
//    uint64_t hi_mask;
//
//
//    void initialise_lt64(const std::vector<state_t>& states);
//    void initialise_gt64(const std::vector<state_t>& states);
//};
//

struct SymbolicPMROperator;

/*
struct cust_xxhash_128 {
    typedef pthash::hash128 hash_type;

    static inline pthash::hash128 hash(const Uint128& val, uint64_t seed) {
        return XXH128(reinterpret_cast<char const*>(&val), sizeof(val), seed);
        //return pthash::hash128(val.uint64[0], val.uint64[1]);
    }
};


typedef pthash::dense_partitioned_phf<cust_xxhash_128,    // base hasher
                              pthash::opt_bucketer,  // bucketer
                              pthash::R_int,         // encoder type
                              true>          // minimal
    pthash_type;
*/

template<typename coeff_t>
struct SymbolicOpSum;

template<typename coeff_t, typename state_t>
void remove_annihilated_states(const SymbolicOpSum<coeff_t>& osm, std::vector<state_t>& states){
        size_t write_idx = 0;
        const size_t n_states = states.size();
        
        for (size_t read_idx = 0; read_idx < n_states; ++read_idx) {
            bool keep = false;
            const auto& this_state = states[read_idx];

            for (size_t op_idx = 0; op_idx < osm.off_diag_terms.size(); ++op_idx) {
                auto psi = this_state;
                if (osm.off_diag_terms[op_idx].second.applyState(psi) != 0) {
                    keep = true;
                    break;
                }
            }
            for (size_t op_idx = 0; op_idx < osm.diagonal_terms.size(); ++op_idx) {
                auto psi = this_state;
                if (osm.diagonal_terms[op_idx].second.applyState(psi) != 0) {
                    keep = true;
                    break;
                }
            }
            
            if (keep) {
                if (write_idx != read_idx) {
                    states[write_idx] = std::move(states[read_idx]);
                }
                ++write_idx;
            }
        }
        states.resize(write_idx);
}

struct ZBasisBase {
    using state_t = Uint128; // type which stores the computational basis state
    using idx_t = int64_t;  // the type to use for the indices themselves

	idx_t dim() const {
		return states.size();
	}

	idx_t size() const {
		return states.size();
	}

	inline state_t operator[](idx_t idx) const {
		return states[idx];
	}

	void load_from_file(const fs::path& bfile, const std::string& dataset="basis");

    template<typename coeff_t>
    void remove_null_states(const SymbolicOpSum<coeff_t>& osm) {
        remove_annihilated_states(osm, this->states);
    }

    protected:
        std::vector<state_t> states;
};



// Wrapper for a std::vector implementing indexable set semantics
// Represents a set of Sz product states spanning some subspace
struct ZBasisBST : public ZBasisBase
{
	// returns the index of a particular basis state
    // throws an error if not present
	idx_t idx_of_state(const state_t& state) const;

    // Inserts the states "others" into the basis, remembering the inserted states 
    // 'new_states'. Leaves "to_insert" holding a de-duplicated, sorted version of its original state.
	size_t insert_states(std::vector<state_t>& to_insert);

    int search(const state_t& state, idx_t& J) const;
};

struct ZBasisInterp : public ZBasisBST {
    void load_from_file(const fs::path& bfile, const std::string& dataset="basis");
    int search(const state_t& state, idx_t& J) const;
    protected:
    std::unordered_map<uint64_t, std::pair<idx_t, idx_t>> bounds;

    void find_bounds(); // finds the bounds
};



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
	inline int applyState(ZBasisBase::state_t& state) const {
        const auto s = state.uint128;
        const auto d = down_mask.uint128;
        const auto u = up_mask.uint128;

        if ( (s & d) != 0 ) return 0;
        if ( (s & u) != u ) return 0;

	    const int non_vanishing = static_cast<int>(((s&d) == 0) && ((s & u) == u ));
        // X mask makes sense: 
        state ^= X_mask;

        // Explanation: There is a factor of -1 for every spin DOWN (i.e. 0) 
        // in state & Z_mask, i.e. every 1 in (~state) & Z_mask. 
        // if there are an even number, we have overall +1. 
        // If odd, there is overall -1.
        //
        //   popcnt_u128((~state) & Z_mask) - spin dn in Z mask 
        //
        return sign * (1 - 2 * (popcnt_u128((~state) & Z_mask) % 2) ) * non_vanishing;
	// STICKY POINT -- mutates state even if it annihilates it
	}

    bool is_diagonal() const {
        return X_mask == ZBasisBase::state_t(0);
    }

    // returns sign of only possibly-nonzero entry, modifies J to its index
    template<Basis BasisType>
    int applyIndex(const BasisType& basis, ZBasisBase::idx_t& J) const {
		ZBasisBase::state_t state = basis[J];
		int _sign = applyState(state);
        if (_sign != 0){
             _sign *= basis.search(state, J);
        }
        return _sign; // 0 in case of miss
    }
	
	ZBasisBase::idx_t highest_set_bit() const {
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
        if ((up_mask & down_mask) != Uint128(0))
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
	ZBasisBase::state_t X_mask = 0;
	ZBasisBase::state_t Z_mask = 0;
	ZBasisBase::state_t down_mask = 0; // init_state & down_mask must be zero
	ZBasisBase::state_t up_mask = 0; // init_state & up_mask must be == up_mask
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
            const std::string valid_ops = "xyz+-pqXYZPQ";
	    bool valid = false;
	    for (auto s : valid_ops){
		    if (op == s) valid = true;
	    }
	    if (!valid){
		    throw std::invalid_argument("Invalid op '" + std::string(1, op) + "' in token: " + token);
	    }

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
                    case 'p':
                    case 'P':
                        // project-up, equivalent to S+ S-
                        // -- situation
                        if (readbit(up_mask&X_mask, J)){
                            sign=0; break; 
                        }
                        // if -+, don't reset the mask
                        if (!readbit(X_mask, J))
                            or_bit(down_mask, J);
                        else
                            or_bit(up_mask, J);
						break;
                    case 'Q':
                    case 'q':
                        // project-down, equivalent to S- S+
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


template<RealOrCplx coeff_t>
struct SymbolicOpSum {

	using Op = SymbolicPMROperator;


    SymbolicOpSum(){
    }

    SymbolicOpSum(const Op& o){
        add_term(1.0, o);
    }

	void add_term(coeff_t c, const Op& op) {
        if (op.is_diagonal()){
            diagonal_terms.emplace_back(c, op); // copies are fine
        } else {
            off_diag_terms.emplace_back(c, op);
        }
	}

	void operator+=(const Op& op) {
        add_term(1.0, op);
	}

	std::vector<std::pair<coeff_t, Op >> diagonal_terms;
	std::vector<std::pair<coeff_t, Op >> off_diag_terms;
};

// define a natural algebra of these boys
template <RealOrCplx coeff_t>
SymbolicOpSum<coeff_t> operator*(coeff_t c, SymbolicPMROperator op){
	SymbolicOpSum<coeff_t> s;
	s.add_term(c, op);
	return s;
}

template <RealOrCplx coeff_t>
SymbolicOpSum<coeff_t> operator*(SymbolicPMROperator op, coeff_t c){
	SymbolicOpSum<coeff_t> s;
	s.add_term(c, op);
	return s;
}



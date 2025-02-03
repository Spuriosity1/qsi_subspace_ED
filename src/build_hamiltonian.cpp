#include "arpackpp/arlssym.h"
#include "bittools.hpp"
#include "json.hpp"
#include "tetra_graph_io.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <admin.hpp>
#include <fstream>
#include <cstdio>
#include <set>

// arpack bindings
#include "arpackpp/arcomp.h"
#include "arpackpp/arlnsmat.h"
#include "arpackpp/arlutil.h"
#include "arpackpp/arlscomp.h"
#include "arpackpp/lcmatrxa.h"
#include "arpackpp/lcompsol.h"


using json = nlohmann::json;
using namespace std;

template< typename T >
typename std::vector<T>::iterator 
   insert_sorted( std::vector<T> & vec, T const& item )
{
    return vec.insert
        ( 
            std::upper_bound( vec.begin(), vec.end(), item ),
            item 
        );
}

struct RingOp {
	RingOp(const spin_set& ring_data){
		ring_mask = ring_data.bitmask;
		ring[0] = 0;
		ring[1] = 0;
		for (int i=0; i<ring_data.member_spin_ids.size(); i++){
			const auto& spin_id = ring_data.member_spin_ids[i];
			or_bit(ring[i%2], spin_id);
		}
	}
	Uint128 ring[2]; // the L-flippable / R-flippable states
	Uint128 ring_mask;

	bool flip_ring(Uint128 & state) const {
		const auto masked_state = state & ring_mask;
		if (masked_state == ring[0] || masked_state == ring[1]){
			state ^= ring_mask;
			return true;	
		}
		return false;
	}
};

typedef std::vector<std::pair<double, unsigned>> ham_entry_t; // pairs of form (coef, state_id)


class SparseEigenvalueProblem {
    public:
    SparseEigenvalueProblem(const std::vector<ham_entry_t>& v, int n_eigvals) : nev(n_eigvals){
        // expects storage such that v[col_id] is a sparse representation
        // of col col_id, sequence of pairs (value, row_id)
        precondition(v);
        
        valA = new arcomplex<double>[nnz]; // Hamiltonian values
        irow = new int[nnz]; // array of corresponding row indices
        pcol = new int[n+1]; // offsets of the different columns
        
        load_matrix(v);

        // allocate storage for the answers
        eigvals = new arcomplex<double>[nev];
        eigvecs = new arcomplex<double>[nev*n];
    }

    ~SparseEigenvalueProblem(){
        // deallocate
        delete[] valA;
        delete[] irow;
        delete[] pcol;
        delete[] eigvals;
        delete[] eigvecs;
    }

    void calc_eigenvalues(){
        // nev -> number of eigenvalues
        ARluNonSymMatrix<arcomplex<double>, double> A(n, nnz, valA, irow, pcol);
        ARluCompStdEig<double> dprob(nev, A, 
                "LR", // Lowest Real part
                0, // ncvp
                0.0 //tolp
                );
        assert(dprob.ParametersDefined());
        nconv = dprob.EigenValVectors(eigvals, eigvecs);
    }



    const int nev;

    int n_convergence() const {
        return nconv;
    }

    protected:

    void precondition(const std::vector<ham_entry_t>& v, bool check_dim = true ) {
        // sets n and nnz by iterating over the vector
        n=v.size();
        nnz=0;

        for (const auto& h: v){
            nnz += h.size();
            // ensure matrix is square
            for (const auto& [a, row] : h) {
                if (row > n) {
                    throw std::logic_error("Row in sparse matrix spec is larger than basis simension");
                }
            }
        }
    }

    void load_matrix(const std::vector<ham_entry_t>& v){
        // creating the matrix.
        pcol[0]=0;
        int curr_idx=0;
        for (int col=0; col<n; col++){
            for (auto& [a, row] : v[col]){
                valA[curr_idx] = a;
                irow[curr_idx] = row;
                curr_idx++;
            }
            pcol[col+1] = curr_idx; // defined position of next col
        }
    }

    int                n;     // Dimension of the problem.
    int                nnz;   // Number of nonzero elements in A.
    int      nconv=0; // number of iterations.



    arcomplex<double>* valA; // values of the matrix
    int* irow; // corresponding row indices
    int* pcol; // indices indicating beginning of columns

    arcomplex<double>* eigvals = nullptr;
    arcomplex<double>* eigvecs = nullptr;
};

class RingflipHamiltonian {
public:
	RingflipHamiltonian(const json &j, std::vector<Uint128> &&basis)
		: lat(j), basis(basis) {
			for (int i=1; i<basis.size(); i++){
				assert(basis[i].uint128 > basis[i-1].uint128);
			}
			for (const auto& ring : lat.rings){
				ring_ops.push_back(RingOp(ring));
			}
			ham_entries.reserve(basis.size());
			for (auto b : basis){
				ham_entry_t h;
				_calc_matrix_elements(b, h);
				ham_entries.push_back(h);
			}
		}

	size_t basis_index(const Uint128& state);

    void calc_eigenvalues(){
        auto prob = SparseEigenvalueProblem(ham_entries);
        


    }


protected:
	void _calc_matrix_elements(const Uint128& state, ham_entry_t& h);


    // constants set by the _precondition function
    int _nnz;
    //int _n_rows;


	// store the Hamiltonian in CSR format
	std::vector<ham_entry_t> ham_entries;

	const lattice lat;
	std::vector<Uint128> basis; // sorted vector, allows finding of states in logarithmic time
	
	std::vector<RingOp> ring_ops;
};


size_t RingflipHamiltonian::basis_index(const Uint128& state){
	size_t left=0;
	size_t right=basis.size();
	while (left < right) {
		size_t mid = left + (right - left) / 2;
		if (basis[mid] == state) {
			return mid; // Target found
		} else if (basis[mid].uint128 < state.uint128) {
			left = mid + 1; // Search right half
		} else {
			right = mid; // Search left half
		}
	}
	throw std::logic_error("Requested state does not exist in the basis");
}




void RingflipHamiltonian::_calc_matrix_elements(const Uint128& state, ham_entry_t& h){
	for (const auto& r: ring_ops){
		Uint128 tmp = state;
		if (r.flip_ring(tmp)) {
			size_t res_idx = basis_index(tmp);
			h.push_back(std::make_pair(1.0, res_idx));
		}
	}
}




std::vector<Uint128> import_basis(std::ifstream& ifs){
	assert(ifs.is_open());
	std::vector<Uint128> res;
	std::string line;
	while(std::getline(ifs, line)){
		if (line.size() < 2 ){ continue; }
		Uint128 x;
		auto n_matches = read_line(line, x);
		if (n_matches == 34){
			insert_sorted(res, x);
		}
	}
	return res;
}

int main (int argc, char *argv[]) {
	if (argc < 2) {
		printf("USAGE: %s <latfile: json\n", argv[0]);
	}

	std::string in_filename(argv[1]);
	string basis_filename=as_basis_file(in_filename);

    // Read the lattice setup
	ifstream in_ifs(in_filename);
    auto jdata = json::parse(in_ifs);
	in_ifs.close();

	ifstream basis_ifs(basis_filename);	
	RingflipHamiltonian H(jdata, import_basis(basis_ifs));
	basis_ifs.close();


	
	
}

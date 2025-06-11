#include "Spectra/Util/CompInfo.h"
#include "bittools.hpp"
#include <stdexcept>
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_NEON
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "bittools.hpp"
#include "admin.hpp"
#include "basis_io.hpp"

using namespace Eigen;
using json = nlohmann::json;

int main(int argc, char* argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: " << std::string(argv[0]) << " <basename>\n";
		return 1;
	}
	std::string base = argv[1];

	// Step 1: Load basis from CSV
	std::vector<Uint128> basis_states = basis_io::read_basis_csv(base + ".csv");
	const size_t N = basis_states.size();

	// Map basis element to index
	std::unordered_map<Uint128, int, Uint128Hash, Uint128Eq> state_to_index;
	for (size_t i = 0; i < N; ++i)
		state_to_index[basis_states[i]] = i;

	// Step 2: Load ring data from JSON
	std::ifstream jfile(base + ".json");
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;

	// Step 3: Build Hamiltonian
	using T = double;
	std::vector<Triplet<T>> triplets;

	for (const auto& ring : jdata["rings"]) {
		std::vector<int> spins = ring["member_spin_idx"];
		std::vector<int> signs = ring["signs"];
		if (spins.size() != signs.size()) {
			std::cerr << "Inconsistent ring format\n";
			continue;
		}

		auto state_L = Uint128(0);
		auto state_R = Uint128(0);
		auto mask = Uint128(0);

		for (size_t i = 0; i < signs.size(); ++i) {
			if (signs[i] == 1){
				or_bit(state_L, spins[i]); 
			} else {
				or_bit(state_R, spins[i]);
			}
		}
		mask = state_L | state_R;

		// Apply ring term to each basis state
		for (int i = 0; i < N; ++i) {
			Uint128 b = basis_states[i] & mask;
			if (b != state_L && b != state_R) {
				continue; // not flippable; nothing to do
			}
			auto it = state_to_index.find(b ^ mask);
			if (it == state_to_index.end()) {
				throw std::logic_error("Basis incomplete");
			} 

			int j = it->second;
			// H is Hermitian, add both (i,j) and (j,i)
			triplets.emplace_back(i, j, 1.0);
			assert(i != j);	
		}
	}

	SparseMatrix<T> H(N, N);
	H.setFromTriplets(triplets.begin(), triplets.end());

	// Step 4: Diagonalize with Spectra
	using OpType=Spectra::SparseSymMatProd<T>;
	OpType op(H);
	Spectra::SymEigsSolver<OpType> eigs(op, 6, std::min(20, N));
	eigs.init();
	int nconv = eigs.compute();

	if (eigs.info() == Spectra::CompInfo::Successful) {
		VectorXd evals = eigs.eigenvalues();
		std::cout << "Eigenvalues:\n" << evals.head(nconv) << "\n";
	} else {
		std::cerr << "Spectra failed\n";
		return 1;
	}

	return 0;
}

#include <argparse/argparse.hpp>

#include "Spectra/Util/CompInfo.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>


#include "Spectra/Util/SelectionRule.h"
#include "operator.hpp"

#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;
using json = nlohmann::json;

std::string get_basis_file(const argparse::ArgumentParser& prog){
// Determine basis_file default if not set
	std::string basis_file;
	if (prog.is_used("--basis_file")) {
		basis_file = prog.get<std::string>("--basis_file");
	} else {
		// Replace extension
		fs::path path(prog.get<std::string>("lattice_file"));
		if (path.extension() == ".json") {
			path.replace_extension(".0.basis.h5");
		} else {
			// fallback if extension isn't ".json"
			path += ".0.basis.h5";
		}
		basis_file = path.string();
	}
	return basis_file;
}

template <typename T>
struct is_sym_solver : std::false_type {};

template <typename OpType>
struct is_sym_solver<Spectra::SymEigsSolver<OpType>> : std::true_type {};

template <typename T>
struct is_gen_solver : std::false_type {};

template <typename OpType>
struct is_gen_solver<Spectra::GenEigsSolver<OpType>> : std::true_type {};

template <typename SolverT>
constexpr Spectra::SortRule default_sort_rule() {
    if constexpr (is_sym_solver<SolverT>::value) {
        return Spectra::SortRule::SmallestAlge;
    } else if constexpr (is_gen_solver<SolverT>::value) {
        return Spectra::SortRule::SmallestReal;
    } else {
        static_assert([]{ return false; }(), "Unsupported solver type");
    }
}

int main(int argc, char* argv[]) {
	argparse::ArgumentParser prog("build_ham");
	prog.add_argument("lattice_file");
	prog.add_argument("-b", "--basis_file");
	prog.add_argument("-g")
		.help("ring exchange constants (H = sum g [O + O'])")
        .nargs(1,4)
		.default_value(std::vector<double>{1.0})
		.scan<'g', double>();
	prog.add_argument("--ncv", "-k")
		.help("Krylov dimension, shoufl be > 2*n_eigvals")
		.default_value(15)
		.scan<'i', int>();
	prog.add_argument("--n_eigvals", "-n")
		.help("Number of eigenvlaues to compute")
		.default_value(5)
		.scan<'i', int>();
	prog.add_argument("--save_matrix")
		.help("Flag to get the solver to export a rep of the matrix")
		.default_value(false)
		.implicit_value(true);
		
    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
		std::cerr << err.what() << std::endl;
		std::cerr << prog;
        std::exit(1);
    }

	auto g = prog.get<std::vector<double>>("-g");
	if (g.size() < 4){
		std::cout<< "Assuming uniform g...\n";
		g.resize(4);
		for (int i=1; i<4; i++) {
			g[i] = g[0];
		}
	}
			


	// Step 1: Load basis from CSV
	ZBasis basis;
	basis.load_from_file(get_basis_file(prog));

	// Step 2: Load ring data from JSON
	std::ifstream jfile(prog.get<std::string>("lattice_file"));
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;


	using T=double;
	SymbolicOpSum<T> H_sym;

	try {
		auto version=jdata.at("__version__");
		if ( atof(version.get<std::string>().c_str()) < 1.0 ){
			throw std::runtime_error("JSON file is old, API version 1.0 is required");
		}
	} catch (const json::out_of_range& e){
		throw std::runtime_error("__version__ field missing, suspect an old file");
	} 	

	for (const auto& ring : jdata.at("rings")) {
		std::vector<int> spins = ring.at("member_spin_idx");

		std::vector<char> ops;
		std::vector<char> conj_ops;
		for (auto s : ring.at("signs")){
			ops.push_back( s == 1 ? '+' : '-');
			conj_ops.push_back( s == 1 ? '-' : '+');
		}
		
		int sl = ring.at("sl").get<int>();
		auto O   = SymbolicPMROperator(     ops, spins);
		auto O_h = SymbolicPMROperator(conj_ops, spins);
		H_sym.add_term(g[sl], O);
		H_sym.add_term(g[sl], O_h);
	}

	auto H = LazyOpSum(basis, H_sym);	

	// materialise
	auto H_sparsemat = H.toSparseMatrix();

	if (prog.get<bool>("--save_matrix")){
		Eigen::saveMarket(H_sparsemat, "H.mtx");
		std::cout<<"Saved to H.mtx"<<std::endl;
	}

	// Step 4: Diagonalize with Spectra
	using OpType = LazyOpSumProd<double>;
	OpType op(H);

	//using OpType = Spectra::SparseSymMatProd<double>;	
	//OpType op(H_sparsemat);

	using Solver = Spectra::SymEigsSolver<OpType>;
	//using Solver = Spectra::GenEigsSolver<OpType>;

	
	size_t n_eigvals = prog.get<int>("--n_eigvals");
	size_t ncv = prog.get<int>("--ncv");
	if (ncv < 2*n_eigvals){
		std::cout<<"Warning: ncv is very small, recommend at leaast 2*n_eigvals";
	}

	ncv = std::min(ncv, basis.dim());
	n_eigvals = std::min(ncv-1, n_eigvals );

	std::cout << "Using ncv="<<ncv<<" n_eigvals="<<n_eigvals<<std::endl;
	Solver eigs(op, n_eigvals, ncv);
	eigs.init();
	Spectra::SortRule sortrule = default_sort_rule<Solver>();
	int nconv = eigs.compute(
			sortrule,
			1000, /*maxit*/
			1e-10, /*tol*/
			sortrule
			);
	

	if (eigs.info() == Spectra::CompInfo::Successful) {
		VectorXd evals = eigs.eigenvalues().real();
		std::cout << "Eigenvalues:\n" << evals.head(nconv) << "\n";
	} else {
		std::cerr << "Spectra failed\n";
		return 1;
	}

	return 0;
}

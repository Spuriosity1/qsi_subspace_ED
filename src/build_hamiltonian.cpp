#include <argparse/argparse.hpp>

#include <Eigen/Eigenvalues>

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


template<typename OpType, typename T>
void compute_spectrum_iterative(const T ham, VectorXd& evals, MatrixXcd evecs, const argparse::ArgumentParser& settings)
{
    OpType op(ham);

    // parse ncv and n_eigvals
	size_t n_eigvals = settings.get<int>("--n_eigvals");
	size_t ncv = settings.get<int>("--ncv");
	if (ncv < 2*n_eigvals){
		std::cout<<"Warning: ncv is very small, recommend at leaast 2*n_eigvals";
	}

	ncv = std::min(ncv, static_cast<decltype(ncv)>(ham.rows()));
	n_eigvals = std::min(ncv-1, n_eigvals );

    auto max_it = settings.get<int>("--max_iters");
    auto tol    = settings.get<double>("--tol");

	std::cout << "Using ncv="<<ncv<<" n_eigvals="<<n_eigvals<<std::endl;
    using Solver = Spectra::SymEigsSolver<OpType>;
    Solver eigs(op, n_eigvals, ncv);
    eigs.init();
    Spectra::SortRule sortrule = default_sort_rule<Solver>();
    auto nconv = eigs.compute(
            sortrule,
            max_it, /*maxit*/
            tol, /*tol*/
            sortrule
            );

    if (eigs.info() == Spectra::CompInfo::Successful) {
        evals = eigs.eigenvalues().head(nconv);
        evecs = eigs.eigenvectors(nconv);
    } else {
        std::cerr << "Spectra failed\n";
        throw std::runtime_error("Eigenvalue decomposition failed");
    }
}


void compute_eigenspectrum_dense(const MatrixXd& ham, Eigen::VectorXd& e, Eigen::MatrixXd& v,
    const argparse::ArgumentParser& settings)
{
	size_t n_eigvals = settings.get<int>("--n_eigvals");
	n_eigvals = std::min(static_cast<decltype(n_eigvals)>(ham.rows()), n_eigvals );

    SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(ham);
    // truncate to # requested eigvals


    if (eigs.info() == ComputationInfo::Success) {
        e = eigs.eigenvalues().head(n_eigvals);
        v = eigs.eigenvectors().leftCols(n_eigvals);
    } else {
        std::cerr << "Spectra failed\n";
        throw std::runtime_error("Eigenvalue decomposition failed");
    }
}


void build_hamiltonian(SymbolicOpSum<double>& H_sym, const nlohmann::json& jdata){

	try {
		auto version=jdata.at("__version__");
		if ( stof(version.get<std::string>()) < 1.0 ){
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

    auto atoms = jdata.at("atoms");



	Matrix<double, 4, 3> local_z;
	local_z <<  1.,  1,  1,
				1., -1, -1,
			   -1.,  1, -1,
			   -1., -1,  1;
	local_z /= std::sqrt(3.0);

	Vector3d B;

    for (const auto& bond : jdata.at("bonds")){
        auto i = bond.at("from_idx").get<int>();
        auto j = bond.at("to_idx").get<int>();
        int sl = stoi(atoms[i].at("sl".get<std::string>())

		// Convert row of local_z to Eigen::Vector3d
		double zi = B.dot(local_z.row(i));
		double zj = B.dot(local_z.row(j));

		zz_coeff = -4.0 * Jpm * zi * zj;

		H_sym.add_term(zz_coeff, SymbolicPMROperator("zz", {i, j}));
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
    prog.add_argument("--max_iters")
        .help("Max steps for iterative solver")
        .default_value(1000)
        .scan<'i', int>();
    prog.add_argument("--tol")
        .help("Tolerance iterative solver")
        .default_value(1e-10)
        .scan<'g', double>();

    prog.add_argument("--algorithm", "-a")
        .choices("dense","sparse","mfsparse")
        .default_value("sparse")
        .help("Variant of ED algorithm to run. dense is best for small problems, mfsparse is a matrix free method that trades off speed for memory.");
		
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
    std::cout<<"Loading basis..."<<std::endl;
	ZBasis basis;
	basis.load_from_file(get_basis_file(prog));
    std::cout<<"Done!"<<std::endl;

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

	build_hamiltonian(H_sym);

	auto H = LazyOpSum(basis, H_sym);	




    ////////////////////////////////////////
    // Do the diagonalisation

    VectorXd eigvals;
    MatrixXd eigvecs;

    if (prog.get<std::string>("--algorithm") == "dense") {

        // materialise
        std::cout<<"Materialising dense matrix..."<<std::endl;
        auto H_densemat = H.toSparseMatrix();
        std::cout<<"Done!"<<std::endl;

       compute_eigenspectrum_dense(H_densemat, eigvals, eigvecs, prog);
    } else if (prog.get<std::string>("--algorithm") == "sparse") {
        
        // materialise
        std::cout<<"Materialising sparse matrix..."<<std::endl;
        auto H_sparsemat = H.toSparseMatrix();
        std::cout<<"Done!"<<std::endl;

        compute_spectrum_iterative< Spectra::SparseSymMatProd<double>>(H_sparsemat, eigvals, eigvecs, prog);

        if (prog.get<bool>("--save_matrix")){
            Eigen::saveMarket(H.toSparseMatrix(), "H.mtx");
            std::cout<<"Saved to H.mtx"<<std::endl;
        }
    } else if (prog.get<std::string>("--algorithm") == "mfsparse"){
        compute_spectrum_iterative< LazyOpSumProd<double> >(H, eigvals, eigvecs, prog);
    }


    std::cout << "Eigenvalues:\n" << eigvals << "\n\n";

	


	return 0;
}


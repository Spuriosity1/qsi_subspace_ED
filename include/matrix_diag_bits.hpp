#pragma once

#include <string>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>


#include <Eigen/Eigenvalues>

#include "Spectra/Util/CompInfo.h"
#include "operator.hpp"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;


inline std::string get_basis_file(const argparse::ArgumentParser& prog){
// Determine basis_file default if not set
	std::string basis_file;
    int n_spinons = prog.get<int>("--n_spinons");

    std::string ext = "." + std::to_string(n_spinons) + ".basis";
	if (prog.is_used("--sector")) {
        ext += ".partitioned";
    } 
    ext += ".h5";
    
    // Replace extension: json-> ext
    std::filesystem::path path(prog.get<std::string>("lattice_file"));
    if (path.extension() == ".json") {
        path.replace_extension(ext);
    } else {
        // fallback if extension isn't ".json"
        path += ext;
    }
    basis_file = path.string();
	
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


template<typename OpType, typename T, ScalarLike S>
void compute_spectrum_iterative(const T ham, VectorXd& evals, MatrixX<S>& evecs, const argparse::ArgumentParser& settings)
{
    OpType op(ham);

    // parse ncv and n_eigvals
	size_t n_eigvals = settings.get<int>("--n_eigvals");
	size_t n_eigvecs = settings.get<int>("--n_eigvecs");
	size_t ncv = settings.get<int>("--ncv");
	if (ncv < 2*n_eigvals){
		std::cout<<"Warning: ncv is very small, recommend at leaast 2*n_eigvals";
	}

	ncv = std::min(ncv, static_cast<decltype(ncv)>(ham.rows()));
	n_eigvals = std::min(ncv-1, n_eigvals );
	n_eigvecs = std::min(n_eigvecs, n_eigvals);

    auto max_it = settings.get<int>("--max_iters");
    auto tol    = settings.get<double>("--tol");

	std::cout << "Using ncv="<<ncv<<" n_eigvals="<<n_eigvals<<std::endl;
	std::cout << "Eigvecs will be truncated to n_eigvecs="<<n_eigvecs<<std::endl;
    using Solver = Spectra::SymEigsSolver<OpType>;
    Solver eigs(op, n_eigvals, ncv);
    eigs.init();
    Spectra::SortRule sortrule = default_sort_rule<Solver>();

	std::cout << "Diagonalising..."<<std::endl;
    auto nconv = eigs.compute(
            sortrule,
            max_it, /*maxit*/
            tol, /*tol*/
            sortrule
            );

	std::cout << "Done!"<<std::endl;

    if (eigs.info() == Spectra::CompInfo::Successful) {
        evals = eigs.eigenvalues().head(nconv);
        evecs = eigs.eigenvectors(std::min(static_cast<size_t>(nconv), n_eigvecs));
    } else {
        std::cerr << "Spectra failed\n";
        throw std::runtime_error("Eigenvalue decomposition failed");
    }
}


inline void compute_eigenspectrum_dense(const MatrixXd& ham, Eigen::VectorXd& e, Eigen::MatrixXd& v,
    const argparse::ArgumentParser& settings)
{
	size_t n_eigvals = settings.get<int>("--n_eigvals");
	size_t n_eigvecs = settings.get<int>("--n_eigvecs");
	n_eigvals = std::min(static_cast<decltype(n_eigvals)>(ham.rows()), n_eigvals );
	n_eigvecs = std::min(n_eigvecs, n_eigvals);

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

std::pair<VectorXd, MatrixXd>
inline diagonalise_real(const LazyOpSum<double>& H, const argparse::ArgumentParser &prog) {
  VectorXd eigvals;
  MatrixXd eigvecs;

  if (prog.get<std::string>("--algorithm") == "dense") {

    // materialise
    std::cout << "Materialising dense matrix..." << std::endl;
    auto H_densemat = H.toSparseMatrix();
    std::cout << "Done!" << std::endl;

    compute_eigenspectrum_dense(H_densemat, eigvals, eigvecs, prog);
  } else if (prog.get<std::string>("--algorithm") == "sparse") {

    // materialise
    std::cout << "Materialising sparse matrix..." << std::endl;
    auto H_sparsemat = H.toSparseMatrix();
    std::cout << "Done!" << std::endl;

    compute_spectrum_iterative<Spectra::SparseSymMatProd<double>>(
        H_sparsemat, eigvals, eigvecs, prog);

    if (prog.get<bool>("--save_matrix")) {
      Eigen::saveMarket(H.toSparseMatrix(), "H.mtx");
      std::cout << "Saved to H.mtx" << std::endl;
    }
  } else if (prog.get<std::string>("--algorithm") == "mfsparse") {
    compute_spectrum_iterative<LazyOpSumProd<double>>(H, eigvals, eigvecs,
                                                      prog);
  }
  return std::make_pair(eigvals, eigvecs);
}

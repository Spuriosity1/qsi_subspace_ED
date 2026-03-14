#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "operator_matrix.hpp"
#include <random>
#include "timeit.hpp"
#include <fstream>
#include "common_bits.hpp"
//#include "lanczos.hpp"



using json = nlohmann::json;


int main(int argc, char* argv[]){
    
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file");
	prog.add_argument("-s", "--sector");
    prog.add_argument("--basis_file", "-b")
        .help("A basis file (HDF5 format). Defaults to ${lattice_file%.json}.h5");
	prog.add_argument("--n_spinons")
        .default_value(0)
        .scan<'i', int>();

    prog.add_argument("--seed")
        .help("Seed for the RNG")
        .scan<'i', unsigned int>()
        .default_value(0u);


    try {
        prog.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << prog;
        return 1;
    }


    unsigned int seed = prog.get<unsigned int>("--seed");

    ZBasisBST    basis_bst;
    ZBasisInterp basis_interp;
    ZBasisBSTFast basis_fast;

	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;


    std::cout<<"[BST]   Loading basis..."<<std::endl;
    load_basis(basis_bst, prog);
    std::cout<<"[BST]   Done! dim="<<basis_bst.dim()<<std::endl;

    std::cout<<"[interp] Loading basis..."<<std::endl;
    load_basis(basis_interp, prog);
    std::cout<<"[interp] Done! dim="<<basis_interp.dim()<<std::endl;

    std::cout<<"[fast]  Loading basis..."<<std::endl;
    load_basis(basis_fast, prog);
    std::cout<<"[fast]  Done! dim="<<basis_fast.dim()<<std::endl;


	using T=double;
	SymbolicOpSum<T> H_sym;

    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    auto H_bst   = LazyOpSum(basis_bst,   H_sym);
    auto H_interp = LazyOpSum(basis_interp, H_sym);
    auto H_fast  = LazyOpSum(basis_fast,  H_sym);

    std::vector<double> v, u;
    v.resize(basis_bst.dim());
    u.resize(basis_bst.dim());
    std::mt19937 rng(seed);
    projED::set_random_unit(v, rng);

    std::cout<<"[BST]   Apply..."<<std::endl;
    std::fill(u.begin(), u.end(), 0);
    TIMEIT("u += Av", H_bst.evaluate_add(v.data(), u.data());)

    std::cout<<"[interp] Apply..."<<std::endl;
    std::fill(u.begin(), u.end(), 0);
    TIMEIT("u += Av", H_interp.evaluate_add(v.data(), u.data());)

    std::cout<<"[fast]  Apply..."<<std::endl;
    std::fill(u.begin(), u.end(), 0);
    TIMEIT("u += Av", H_fast.evaluate_add(v.data(), u.data());)

    return 0;
}

#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "operator_matrix.hpp"
#include <random>
#include "timeit.hpp"



using json = nlohmann::json;


int main(int argc, char* argv[]){
    
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file");
	prog.add_argument("-s", "--sector");
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

	ZBasisBST basis;
	ZBasisHashmap basis_h;
   
	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;


    std::cout<<"[BST]  Loading basis..."<<std::endl;
    load_basis(basis, prog);
    std::cout<<"[BST]  Done! Basis dim="<<basis.dim()<<std::endl;

    std::cout<<"[Hash] Loading basis..."<<std::endl;
    load_basis(basis_h, prog);
    std::cout<<"[Hash] Done! "<<std::endl;

	using T=double;
	SymbolicOpSum<T> H_sym;
    
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    auto H_bst = LazyOpSum(basis, H_sym);
    auto H_hash = LazyOpSum(basis_h, H_sym);

    std::vector<double> v, u1, u2;
    v.resize(basis.dim());
    u1.resize(basis.dim());
    u2.resize(basis.dim());
    std::mt19937 rng(seed);
    set_random_unit(v, rng);

    std::fill(u1.begin(), u1.end(), 0);
    std::fill(u2.begin(), u2.end(), 0);

    std::cout<<"[BST]  Apply..."<<std::endl;
    TIMEIT("u += Av", H_bst.evaluate_add(v.data(), u1.data());)
    std::cout<<"[Hash] Apply..."<<std::endl;
    TIMEIT("u += Av", H_hash.evaluate_add(v.data(), u2.data());)
//    TIMEIT("u <- Av", H.evaluate(v.data(), u.data());)
    return 0;
}

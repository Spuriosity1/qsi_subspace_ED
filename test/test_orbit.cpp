#include "argparse/argparse.hpp"
#include <nlohmann/json.hpp>
#include "hamiltonian_setup.hpp"
#include "operator_matrix.hpp"
#include "group_theory.hpp"
#include <fstream>
#include "basis_format_bits.hpp"


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

//    unsigned int seed = prog.get<unsigned int>("--seed");

   
	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;

    std::cout<<"[G]    Deducing symmetries..."<<std::endl;
    std::vector<PermutationGroup<Uint128>::perm_t> generators;

    PermutationGroup<Uint128>G(generators);
    std::vector<double> chi;
    std::fill(chi.begin(), chi.end(), 1.0); 

    Representation R(G, chi);
	GAdaptedZBasisBST basis(R);

    std::cout<<"[BST]  Loading basis..."<<std::endl;
    load_basis(basis, prog);
    std::cout<<"[BST]  Done! Basis dim="<<basis.dim()<<std::endl;


    using T = double;
	SymbolicOpSum<T> H_sym;
    
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    auto H_bst = LazyOpSum(basis, H_sym);

    

}

#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "operator_matrix.hpp"
#include "operator_mpi.hpp"
#include <random>
#include "timeit.hpp"
//#include "lanczos.hpp"
#include "common_bits.hpp"
#include <fstream>
#include "basis_format_bits.hpp"




using json = nlohmann::json;


int main(int argc, char* argv[]){
    
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file");
    prog.add_argument("--basis_file", "-b")
        .help("A basis file (HDF5 format). Defaults to ${lattice_file%.json}.h5");
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
    
    MPI_Init(NULL, NULL);


	MPI_ZBasisBST basis;
   
	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;

    // Step 2: load and partition the basis
    std::cout<<"[MPI_BST]  Loading basis..."<<std::endl;
    SparseMPIContext ctx = load_basis(basis, prog);
    std::cout<<"[MPI_BST]  Done! Basis dim="<<basis.dim()<<std::endl;

    assert(ctx.local_block_size() == basis.size());

    for (size_t il=0; il<basis.size(); il++){
        assert(ctx.rank_of_state(basis[il]) == ctx.my_rank);
    }

    MPI_Finalize();
    return 0;
}

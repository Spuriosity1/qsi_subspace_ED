#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "operator_mpi.hpp"
#include "common_bits_mpi.hpp"
#include <random>
#include "timeit.hpp"
#include <fstream>


using json = nlohmann::json;


int main(int argc, char* argv[]){
    
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file");
	prog.add_argument("-s", "--sector");
	prog.add_argument("--n_spinons")
        .default_value(0)
        .scan<'i', int>();
    prog.add_argument("--basis_file", "-b")
        .help("A basis file (HDF5 format). Defaults to ${lattice_file%.json}.h5");

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
    TIMEIT("Loading basis", MPIContext ctx = load_basis(basis, prog);)
    std::cout<<"[rank "<<ctx.my_rank<<"] Done! Basis dim="<<basis.dim()<<std::endl;


	using T=double;
	SymbolicOpSum<T> H_sym;
    
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    auto H_mpi = MPILazyOpSum(basis, H_sym, ctx);

    std::vector<double> v_local, u_local;
    v_local.resize(ctx.local_block_size());

    assert(ctx.local_block_size() == basis.dim());
    u_local.resize(ctx.local_block_size());

    std::mt19937 rng(seed);
    projED::set_random_unit_mpi(v_local, rng);

    std::fill(u_local.begin(), u_local.end(), 0);

    std::cout<<"[BST_MPI "<<ctx.my_rank<<"]  Apply..."<<std::endl;
    // NOTE: add the local block offset to stay correct
    TIMEIT("u += Av", H_mpi.evaluate_add(v_local.data(), u_local.data());)

    MPI_Finalize();
    return 0;
}

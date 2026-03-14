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


    prog.add_argument("--trim")
        .default_value(false)
        .implicit_value(true);


    prog.add_argument("--rebalance")
        .default_value(false)
        .implicit_value(true);


    try {
        prog.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << prog;
        return 1;
    }

    unsigned int seed = prog.get<unsigned int>("--seed");
    
    MPI_Init(NULL, NULL);

    ZBasisBST_HashMPI     basis_bst;
    ZBasisInterp_HashMPI  basis_interp;
    ZBasisBSTFast_HashMPI basis_fast;

	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;

    MPIHashContext ctx;

    TIMEIT("[BST]    load", load_basis(basis_bst,    prog);)
    std::cout<<"[rank "<<ctx.my_rank<<"] BST    dim="<<basis_bst.dim()<<std::endl;

    TIMEIT("[interp] load", load_basis(basis_interp, prog);)
    std::cout<<"[rank "<<ctx.my_rank<<"] interp dim="<<basis_interp.dim()<<std::endl;

    TIMEIT("[fast]   load", load_basis(basis_fast,   prog);)
    std::cout<<"[rank "<<ctx.my_rank<<"] fast   dim="<<basis_fast.dim()<<std::endl;


	using T=double;
	SymbolicOpSum<T> H_sym;

    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    if (prog.get<bool>("--trim")){
        TIMEIT("trim", basis_bst.remove_null_states(H_sym);
                       basis_interp.remove_null_states(H_sym);
                       basis_fast.remove_null_states(H_sym);)
    }

    auto H_bst    = MPILazyOpSum(basis_bst,    H_sym, ctx);
    auto H_interp = MPILazyOpSum(basis_interp, H_sym, ctx);
    auto H_fast   = MPILazyOpSum(basis_fast,   H_sym, ctx);
    H_bst.allocate_temporaries();
    H_interp.allocate_temporaries();
    H_fast.allocate_temporaries();

    std::vector<double> v_local(basis_bst.dim()), u_local(basis_bst.dim());
    std::mt19937 rng(seed);
    projED::set_random_unit_mpi(v_local, rng);

    std::cout<<"[BST_MPI "<<ctx.my_rank<<"] Apply..."<<std::endl;
    std::fill(u_local.begin(), u_local.end(), 0);
    TIMEIT("[BST]    u += Av", H_bst.evaluate_add(v_local.data(), u_local.data());)

    std::cout<<"[interp_MPI "<<ctx.my_rank<<"] Apply..."<<std::endl;
    std::fill(u_local.begin(), u_local.end(), 0);
    TIMEIT("[interp] u += Av", H_interp.evaluate_add(v_local.data(), u_local.data());)

    std::cout<<"[fast_MPI "<<ctx.my_rank<<"] Apply..."<<std::endl;
    std::fill(u_local.begin(), u_local.end(), 0);
    TIMEIT("[fast]   u += Av", H_fast.evaluate_add(v_local.data(), u_local.data());)

    MPI_Finalize();
    return 0;
}

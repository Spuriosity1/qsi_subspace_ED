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

    prog.add_argument("--n_buffers")
        .help("number of comm buffers")
        .scan<'i', size_t>()
        .default_value(static_cast<size_t>(2));

    prog.add_argument("--trim")
        .default_value(false)
        .implicit_value(true);


    prog.add_argument("--algo", "-a")
        .choices("0","1","2")
        .scan<'i', unsigned int>();


    try {
        prog.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << prog;
        return 1;
    }

    unsigned int seed = prog.get<unsigned int>("--seed");
    size_t n_buffers = prog.get<size_t>("--n_buffers");
    
    MPI_Init(NULL, NULL);

	ZBasisBST_MPI basis;
   
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
    TIMEIT("Loading basis", SparseMPIContext ctx = load_basis(basis, prog);)
    std::cout<<"[rank "<<ctx.my_rank<<"] Done! Basis dim="<<basis.dim()<<std::endl;


	using T=double;
	SymbolicOpSum<T> H_sym;
    
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);


    if (prog.get<bool>("--trim")){
        TIMEIT("remove unneeded elements",
            basis.remove_null_states(H_sym, ctx);
        )
    }

    std::vector<MPILazyOpSumBase<double, ZBasisBST_MPI>*> operators;

    TIMEIT("H_mpi_pipeP construct",
    auto H_mpi_pipeP = MPILazyOpSumPipePrealloc<double, ZBasisBST_MPI>(basis, H_sym, ctx, n_buffers);
    )

    std::vector<std::string> names = {"MPI batch", "MPI pipe", "MPI pipe prealloc"};


    std::vector<double> v_local, u_local;
    v_local.resize(ctx.local_block_size());
    u_local.resize(ctx.local_block_size());

    std::mt19937 rng(seed);
    projED::set_random_unit_mpi(v_local, rng);

    assert(ctx.local_block_size() == basis.dim());

    TIMEIT("allocating temporaries",
            H_mpi_pipeP.allocate_temporaries();
          )

    std::cout<<"[rank "<<ctx.my_rank<<"] op construct finish"<<std::endl;

    std::fill(u_local.begin(), u_local.end(), 0);

    std::cout<<"[BST_MPI "<<ctx.my_rank<<"]  Apply..."<<std::endl;
    // NOTE: add the local block offset to stay correct
    TIMEIT("u += Av", H_mpi_pipeP.evaluate_add(v_local.data(), u_local.data());)
    

    MPI_Finalize();
    return 0;
}

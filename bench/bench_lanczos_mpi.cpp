#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include <random>
#include "timeit.hpp"
#include <fstream>
#include "common_bits.hpp"
#include "lanczos_mpi.hpp"



using json = nlohmann::json;

using namespace projED;


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

    prog.add_argument("--dim")
        .help("Matrix dimension")
        .scan<'i', int>()
        .default_value(100);

    prog.add_argument("--krylov_dim", "-k")
        .help("Krylov space dimension")
        .scan<'i', int>()
        .default_value(30);

    prog.add_argument("--max_iterations", "-M")
        .help("Max iterations before giving up")
        .scan<'i', int>()
        .default_value(5000);

    prog.add_argument("--min_iterations", "-M")
        .help("Min iterations")
        .scan<'i', int>()
        .default_value(30);

    prog.add_argument("--abs_tol", "-a")
        .help("Lanczos eigval atol e.g. -8 = 1e-8")
        .scan<'i', int>()
        .default_value(-8);

    prog.add_argument("--rel_tol", "-r")
        .help("Lanczos eigval rtol e.g. -8 = 1e-8")
        .scan<'i', int>()
        .default_value(-8);


    try {
        prog.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << prog;
        return 1;
    }

    MPI_Init(nullptr, nullptr);


    unsigned int seed = prog.get<unsigned int>("--seed");

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
    TIMEIT("[MPI_BST] Loading basis...",
    SparseMPIContext ctx = load_basis(basis, prog);
    )
    std::cout<<"[MPI_BST]  Done! Basis dim="<<basis.dim()<<std::endl;

	using T=double;
	SymbolicOpSum<T> H_sym;
    
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    auto H = MPILazyOpSum(basis, H_sym, ctx);

    std::vector<double> v, u1;
    v.resize(basis.dim());

    u1.resize(basis.dim());
    std::mt19937 rng(seed);
    projED::set_random_unit(v, rng);

    lanczos_mpi::Settings settings(ctx);
    settings.krylov_dim = prog.get<int>("--krylov_dim");
    settings.abs_tol = pow(10, prog.get<int>("--abs_tol"));
    settings.rel_tol = pow(10, prog.get<int>("--rel_tol"));

    settings.max_iterations = prog.get<int>("--max_iterations");
    settings.min_iterations = prog.get<int>("--min_iterations");

    settings.verbosity = 3;
    settings.calc_eigenvector = true;


    using coeff_t = double;
    RealApplyFn evadd = [&H](const coeff_t* x, coeff_t* y){
        H.evaluate_add(x, y);
    };
    double eigval_lanczos = 0.0;
    // Output vector
    std::vector<double> v0(basis.dim());
    auto res = lanczos_mpi::eigval0(evadd, eigval_lanczos, v0, settings);
    if (ctx.my_rank == 0){
    std::cout <<res;
    }

    MPI_Finalize();

    return 0;
}

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


    prog.add_argument("--notrim")
        .default_value(false)
        .implicit_value(true);


    prog.add_argument("--rebalance")
        .default_value(false)
        .implicit_value(true);

    prog.add_argument("--basis-type")
        .help("Basis search structure: bst | interp | fast  (default: run all three)")
        .default_value(std::string("all"));

    try {
        prog.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << prog;
        return 1;
    }

    auto bt = prog.get<std::string>("--basis-type");
    if (bt != "all" && bt != "bst" && bt != "interp" && bt != "fast") {
        std::cerr << "Invalid --basis-type '" << bt << "'. Must be one of: bst, interp, fast, all\n";
        std::cerr << prog;
        return 1;
    }

    unsigned int seed = prog.get<unsigned int>("--seed");

    MPI_Init(NULL, NULL);

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

	using T=double;
	SymbolicOpSum<T> H_sym;

    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    auto bench_one = [&](auto& basis, const char* tag) {
        TIMEIT((std::string("[") + tag + "] load").c_str(),  load_basis(basis, prog);)
        if (!prog.get<bool>("--notrim")) basis.remove_null_states(H_sym);
        std::cout << "[rank " << ctx.my_rank << "] " << tag
                  << " dim=" << basis.dim() << "\n";
        auto H = MPILazyOpSum(basis, H_sym, ctx);
        H.allocate_temporaries();
        std::vector<double> v(basis.dim()), u(basis.dim(), 0.0);
        std::mt19937 rng(seed);
        projED::set_random_unit_mpi(v, rng);
        TIMEIT((std::string("[") + tag + "] u += Av").c_str(),
               H.evaluate_add(v.data(), u.data());)
    };

    if (bt == "all" || bt == "bst")    { ZBasisBST_HashMPI     b; bench_one(b, "BST");    }
    if (bt == "all" || bt == "interp") { ZBasisInterp_HashMPI  b; bench_one(b, "interp"); }
    if (bt == "all" || bt == "fast")   { ZBasisBSTFast_HashMPI b; bench_one(b, "fast");   }

    MPI_Finalize();
    return 0;
}

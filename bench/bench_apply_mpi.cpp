#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "operator_mpi.hpp"
#include "common_bits_mpi.hpp"
#include <random>
#include "timeit.hpp"
#include <fstream>

// Returns resident set size in bytes (Linux /proc, falls back to 0).
static size_t rss_bytes() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            size_t kb = std::stoull(line.substr(6));
            return kb * 1024;
        }
    }
    return 0;
}

static void print_mem(const MPIHashContext& ctx, const char* label) {
    size_t rss = rss_bytes();
    size_t rss_max = 0;
    MPI_Reduce(&rss, &rss_max, 1, get_mpi_type<size_t>(), MPI_MAX, 0, MPI_COMM_WORLD);
    size_t rss_sum = 0;
    MPI_Reduce(&rss, &rss_sum, 1, get_mpi_type<size_t>(), MPI_SUM, 0, MPI_COMM_WORLD);
    if (ctx.my_rank == 0)
        std::cout << "[mem] " << label
                  << "  max=" << rss_max / (1<<20) << " MiB"
                  << "  total=" << rss_sum / (1<<20) << " MiB\n";
}


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
        print_mem(ctx, (std::string(tag) + " before load").c_str());
        TIMEIT((std::string("[") + tag + "] load raw").c_str(), load_basis_raw(basis, prog);)
        print_mem(ctx, (std::string(tag) + " after load raw").c_str());

        if (!prog.get<bool>("--notrim")) basis.remove_null_states(H_sym);
        print_mem(ctx, (std::string(tag) + " after trim").c_str());

        TIMEIT((std::string("[") + tag + "] redistribute").c_str(), basis.redistribute();)
        print_mem(ctx, (std::string(tag) + " after redistribute").c_str());

        // Per-rank breakdown
        size_t states_bytes = basis.dim() * sizeof(ZBasisBase::state_t);
        if (ctx.my_rank == 0)
            std::cout << "[" << tag << "] local dim=" << basis.dim()
                      << "  states=" << states_bytes / (1<<20) << " MiB";
        if constexpr (std::is_base_of_v<ZBasisInterp, std::decay_t<decltype(basis)>>) {
            size_t nb = basis.n_bounds_entries();
            if (ctx.my_rank == 0)
                std::cout << "  bounds_entries=" << nb
                          << " (~" << nb * 56 / (1<<20) << " MiB)";
        }
        if (ctx.my_rank == 0) std::cout << "\n";

        auto H = MPILazyOpSum(basis, H_sym, ctx);
        H.allocate_temporaries();

        std::vector<double> v(basis.dim()), u(basis.dim(), 0.0);
        std::mt19937 rng(seed);
        projED::set_random_unit_mpi(v, rng);
        print_mem(ctx, (std::string(tag) + " before apply (vecs allocated)").c_str());

        TIMEIT((std::string("[") + tag + "] u += Av").c_str(),
               H.evaluate_add(v.data(), u.data());)
        print_mem(ctx, (std::string(tag) + " after apply").c_str());
    };

    if (bt == "all" || bt == "bst")    { ZBasisBST_HashMPI     b; bench_one(b, "BST");    }
    if (bt == "all" || bt == "interp") { ZBasisInterp_HashMPI  b; bench_one(b, "interp"); }
    if (bt == "all" || bt == "fast")   { ZBasisBSTFast_HashMPI b; bench_one(b, "fast");   }

    MPI_Finalize();
    return 0;
}

#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "logging.hpp"
#include "operator_mpi.hpp"
#include "common_bits_mpi.hpp"
#include <random>
#include "timeit.hpp"
#include <fstream>
#include <omp.h>
#include <iomanip>
#include <cstdlib>
#include <algorithm>


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

    prog.add_argument("--interp-bits")
        .help("For interp basis: number of high bits of uint64[1] to use as bounds-map key (1-64, default: 64). "
              "Fewer bits = smaller map (~56 bytes * 2^N) but wider per-key search range.")
        .default_value(64)
        .scan<'i', int>();

    prog.add_argument("--repeats")
        .help("Number of timed apply repetitions. With N>1 the first repeat is "
              "treated as warm-up and excluded from the min/avg summary.")
        .default_value(1)
        .scan<'i', int>();

    prog.add_argument("--threads")
        .help("OpenMP threads per rank for the apply (0 = leave OMP_NUM_THREADS "
              "/ runtime default untouched).")
        .default_value(0)
        .scan<'i', int>();

    prog.add_argument("--verbosity")
        .help("Level of detail to print")
        .default_value(2)
        .scan<'i', int>();

    prog.add_argument("--all-rank-info")
        .help("Prints stats for all ranks (default: only rank 0)")
        .default_value(false)
        .implicit_value(true);

    prog.add_argument("--strategy")
        .help("Choice of apply kernel")
        .choices("prealloc", "pipe", "pipe_plain", "prealloc_p2p")
        .default_value("pipe");

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

    int interp_bits = prog.get<int>("--interp-bits");
    if (interp_bits < 1 || interp_bits > 64) {
        std::cerr << "--interp-bits must be between 1 and 64\n";
        return 1;
    }
    // Top-N-bit mask: 0xFFFF...FF00...00 with N high bits set
    uint64_t interp_hi_mask = (interp_bits >= 64) ? ~0ULL : (~0ULL << (64 - interp_bits));

    unsigned int seed = prog.get<unsigned int>("--seed");
    int repeats = std::max(1, prog.get<int>("--repeats"));

    int threads = prog.get<int>("--threads");
    if (threads > 0) omp_set_num_threads(threads);

    // Threads call MPI only between parallel regions, so FUNNELED suffices.
    int provided = 0;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Warning: MPI provides thread level " << provided
                  << " < FUNNELED; hybrid runs may be unsafe\n";
    }

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
    logging::configure(ctx.my_rank, prog.get<int>("--verbosity"), prog.get<bool>("--all-rank-info"));

	using T=double;
	SymbolicOpSum<T> H_sym;

    // These couplings build a purely off-diagonal Hamiltonian (ring exchange
    // only; no Ising/field diagonal term). So evaluate_add's diagonal pass is
    // trivial and the timed apply is entirely the searched off-diagonal path —
    // no need to isolate it, the whole measurement already is it.
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);


    MPILazyOpSumStrategy strat = parse_mpi_strategy(prog.get<std::string>("--strategy"));

    // Interleaved-search / scatter-prefetch knobs (consumed inside the apply via
    // the same env vars); recorded here so the sweep CSV captures what was used.
    auto env_int = [](const char* name, int def) {
        const char* e = std::getenv(name);
        return e ? std::atoi(e) : def;
    };
    const int search_group = std::min(32, std::max(1, env_int("APPLY_SEARCH_GROUP", 8)));
    const int scatter_pd   = std::max(0, env_int("APPLY_SCATTER_PD", 16));

    auto bench_one = [&](auto& basis, const char* tag) {
        if constexpr (std::is_base_of_v<ZBasisInterp, std::decay_t<decltype(basis)>>) {
            basis.set_hi_mask(interp_hi_mask);
            if (ctx.my_rank == 0)
                std::cout << "[" << tag << "] hi_mask=0x" << std::hex << interp_hi_mask
                          << std::dec << " (" << interp_bits << " bits, max "
                          << (1ULL << std::min(interp_bits, 20)) << (interp_bits > 20 ? "..." : "")
                          << " entries)\n";
        }
        print_mem(ctx, (std::string(tag) + " before load").c_str());
        TIMEIT((std::string("[") + tag + "] load raw").c_str(), load_basis_raw(basis, prog);)
        print_mem(ctx, (std::string(tag) + " after load raw").c_str());

        if (!prog.get<bool>("--notrim")) basis.remove_null_states(H_sym);
        print_mem(ctx, (std::string(tag) + " after trim").c_str());

        TIMEIT((std::string("[") + tag + "] redistribute").c_str(), basis.redistribute();)
        print_mem(ctx, (std::string(tag) + " after redistribute").c_str());

        // Per-rank breakdown. The searched working set (sorted states + any
        // acceleration structure) is what must exceed L3 for this benchmark to
        // exercise the DRAM-latency regime rather than a cache-resident toy.
        size_t states_bytes = basis.dim() * sizeof(ZBasisBase::state_t);
        size_t accel_bytes = 0;
        if constexpr (std::is_base_of_v<ZBasisInterp, std::decay_t<decltype(basis)>>)
            accel_bytes = basis.n_bounds_entries() * 56;
        size_t wset_local = states_bytes + accel_bytes;
        // Local dims differ across ranks; report the largest so the field
        // reflects the rank that actually has to reach furthest into memory.
        size_t wset_max = 0;
        MPI_Reduce(&wset_local, &wset_max, 1, get_mpi_type<size_t>(), MPI_MAX,
                   0, MPI_COMM_WORLD);
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

        auto H = MPILazyOpSum(basis, H_sym, ctx, strat);

        std::vector<double> v(basis.dim()), u(basis.dim(), 0.0);
        std::mt19937 rng(seed);
        projED::set_random_unit_mpi(v, rng);
        print_mem(ctx, (std::string(tag) + " before apply (vecs allocated)").c_str());

        // Per-repeat wall time = slowest rank (barrier-synchronised entry).
        double t_min = 0, t_sum = 0;
        int n_counted = 0;
        for (int rep = 0; rep < repeats; rep++) {
            std::fill(u.begin(), u.end(), 0.0);
            if (ctx.my_rank == 0)
                std::cout << "[" << tag << "] u += Av rep " << rep << ": ";
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            H.evaluate_add(v.data(), u.data());
            double dt = MPI_Wtime() - t0, dt_max = 0;
            MPI_Reduce(&dt, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            bool warmup = (repeats > 1 && rep == 0);
            if (ctx.my_rank == 0)
                std::cout << dt_max * 1e3 << " ms" << (warmup ? " (warm-up)" : "") << "\n";

            if (!warmup) {
                t_min = (n_counted == 0) ? dt_max : std::min(t_min, dt_max);
                t_sum += dt_max;
                n_counted++;
            }
        }
        if (ctx.my_rank == 0 && n_counted > 1)
            std::cout << "[" << tag << "] u += Av summary over " << n_counted
                      << " repeats: min=" << t_min * 1e3
                      << " ms  avg=" << t_sum / n_counted * 1e3 << " ms\n";

        // Global checksum of u so different strategies can be cross-checked:
        // sum and sum-of-squares over the whole distributed vector. Parallel
        // reductions reorder FP adds, so expect agreement to ~1e-10 relative,
        // not bit-identical.
        double loc_sum = 0.0, loc_sq = 0.0;
        for (ZBasisBase::idx_t i = 0; i < basis.dim(); ++i) {
            loc_sum += u[i];
            loc_sq  += u[i] * u[i];
        }
        double glob_sum = 0.0, glob_sq = 0.0;
        MPI_Reduce(&loc_sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&loc_sq,  &glob_sq,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Machine-readable one-liner for sweep.sh to grep into a CSV row.
        if (ctx.my_rank == 0 && n_counted >= 1) {
            std::cout << "[result]"
                      << " tag=" << tag
                      << " strategy=" << strat
                      << " ranks=" << ctx.world_size
                      << " threads=" << omp_get_max_threads()
                      << " search_group=" << search_group
                      << " scatter_pd=" << scatter_pd
                      << " wset_kib=" << wset_max / 1024
                      << " repeats=" << n_counted
                      << " min_ms=" << t_min * 1e3
                      << " avg_ms=" << t_sum / n_counted * 1e3
                      << " sum=" << std::setprecision(12) << glob_sum
                      << " sumsq=" << std::setprecision(12) << glob_sq
                      << "\n";
        }
        print_mem(ctx, (std::string(tag) + " after apply").c_str());
    };

    if (bt == "all" || bt == "bst")    { ZBasisBST_HashMPI     b; bench_one(b, "BST");    }
    if (bt == "all" || bt == "interp") { ZBasisInterp_HashMPI  b; bench_one(b, "interp"); }
    if (bt == "all" || bt == "fast")   { ZBasisBSTFast_HashMPI b; bench_one(b, "fast");   }

    MPI_Finalize();
    return 0;
}

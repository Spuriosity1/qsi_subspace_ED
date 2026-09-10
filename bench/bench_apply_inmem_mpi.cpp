// bench_apply_inmem_mpi
// =====================
// A variant of bench_apply_mpi that never touches disk. Instead of loading a
// pre-generated HDF5 basis, it enumerates the constrained basis straight into
// RAM using the MPI work-stealing tree search with an in-memory sink
// (MemoryShard), then times a single `u += A v` apply on the resulting
// distributed basis.
//
// Motivation: the largest targets (e.g. the 128-site 16,16,16,16 sector) have
// bases far too big to materialise on the shared filesystem, so bench_apply_mpi
// (which reads a basis file) cannot be pointed at them. This binary builds the
// basis entirely in memory so we can estimate the cost of one matvec step for
// exactly those problems.
//
// Bounding memory: a rank can *find* far more states than it will ultimately
// *own* (the search partitions by tree, the basis by hash), so instead of
// spilling to disk we rely on periodic mid-search hash-redistribution rounds
// (--redist-interval): every so often each rank ships its found states to their
// hash-owner, capping the per-rank high-water mark near its owned share.
//
// The apply, timing, memory reporting, and machine-readable [result] line are
// identical to bench_apply_mpi so the two can be compared directly.

#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "logging.hpp"
#include "operator_mpi.hpp"
#include "common_bits_mpi.hpp"

// in-memory basis-build path
#include "pyro_tree.hpp"
#include "pyro_tree_mpi.hpp"
#include "shard.hpp"          // MemoryShard
#include "local_memory_cli.hpp"
#include "admin.hpp"
#include "basis_io_h5.hpp"    // basis_io::make_sector_string

#include <random>
#include "timeit.hpp"
#include <fstream>
#include <filesystem>
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
using namespace projED;


// Enumerate this rank's slice of the constrained basis straight into RAM
// (MemoryShard). With redist_interval > 0 the search periodically hash-
// redistributes the accumulated states to their owning ranks, bounding the
// per-rank footprint. Returns the states this rank holds when the search ends
// (still one-copy-globally, but not yet hash-final: adopt_states+redistribute()
// completes that). raw_count_local is set to this rank's held count.
template <typename LatC>
static std::vector<Uint128> search_basis_inmem(
        const lattice& lat, int num_spinon_pairs,
        const std::vector<size_t>& perm,
        const std::string& job_tag,
        size_t mem_block,
        const argparse::ArgumentParser& prog,
        const std::vector<int>& sector,
        size_t& raw_count_local)
{
    // workdir is only touched to name a checkpoint file if the search is
    // interrupted (SIGINT); nothing is written on a normal run.
    std::filesystem::path workdir = std::filesystem::temp_directory_path();

    mpi_par_searcher<LatC, MemoryShard> L(lat, num_spinon_pairs, perm,
            workdir, job_tag, mem_block);
    if constexpr (std::is_same_v<LatC, lat_container_with_sector>) {
        L.set_sector(sector);
    }
    L.set_iter_opts(prog.get<int>("--check_interval"),
                    prog.get<int>("--print_interval"),
                    prog.get<int>("--chunk_size"));
    L.set_redist_interval(prog.get<double>("--redist-interval"));
    L.set_local_memory_limit(
            resolve_local_memory_bytes(prog.get<double>("--local-memory")));
    L.build_state_tree();

    std::vector<Uint128> states = L.sink().take_states();
    raw_count_local = states.size();
    return states;
}


int main(int argc, char* argv[]){

    argparse::ArgumentParser prog(argv[0]);
    prog.add_argument("lattice_file");
    prog.add_argument("n_spinon_pairs")
        .help("number of spinon pairs (default 0 = ground constraint sector)")
        .default_value(0)
        .scan<'i', int>();

    // ---- basis-build options ----------------------------------------------
    prog.add_argument("-r", "--sector")
        .help("target global polarisation sector (empty = full constrained basis)")
        .nargs(argparse::nargs_pattern::any)
        .default_value(std::vector<int>{})
        .scan<'d', int>();
    prog.add_argument("--order_spins")
        .help("index ordering used to prune the search tree")
        .choices("none", "greedy", "random")
        .default_value(std::string{"greedy"});
    prog.add_argument("--check_interval")
        .help("iterations before checking for dry ranks")
        .default_value(10000)
        .scan<'i', int>();
    prog.add_argument("--print_interval")
        .help("check cycles before printing search debug info")
        .default_value(5000)
        .scan<'i', int>();
    prog.add_argument("--chunk_size")
        .help("minimum stack size to permit sending to another rank")
        .default_value(2)
        .scan<'i', int>();
    prog.add_argument("--mem-block-size")
        .help("MemoryShard block size in states: the search accumulates states "
              "in blocks of this size, and each periodic redistribution round "
              "processes one block at a time (larger = fewer, bigger comm rounds)")
        .default_value(1<<20)
        .scan<'i', int>();
    prog.add_argument("--redist-interval")
        .help("wall-clock seconds between in-search hash-redistribution rounds "
              "(0 = only redistribute once, at the end). >0 caps per-rank RAM "
              "during a large search.")
        .default_value(30.0)
        .scan<'g', double>();
    prog.add_argument("--local-memory")
        .help("soft cap on this rank's in-RAM basis shard, in GiB; on reaching "
              "it the rank pauses enumeration until the next --redist-interval "
              "round drains the shard. 0 = unlimited; <0 (default) = auto "
              "(0.5 x SLURM_MEM_PER_CPU x SLURM_CPUS_PER_TASK)")
        .default_value(-1.0)
        .scan<'g', double>();

    // ---- apply/benchmark options (mirror bench_apply_mpi) ------------------
    prog.add_argument("--seed")
        .help("Seed for the RNG")
        .scan<'i', unsigned int>()
        .default_value(0u);
    prog.add_argument("--notrim")
        .help("keep states annihilated by every term of H")
        .default_value(false)
        .implicit_value(true);
    prog.add_argument("--basis-type")
        .help("Basis search structure: bst | interp | fast  (default: run all three)")
        .default_value(std::string("all"));
    prog.add_argument("--interp-bits")
        .help("For interp basis: high bits of uint64[1] used as bounds-map key (1-64, default 64).")
        .default_value(64)
        .scan<'i', int>();
    prog.add_argument("--repeats")
        .help("Number of timed apply repetitions. With N>1 the first repeat is "
              "treated as warm-up and excluded from the min/avg summary.")
        .default_value(1)
        .scan<'i', int>();
    prog.add_argument("--threads")
        .help("OpenMP threads per rank for the apply (0 = leave runtime default).")
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

    // Purely off-diagonal Hamiltonian (ring exchange only; no Ising/field
    // diagonal). The diagonal pass is trivial, so the timed apply is entirely
    // the searched off-diagonal path -- same choice as bench_apply_mpi.
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    MPILazyOpSumStrategy strat = parse_mpi_strategy(prog.get<std::string>("--strategy"));

    auto env_int = [](const char* name, int def) {
        const char* e = std::getenv(name);
        return e ? std::atoi(e) : def;
    };
    const int search_group = std::min(32, std::max(1, env_int("APPLY_SEARCH_GROUP", 8)));
    const int scatter_pd   = std::max(0, env_int("APPLY_SCATTER_PD", 16));

    // ------------------------------------------------------------------------
    // Step 2: enumerate the basis directly into RAM (no disk).
    // The lattice is permuted only to prune the search tree; emitted states are
    // unpermuted, so H_sym (built from the un-permuted JSON above) stays valid.
    // ------------------------------------------------------------------------
    lattice lat(jdata);
    auto choice = prog.get<std::string>("--order_spins");
    std::vector<size_t> perm = get_permutation(choice, lat);
    MPI_Bcast(perm.data(), perm.size(), get_mpi_type<size_t>(), 0, MPI_COMM_WORLD);
    lat.apply_permutation(perm);

    int num_spinon_pairs = prog.get<int>("n_spinon_pairs");
    auto target_sector = prog.get<std::vector<int>>("--sector");
    size_t mem_block = static_cast<size_t>(prog.get<int>("--mem-block-size"));
    std::string job_tag = "benchbuild-" +
        std::filesystem::path(lattice_file).stem().string();

    if (ctx.my_rank == 0)
        std::cout << "[build] enumerating basis in-memory (sector="
                  << (target_sector.empty() ? std::string("full")
                        : basis_io::make_sector_string(target_sector))
                  << ", redist-interval=" << prog.get<double>("--redist-interval")
                  << "s) ...\n";

    print_mem(ctx, "before search");
    size_t raw_local = 0;
    double t_build0 = MPI_Wtime();
    std::vector<Uint128> found = target_sector.empty()
        ? search_basis_inmem<lat_container>(lat, num_spinon_pairs, perm,
                job_tag, mem_block, prog, target_sector, raw_local)
        : search_basis_inmem<lat_container_with_sector>(lat, num_spinon_pairs, perm,
                job_tag, mem_block, prog, target_sector, raw_local);
    double t_build = MPI_Wtime() - t_build0, t_build_max = 0;
    MPI_Reduce(&t_build, &t_build_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int interrupted = GLOBAL_SHUTDOWN_REQUEST ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &interrupted, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (interrupted) {
        if (ctx.my_rank == 0)
            std::cerr << "[build] search was interrupted; aborting.\n";
        MPI_Finalize();
        return 1;
    }

    size_t raw_global = 0;
    MPI_Allreduce(&raw_local, &raw_global, 1, get_mpi_type<size_t>(),
            MPI_SUM, MPI_COMM_WORLD);
    if (ctx.my_rank == 0)
        std::cout << "[build] search done in " << t_build_max << " s; raw basis dim="
                  << raw_global << "\n";
    print_mem(ctx, "after search (states in RAM)");

    // bench_one: adopt the given state slab into the basis structure, trim,
    // redistribute to hash-owners, then time the apply. Structured identically
    // to bench_apply_mpi::bench_one, with the file-load prelude replaced by the
    // in-memory adopt_states().
    auto bench_one = [&](auto& basis, const char* tag, std::vector<Uint128>&& slab) {
        if constexpr (std::is_base_of_v<ZBasisInterp, std::decay_t<decltype(basis)>>) {
            basis.set_hi_mask(interp_hi_mask);
            if (ctx.my_rank == 0)
                std::cout << "[" << tag << "] hi_mask=0x" << std::hex << interp_hi_mask
                          << std::dec << " (" << interp_bits << " bits, max "
                          << (1ULL << std::min(interp_bits, 20)) << (interp_bits > 20 ? "..." : "")
                          << " entries)\n";
        }
        print_mem(ctx, (std::string(tag) + " before adopt").c_str());
        basis.adopt_states(std::move(slab));
        if (!prog.get<bool>("--notrim")) basis.remove_null_states(H_sym);
        print_mem(ctx, (std::string(tag) + " after trim").c_str());

        TIMEIT((std::string("[") + tag + "] redistribute").c_str(), basis.redistribute();)
        print_mem(ctx, (std::string(tag) + " after redistribute").c_str());

        if (ctx.my_rank == 0)
            std::cout << "[" << tag << "] global basis dim=" << basis.global_dim()
                      << " (trimmed " << raw_global - basis.global_dim() << ")\n";

        // Per-rank working-set report. The searched working set (sorted states +
        // any acceleration structure) is what must exceed L3 for this benchmark
        // to exercise the DRAM-latency regime rather than a cache-resident toy.
        size_t states_bytes = basis.dim() * sizeof(ZBasisBase::state_t);
        size_t accel_bytes = 0;
        if constexpr (std::is_base_of_v<ZBasisInterp, std::decay_t<decltype(basis)>>)
            accel_bytes = basis.n_bounds_entries() * 56;
        size_t wset_local = states_bytes + accel_bytes;
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

        // Global checksum of u so different strategies can be cross-checked.
        double loc_sum = 0.0, loc_sq = 0.0;
        for (ZBasisBase::idx_t i = 0; i < basis.dim(); ++i) {
            loc_sum += u[i];
            loc_sq  += u[i] * u[i];
        }
        double glob_sum = 0.0, glob_sq = 0.0;
        MPI_Reduce(&loc_sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&loc_sq,  &glob_sq,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (ctx.my_rank == 0 && n_counted >= 1) {
            std::cout << "[result]"
                      << " tag=" << tag
                      << " strategy=" << strat
                      << " ranks=" << ctx.world_size
                      << " threads=" << omp_get_max_threads()
                      << " search_group=" << search_group
                      << " scatter_pd=" << scatter_pd
                      << " global_dim=" << basis.global_dim()
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

    // Requested basis structures. All but the last adopt a copy of the found
    // states; the last moves them (so the common single-type run never doubles
    // the state footprint). The redistribute() inside bench_one frees the slab.
    std::vector<std::string> types;
    if (bt == "all" || bt == "bst")    types.push_back("bst");
    if (bt == "all" || bt == "interp") types.push_back("interp");
    if (bt == "all" || bt == "fast")   types.push_back("fast");

    for (size_t ti = 0; ti < types.size(); ++ti) {
        const bool last = (ti + 1 == types.size());
        std::vector<Uint128> slab = last ? std::move(found)
                                          : std::vector<Uint128>(found);
        if (types[ti] == "bst")    { ZBasisBST_HashMPI     b; bench_one(b, "BST",    std::move(slab)); }
        if (types[ti] == "interp") { ZBasisInterp_HashMPI  b; bench_one(b, "interp", std::move(slab)); }
        if (types[ti] == "fast")   { ZBasisBSTFast_HashMPI b; bench_one(b, "fast",   std::move(slab)); }
    }

    MPI_Finalize();
    return 0;
}

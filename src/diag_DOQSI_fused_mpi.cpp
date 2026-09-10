#include <argparse/argparse.hpp>

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <mpi.h>

#include "pyro_tree.hpp"
#include "pyro_tree_mpi.hpp"
#include "shard.hpp"        // MemoryShard
#include "admin.hpp"
#include "physics/geometry.hpp"

#include "hamiltonian_setup.hpp"
#include "local_memory_cli.hpp"
#include "operator_mpi.hpp"
#include "lanczos_mpi.hpp"
#include "lanczos_cli.hpp"
#include "logging_cli.hpp"
#include "expectation_eval.hpp"

using json = nlohmann::json;
using namespace projED;
using basis_t = ZBasisBSTFast_HashMPI;

// Fused pipeline: gen_spinon_basis/sbsearch_mpi + merge_shards +
// diag_DOQSI_ham_mpi in one process, without the basis ever landing on the
// shared filesystem. Per rank:
//   i)   enumerate the constrained basis (optionally within a fixed
//        polarisation --sector) via the MPI work-stealing tree search,
//        streaming the found states to a binary shard on a rank-local scratch
//        disk (--scratch_dir, else $TMPDIR/tmp). A large search can find many
//        more states on a rank than that rank will ultimately own, so spilling
//        to local disk keeps the search-phase RAM bounded;
//   ii)  stream the shard back in blocks, dropping states annihilated by every
//        term of H (--notrim to skip), redistribute each block to its
//        hash-correct rank, and precompute the static apply plan (remote target
//        indices; --noplan to skip);
//   iii) run the checkpointed Lanczos recurrence and write eigenpairs to HDF5.
//
// The tree-search stack checkpoint is NOT usable across runs here: states
// found before an interrupt lived only in RAM, so a resumed search would
// silently yield an incomplete basis. Stale checkpoints are deleted and an
// interrupted search aborts the run.

// Enumerate this rank's slice of the constrained basis straight into RAM
// (MemoryShard) -- nothing touches disk. On a very large search a rank can
// *find* far more states than it will ultimately *own* after the hash
// redistribution; to keep that from exhausting the node, the search runs
// periodic in-RAM hash-redistribution rounds (--redist-interval), each shipping
// found states to their owning rank and capping the per-rank footprint near its
// owned share. Returns the states this rank holds when the search ends (one copy
// globally, but not yet hash-final -- adopt_states()+redistribute() completes
// that). raw_count_local is set to this rank's held count.
template <typename LatC>
static std::vector<Uint128> search_basis_inmem(
        const lattice& lat, int num_spinon_pairs,
        const std::vector<size_t>& perm,
        const std::filesystem::path& workdir,
        const std::string& job_tag,
        const argparse::ArgumentParser& prog,
        const std::vector<int>& sector,
        size_t& raw_count_local)
{

    // Discard any stack checkpoint left by an interrupted fused run (see above)
    std::filesystem::remove(workdir /
            ("checkpoint-" + job_tag + "-" + std::to_string(get_mpi_rank()) + ".bin"));

    mpi_par_searcher<LatC, MemoryShard> L(lat, num_spinon_pairs, perm,
            workdir, job_tag, (1u << 20));
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


int main(int argc, char* argv[]) {
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file")
		.help("The json-valued lattice spec");
	prog.add_argument("n_spinon_pairs")
		.default_value(0)
		.scan<'i', int>();
    prog.add_argument("--sector", "-r")
        .help("target global polarisation sector")
        .nargs(argparse::nargs_pattern::any)
        .default_value(std::vector<int>{})
        .scan<'d', int>();
	prog.add_argument("--order_spins")
		.choices("none", "greedy", "random")
		.default_value(std::string{"greedy"});

    // basis search tuning
    prog.add_argument("--check_interval")
        .help("Iterations before checking for dry ranks")
        .default_value(10000)
        .scan<'i', int>();
    prog.add_argument("--print_interval")
        .help("number of check cycles before printing debug info")
        .default_value(5000)
        .scan<'i', int>();
    prog.add_argument("--chunk_size")
        .help("minimum size of a stack to permit sending to another rank")
        .default_value(2)
        .scan<'i', int>();

    // DEPRECATED / no-op. The basis is now enumerated entirely in RAM, so these
    // disk-shard knobs do nothing; they are still accepted (and a warning is
    // printed if passed) so existing job scripts keep working unchanged.
    prog.add_argument("--scratch_dir")
        .help("[deprecated, ignored] rank-local scratch directory for basis "
              "search shards -- the search is now fully in-memory")
        .default_value(std::string(""));
    prog.add_argument("--ingest-block-size")
        .help("[deprecated, ignored] states per redistribution round when "
              "streaming search shards back from disk -- no longer used")
        .default_value(1<<20)
        .scan<'i', int>();

    // Periodic in-search redistribution. A rank can *find* far more states than
    // it will ultimately *own*, so the in-RAM found set can overrun the node's
    // memory before the search finishes. Every this-many seconds of wall-clock
    // time all ranks synchronise and hash-redistribute their found states to the
    // owning ranks (in RAM), bounding each rank's footprint to its share. This
    // is the safety valve that replaces the old disk spill, so it defaults ON.
    // 0 keeps only the one-shot redistribute-at-the-end behaviour.
    prog.add_argument("--redist-interval")
        .help("wall-clock seconds between periodic in-search hash-redistribution "
              "rounds (0 = only redistribute once, at the end)")
        .default_value(60.0)
        .scan<'g', double>();

    // Soft per-rank cap on the in-RAM basis shard. If a rank's found set spikes
    // to this size between redistribution rounds it stops enumerating new states
    // until the next round drains it, bounding the per-rank high-water mark
    // (guards against an OOM/UCX-registration failure from a shard overrunning
    // the node before a redistribution can rebalance it). Needs
    // --redist-interval > 0 to have any effect.
    prog.add_argument("--local-memory")
        .help("soft cap on this rank's in-RAM basis shard, in GiB; on reaching "
              "it the rank pauses enumeration until the next --redist-interval "
              "round drains the shard. 0 = unlimited; <0 (default) = auto "
              "(0.5 x SLURM_MEM_PER_CPU x SLURM_CPUS_PER_TASK)")
        .default_value(-1.0)
        .scan<'g', double>();


    // G specification
    {
        auto &group = prog.add_mutually_exclusive_group(true);
        group.add_argument("--B")
            .help("magnetic field, units of Jzz")
            .nargs(3)
            .scan<'g', double>();

        group.add_argument("--g")
            .help("raw ring exchange, units of Jzz")
            .nargs(4)
            .scan<'g', double>();

        prog.add_argument("--Jpm")
            .help("Jpm, units of Jzz")
            .scan<'g', double>();
    }

    prog.add_argument("--notrim")
        .help("Keep states annihilated by every term of H")
        .default_value(false)
        .implicit_value(true);

    prog.add_argument("-o", "--output_dir")
        .required()
        .help("output directory ");

    provide_lanczos_options(prog);
    provide_logging_options(prog);

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
		std::cerr << err.what() << std::endl;
		std::cerr << prog;
        std::exit(1);
    }

    MPI_Init(&argc, &argv);

	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;

    MPIctx ctx;
    configure_logging(prog, ctx.my_rank);

    // The basis is now built entirely in RAM; the old disk-shard knobs are
    // accepted for script compatibility but do nothing. Warn if either is set.
    if (ctx.my_rank == 0) {
        if (prog.is_used("--scratch_dir"))
            logging::log(logging::INFO) << "[Main] WARNING: --scratch_dir is deprecated "
                "and ignored: the basis search is now fully in-memory.\n";
        if (prog.is_used("--ingest-block-size"))
            logging::log(logging::INFO) << "[Main] WARNING: --ingest-block-size is "
                "deprecated and ignored: states are no longer streamed from disk.\n";
    }

	using coeff_t=double;
    bool calc_partial_vol = true;

	SymbolicOpSum<coeff_t> H_sym;

    char outfilename_buf[1024];

    std::stringstream s;

    if (prog.is_used("--g")){
        auto gv = prog.get<std::vector<double>>("--g");
        build_hamiltonian(H_sym, jdata, gv);

        snprintf(outfilename_buf, 1024, "g0=%.4f%%g1=%.4f%%g2=%.4f%%g3=%.4f%%",
                gv[0], gv[1], gv[2],gv[3]);
    } else {
        auto Jpm = prog.get<double>("--Jpm");
        auto Bv = prog.get<std::vector<double>>("B");

        Eigen::Vector3d B;
        for (size_t i=0; i<3; i++)
            B[i] = Bv[i];

        snprintf(outfilename_buf, 1024, "Jpm=%.4f%%Bx=%.4f%%By=%.4f%%Bz=%.4f%%",
                Jpm, B[0], B[1], B[2]);

        build_hamiltonian(H_sym, jdata, Jpm, B, ctx.my_rank != 0);
    }

    // make the out dir if not exists
    std::filesystem::create_directories(prog.get<std::string>("--output_dir"));

    s << prog.get<std::string>("--output_dir") << "/" << outfilename_buf;

	// Step 2: enumerate the basis directly into RAM
	lattice lat(jdata);

	auto choice = prog.get<std::string>("--order_spins");

	// Permute the indices to make early tree truncation as efficient as
	// possible; states are unpermuted as they are emitted. Every rank must
	// search with the same ordering, so rank 0's choice is broadcast.
	std::vector<size_t> perm = get_permutation(choice, lat);
	MPI_Bcast(perm.data(), perm.size(), get_mpi_type<size_t>(), 0, MPI_COMM_WORLD);
	lat.apply_permutation(perm);

    auto num_spinon_pairs = prog.get<int>("n_spinon_pairs");
    auto target_sector = prog.get<std::vector<int>>("--sector");
    std::filesystem::path workdir(prog.get<std::string>("--output_dir"));
    std::string job_tag = "fused-" +
        std::filesystem::path(lattice_file).stem().string();

    if (ctx.my_rank == 0)
        logging::log(logging::INFO) << "[Search] Building basis in-memory "
            "(redist-interval=" << prog.get<double>("--redist-interval") << "s) ...\n";

    size_t raw_local = 0;
    std::vector<Uint128> found = target_sector.empty()
        ? search_basis_inmem<lat_container>(lat, num_spinon_pairs,
                perm, workdir, job_tag, prog, target_sector, raw_local)
        : search_basis_inmem<lat_container_with_sector>(lat,
                num_spinon_pairs, perm, workdir, job_tag, prog,
                target_sector, raw_local);

    // An interrupted search means an incomplete basis: abort, do not diagonalise.
    int interrupted = GLOBAL_SHUTDOWN_REQUEST ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &interrupted, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (interrupted) {
        if (ctx.my_rank == 0) {
            logging::log(logging::INFO) << "[Main] Basis search was interrupted; the in-memory "
                "pipeline cannot resume a partial search. Exiting.\n";
        }
        MPI_Finalize();
        return 1;
    }

    size_t raw_global = 0;
    MPI_Allreduce(&raw_local, &raw_global, 1, get_mpi_type<size_t>(),
            MPI_SUM, MPI_COMM_WORLD);
    if (ctx.my_rank == 0) {
        logging::log(logging::INFO) << "[Search] Done! raw basis dim=" << raw_global << "\n";
    }

	// Step 3: adopt the in-RAM found states, trim those annihilated by every
	// term of H (--notrim to skip), then hash-redistribute to the owning ranks.
	// Periodic in-search redistribution has already kept each rank's found set
	// near its owned share, so this final pass is on owned-size data.
    basis_t basis;
    basis.adopt_states(std::move(found));
    if (!prog.get<bool>("--notrim")) {
        basis.remove_null_states(H_sym);
    }
    basis.redistribute();
    logging::log(logging::DEBUG)<<"[MPI_BST]  Done! local basis dim="<<basis.dim()<<std::endl;
    if (ctx.my_rank == 0) {
        logging::log(logging::INFO) << "[MPI_BST] global basis dim=" << basis.global_dim()
                  << " (trimmed " << raw_global - basis.global_dim() << ")\n";
    }

    // Estimate the steady-state RAM of the diagonalisation before committing to
    // it, summed over ranks. Three contributions dominate:
    //   (i)   the Uint128 basis states themselves;
    //   (ii)  the three length-dim double vectors the Lanczos eigenvector pass
    //         keeps live simultaneously (current v = local_v0, the scratch/
    //         previous vector u inside the iterator, and the accumulated
    //         eigenvector); the eigenvalue-only pass peaks at two of these;
    //   (iii) the operator apply plan's index cache, which stores one uint32
    //         local target index per off-diagonal nonzero (~4 B each). We count
    //         the local nonvanishing off-diagonal records exactly here (one
    //         applyState pass, cheap next to Lanczos); this is an upper bound on
    //         the plan, since records whose target is not in the basis are
    //         dropped when the plan is built.
    {
        const size_t local_dim = basis.dim();
        size_t local_offdiag_nnz = 0;
        for (size_t i = 0; i < local_dim; ++i) {
            for (const auto& [c, op] : H_sym.off_diag_terms) {
                (void)c;
                Uint128 s = basis[i];
                if (op.applyState(s) != 0) ++local_offdiag_nnz;
            }
        }

        size_t local[3] = {
            local_dim * sizeof(Uint128),          // (i)   basis states
            3 * local_dim * sizeof(double),       // (ii)  Lanczos vectors
            local_offdiag_nnz * sizeof(uint32_t), // (iii) operator index cache
        };
        size_t g[3] = {0, 0, 0};
        MPI_Allreduce(local, g, 3, get_mpi_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);

        if (ctx.my_rank == 0) {
            const double MiB = 1.0 / (1 << 20);
            logging::log(logging::INFO) << "[Search] Estimated steady-state memory (global):\n"
                      << "    basis (Uint128)        : " << g[0] * MiB << " MiB\n"
                      << "    Lanczos vectors (3x)   : " << g[1] * MiB << " MiB\n"
                      << "    total persistent       : "
                      << (g[0] + g[1] ) * MiB << " MiB\n"
                      << "    operator indices       : " << g[2] * MiB << " MiB (~4 B x off-diag nonzeros, not cached)\n" ;
        }
    }

    ////////////////////////////////////////
    // Do the diagonalisation
    lanczos_mpi::Settings settings(ctx);
    parse_lanczos_settings(prog, settings);
    settings.verbosity = prog.get<int>("--verbosity");
    settings.calc_eigenvector = true;

    double eigval;
    std::vector<double> local_v0(basis.dim());
    std::vector<double> evector;
    std::vector<double> alphas, betas;

    // Scope the Hamiltonian so its pipelined comm buffers are freed before the
    // observable phase.
    {
        auto H = MPILazyOpSum(basis, H_sym, ctx, MPILazyOpSumStrategy::PIPE);

        RealApplyFn evadd = [&H](const coeff_t* x_local, coeff_t* y_local){
            H.evaluate_add(x_local, y_local);
        };

        logging::log(logging::INFO) << "[Lanczos] finding lowest eigenvalue\n";
        auto res=  lanczos_mpi::lanczos_iterate(evadd, local_v0, alphas, betas, settings);

        logging::log(logging::DEBUG)<<"[rank "<<ctx.my_rank<<"] "<<res;

        // If we hit checkpoint and exited early, clean exit
        if (!res.eigval_converged) {
            if (ctx.my_rank == 0) {
                logging::log(logging::INFO) << "[Main] Exited initial iteration at n="<<
                    res.n_iterations<<" due to time limit. Restart to continue.\n";
            }
            MPI_Finalize();
            return 0; // Clean exit for restart
        }

        // If converged, compute final eigenvalue and eigenvector
        logging::log(logging::INFO) << "[Lanczos] tridiagonalising in Krylov space\n";
        std::vector<double> ritz;
        {
            std::vector tmp_alphas(alphas);
            std::vector tmp_betas(betas);
            tridiagonalise_one(tmp_alphas, tmp_betas, eigval, ritz);
        }

        logging::log(logging::INFO) << "[Lanczos] iterating to determine eigenvector\n";
        evector.resize(basis.dim());

        // Second pass for eigenvector
        res = lanczos_mpi::lanczos_iterate(
            evadd, local_v0, alphas, betas, settings,
            &ritz, &evector
        );
        // The ground state is accumulated into evector; local_v0 holds the
        // (now unneeded) final Lanczos vector and is recycled as a scratch
        // buffer in the observable phase below.

        // If we hit checkpoint and exited early (eigenvector not fully
        // reconstructed), clean exit
        if (!res.eigvec_converged) {
            if (ctx.my_rank == 0) {
            logging::log(logging::INFO) << "[Main] Exited second iteration at n="<<
                res.n_iterations<<" due to time limit. Restart to continue.\n";
            }
            MPI_Finalize();
            return 0; // Clean exit for restart
        }
    } // H freed here


    std::string out_filename = std::filesystem::path(s.str()+".eigs.h5")
        .replace_extension(".out.h5");

    if (ctx.my_rank == 0){
        logging::log(logging::INFO) << "Eigenvalues:\n" << eigval << "\n\n";
        logging::log(logging::INFO) << "Writing to\n"<<out_filename<<std::endl;
    }

    // dummy (here in case we calc more later
    std::vector<double> eigvals{eigval};

    // Step 4: evaluate the ring observables (matrix free) in the ground
    // state |psi> = evector
    auto [ringL, ringR, sl_list]  = get_ring_ops(jdata);

    int n_operators = ringL.size();
    std::vector<double> expect_O(n_operators); // < O >
    std::vector<double> expect_O_O(n_operators); // < O_0' O_j >
    std::vector<double> expect_F(n_operators); // < O'_j O_j >
    std::array<std::vector<double>, 4> partial_vol; // < O O O > around missing plaq

    {
        if (ctx.my_rank == 0)
            logging::log(logging::INFO)<<"Compute <O> and <OO>... "<<std::flush;

        std::vector<double> chi(basis.dim()); // |chi> = O_j |psi>
        std::vector<double>& u = local_v0;    // recycled: |u> = O_0 |psi>

        for (int opi=0; opi<n_operators; opi++){
            if (ctx.my_rank == 0) logging::log(logging::INFO)<<opi<<" " <<std::flush;
            // Constructed per iteration so each operator's comm buffers are
            // freed again before the next one.
            MPILazyOpSum<double, basis_t> op(basis, ringL[opi], ctx);
            op.evaluate(evector.data(), chi.data());
            if (opi == 0){
                u = chi; // copies into the existing buffer, no new allocation
            }
            double res_local = projED::inner(chi, evector);
            double res;
            MPI_Reduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            // res now contains full total <O psi | psi>
            if (ctx.my_rank == 0){
                expect_O[opi] = res;
            }

            // < psi | O_0' O_j | psi > == <u | chi >
            res_local = projED::inner(u, chi);
            MPI_Reduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            // res now contains full total <O_0 psi | O_j psi>
            if (ctx.my_rank == 0){
                expect_O_O[opi] = res;
            }

            // Flippability === <chi | chi> == <psi | O_j' O_j |psi>
            res_local = projED::inner(chi, chi);
            MPI_Reduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (ctx.my_rank == 0){
                expect_F[opi] = res;
            }
        }

        // partial volume operators
        if (calc_partial_vol){
            if (ctx.my_rank == 0) logging::log(logging::INFO)<<"Compute <OOO>... "<<std::flush;
            // computing expectation values of the incomplete volumes
            for (int sl=0; sl<4; sl++){
                auto par_vol_operators = get_partial_vol_ops(jdata, ringL, sl);
                partial_vol[sl].resize(par_vol_operators.size(), 0.0);
                for (size_t opi=0; opi<par_vol_operators.size(); opi++){
                    if (ctx.my_rank == 0) logging::log(logging::INFO)<<opi<<" "<<std::flush;
                    // Per-iteration construction, as in the ring loop.
                    MPILazyOpSum<double, basis_t> op(basis, par_vol_operators[opi], ctx);
                    op.evaluate(evector.data(), chi.data());
                    double res = 0;
                    double res_local = projED::inner(chi, evector);
                    MPI_Reduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    // res contains the goods (root only)
                    if (ctx.my_rank == 0){
                        partial_vol[sl][opi] = res;
                    }
                }
            }

            if (ctx.my_rank == 0)
                logging::log(logging::INFO)<<"\nCompute <OOO> complete "<<std::endl;
        }


    }



    if (ctx.my_rank == 0){
        hid_t out_fid =
            H5Fcreate(out_filename.c_str(),
                    H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (out_fid < 0)
            throw std::runtime_error("Failed to create HDF5 file");
        logging::log(logging::INFO) << "Done!" << std::endl;

        // write out the data
        write_string_to_hdf5(out_fid, "latfile_json", lattice_file);

        std::string sector_str = target_sector.empty() ? 
            "basis" : basis_io::make_sector_string(target_sector);

        write_string_to_hdf5(out_fid, "sector", sector_str);

        write_expectation_vals_h5(out_fid, "ring", expect_O, ringL.size(), 1);
        write_expectation_vals_h5(out_fid, "flippability", expect_F, ringL.size(), 1);
        write_expectation_vals_h5(out_fid, "ring_2", expect_O_O, ringL.size(), 1);

        hsize_t dims[1]={1};
        write_dataset(out_fid, "eigenvalues", eigvals.data(), dims, 1);


        if (calc_partial_vol){
            // save the incomplete vol operators (each sl)
            for (size_t sl = 0; sl < partial_vol.size(); ++sl) {
                const auto& vec = partial_vol[sl];
                std::string name = "partial_vol_sl" + std::to_string(sl);
                write_expectation_vals_h5(out_fid, name.c_str(), vec, vec.size(), 1);
            }
        }

        H5Fclose(out_fid);
    }


    MPI_Finalize();

	return 0;
}

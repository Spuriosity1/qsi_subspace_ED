#include <argparse/argparse.hpp>

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <mpi.h>

#include "pyro_tree.hpp"
#include "pyro_tree_mpi.hpp"
#include "admin.hpp"
#include "physics/geometry.hpp"

#include "hamiltonian_setup.hpp"
#include "operator_mpi.hpp"
#include "lanczos_mpi.hpp"
#include "lanczos_cli.hpp"
#include "expectation_eval.hpp"

using json = nlohmann::json;
using namespace projED;
using basis_t = ZBasisBSTFast_HashMPI;

// Fused pipeline: gen_spinon_basis/sbsearch_mpi + merge_shards +
// diag_DOQSI_ham_mpi in one process, with the basis held in RAM throughout
// (nothing basis-sized ever touches the filesystem). Per rank:
//   i)   enumerate the constrained basis (optionally within a fixed
//        polarisation --sector) via the MPI work-stealing tree search,
//        accumulating states in memory instead of disk shards;
//   ii)  drop states annihilated by every term of H (--notrim to skip),
//        redistribute states to their hash-correct ranks, and precompute the
//        static apply plan (remote target indices; --noplan to skip);
//   iii) run the checkpointed Lanczos recurrence and write eigenpairs to HDF5.
//
// The tree-search stack checkpoint is NOT usable across runs here: states
// found before an interrupt lived only in RAM, so a resumed search would
// silently yield an incomplete basis. Stale checkpoints are deleted and an
// interrupted search aborts the run.

template <typename LatC>
static std::vector<Uint128> build_basis_states(
        const lattice& lat, int num_spinon_pairs,
        const std::vector<size_t>& perm,
        const std::filesystem::path& workdir,
        const std::string& job_tag,
        const argparse::ArgumentParser& prog,
        const std::vector<int>& sector,
        const SymbolicOpSum<double>& H_sym,
        size_t& raw_count_local)
{

    // Discard any stack checkpoint left by an interrupted fused run (see above)
    std::filesystem::remove(workdir /
            ("checkpoint-" + job_tag + "-" + std::to_string(get_mpi_rank()) + ".bin"));

    mpi_par_searcher<LatC, MemoryShard> L(lat, num_spinon_pairs, perm,
            workdir, job_tag);
    if constexpr (std::is_same_v<LatC, lat_container_with_sector>) {
        L.set_sector(sector);
    }
    L.set_iter_opts(prog.get<int>("--check_interval"),
                    prog.get<int>("--print_interval"),
                    prog.get<int>("--chunk_size"));
    L.build_state_tree();
    raw_count_local = L.sink().size();

    // Trim unflippable states block-by-block *before* assembling the slab
    // into one contiguous vector: the high-water mark stays at
    // (trimmed + one block) instead of (raw + trimmed).
    if (!GLOBAL_SHUTDOWN_REQUEST && !prog.get<bool>("--notrim")) {
        for (auto& blk : L.sink().mutable_blocks()) {
            remove_annihilated_states(H_sym, blk);
            blk.shrink_to_fit();
        }
    }
    return L.sink().take_states();
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
        .default_value(50)
        .scan<'i', int>();
    prog.add_argument("--chunk_size")
        .help("minimum size of a stack to permit sending to another rank")
        .default_value(2)
        .scan<'i', int>();

    // Checkpointing (Lanczos only -- the basis search cannot checkpoint in
    // this in-memory pipeline)
    prog.add_argument("--checkpoint", "-c")
        .help("Checkpoint file for resuming Lanczos (default: auto-generated)")
        .default_value(std::string(""));

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

    prog.add_argument("--batch-size")
        .help("Local basis states per MPI communication round for batched "
              "apply (-1 = all in one round). The default is bounded to keep "
              "the apply send/recv buffers small.")
        .default_value(1<<20)
        .scan<'i', int>();

    prog.add_argument("--noplan")
        .help("Skip the static apply plan (precomputed remote target indices) "
              "and fall back to the batched search path.")
        .default_value(false)
        .implicit_value(true);

    prog.add_argument("-o", "--output_dir")
        .required()
        .help("output directory (also used for Lanczos checkpoints)");

    provide_lanczos_options(prog);

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

        build_hamiltonian(H_sym, jdata, Jpm, B);
    }

    // make the out dir if not exists
    std::filesystem::create_directories(prog.get<std::string>("--output_dir"));

    s << prog.get<std::string>("--output_dir") << "/" << outfilename_buf;

    // Build checkpoint filename
    std::string checkpoint_file;
    if (prog.get<std::string>("--checkpoint").empty()) {
        checkpoint_file = s.str() + ".checkpoint.h5";
    } else {
        checkpoint_file = prog.get<std::string>("--checkpoint");
    }

    if (ctx.my_rank == 0) {
        std::cout << "[Main] Checkpoint file: " << checkpoint_file << "\n";
    }

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

    if (ctx.my_rank == 0) std::cout << "[Search] Building basis in memory...\n";

    size_t raw_local = 0;
    std::vector<Uint128> my_states;
    if (target_sector.empty()){
        my_states = build_basis_states<lat_container>(lat, num_spinon_pairs,
                perm, workdir, job_tag, prog, target_sector, H_sym, raw_local);
    } else {
        my_states = build_basis_states<lat_container_with_sector>(lat,
                num_spinon_pairs, perm, workdir, job_tag, prog, target_sector,
                H_sym, raw_local);
    }

    // An interrupted search means an incomplete basis: abort, do not diagonalise.
    int interrupted = GLOBAL_SHUTDOWN_REQUEST ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &interrupted, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (interrupted) {
        if (ctx.my_rank == 0) {
            std::cout << "[Main] Basis search was interrupted; the in-memory "
                "pipeline cannot resume a partial search. Exiting.\n";
        }
        MPI_Finalize();
        return 1;
    }

    size_t raw_global = 0;
    MPI_Allreduce(&raw_local, &raw_global, 1, get_mpi_type<size_t>(),
            MPI_SUM, MPI_COMM_WORLD);
    if (ctx.my_rank == 0) {
        std::cout << "[Search] Done! raw basis dim=" << raw_global << "\n";
    }

	// Step 3: redistribute to hash-correct ranks (states already trimmed
	// block-wise inside build_basis_states)
    basis_t basis;
    basis.adopt_states(std::move(my_states));
    basis.redistribute();
    std::cout<<"[MPI_BST]  Done! local basis dim="<<basis.dim()<<std::endl;
    if (ctx.my_rank == 0) {
        std::cout << "[MPI_BST] global basis dim=" << basis.global_dim()
                  << " (trimmed " << raw_global - basis.global_dim() << ")\n";
    }

    ////////////////////////////////////////
    // Do the diagonalisation
    lanczos_mpi::Settings settings(ctx);
    parse_lanczos_settings(prog, settings);
    settings.calc_eigenvector = true;

    const int batch_size = prog.get<int>("--batch-size");

    double eigval;
    std::vector<double> local_v0(basis.dim());
    std::vector<double> evector;
    std::vector<double> alphas, betas;

    // Scope the Hamiltonian so its apply plan (~4 B per off-diagonal nonzero)
    // and dy send/recv buffers are freed before the observable phase.
    {
        auto H = MPILazyOpSum(basis, H_sym, ctx);
        if (!prog.get<bool>("--noplan")) {
            H.build_plan(batch_size);
            H.release_state_buffers();
        } else {
            H.allocate_temporaries(batch_size);
        }

        RealApplyFn evadd = [&H](const coeff_t* x_local, coeff_t* y_local){
            H.evaluate_add(x_local, y_local);
        };

        std::cout << "[Lanczos] finding lowest eigenvalue\n";
        auto res=  lanczos_mpi::lanczos_iterate_checkpoint(evadd, local_v0, alphas, betas, settings, checkpoint_file);

        std::cout<<"[rank "<<ctx.my_rank<<"] "<<res;

        // If we hit checkpoint and exited early, clean exit
        if (!res.eigval_converged) {
            if (ctx.my_rank == 0) {
                std::cout << "[Main] Exited initial iteration at n="<<
                    res.n_iterations<<" due to time limit. Restart to continue.\n";
            }
            MPI_Finalize();
            return 0; // Clean exit for restart
        }

        // If converged, compute final eigenvalue and eigenvector
        std::cout << "[Lanczos] tridiagonalising in Krylov space\n";
        std::vector<double> ritz;
        {
            std::vector tmp_alphas(alphas);
            std::vector tmp_betas(betas);
            tridiagonalise_one(tmp_alphas, tmp_betas, eigval, ritz);
        }

        std::cout << "[Lanczos] iterating to determine eigenvector\n";
        evector.resize(basis.dim());

        // Second pass for eigenvector
        res = lanczos_mpi::lanczos_iterate_checkpoint(
            evadd, local_v0, alphas, betas, settings,
            checkpoint_file + ".eigvec",
            &ritz, &evector
        );
        // The ground state is accumulated into evector; local_v0 holds the
        // (now unneeded) final Lanczos vector and is recycled as a scratch
        // buffer in the observable phase below.

        // If we hit checkpoint and exited early (eigenvector not fully
        // reconstructed), clean exit
        if (!res.eigvec_converged) {
            if (ctx.my_rank == 0) {
            std::cout << "[Main] Exited second iteration at n="<<
                res.n_iterations<<" due to time limit. Restart to continue.\n";
            }
            MPI_Finalize();
            return 0; // Clean exit for restart
        }
    } // H freed here


    std::string out_filename = std::filesystem::path(s.str()+".eigs.h5")
        .replace_extension(".out.h5");

    std::cout << "Eigenvalues:\n" << eigval << "\n\n";
	std::cout << "Writing to\n"<<out_filename<<std::endl;

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
            std::cout<<"Compute <O>... "<<std::flush;

        std::vector<double> chi(basis.dim()); // |chi> = O_j |psi>
        std::vector<double>& u = local_v0;    // recycled: |u> = O_0 |psi>

        for (int opi=0; opi<n_operators; opi++){
            if (ctx.my_rank == 0) std::cout<<opi<<" " <<std::flush;
            // Constructed per iteration so the bounded comm buffers of each
            // operator are freed again before the next one.
            MPILazyOpSum<double, basis_t> op(basis, ringL[opi], ctx);
            op.allocate_temporaries(batch_size);
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
            if (ctx.my_rank == 0) std::cout<<"Compute <OOO>... "<<std::flush;
            // computing expectation values of the incomplete volumes
            for (int sl=0; sl<4; sl++){
                auto par_vol_operators = get_partial_vol_ops(jdata, ringL, sl);
                partial_vol[sl].resize(par_vol_operators.size(), 0.0);
                for (size_t opi=0; opi<par_vol_operators.size(); opi++){
                    if (ctx.my_rank == 0) std::cout<<opi<<" "<<std::flush;
                    // Per-iteration construction with bounded batches, as in
                    // the ring loop: the default allocate_temporaries() (-1 =
                    // whole slab in one round) would size the send/recv
                    // buffers at up to 6 records/state for these 6-term
                    // opsums -- far more than the basis itself.
                    MPILazyOpSum<double, basis_t> op(basis, par_vol_operators[opi], ctx);
                    op.allocate_temporaries(batch_size);
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
                std::cout<<"\nCompute <OOO> complete "<<std::endl;
        }


    }



    if (ctx.my_rank == 0){
        hid_t out_fid =
            H5Fcreate(out_filename.c_str(),
                    H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (out_fid < 0)
            throw std::runtime_error("Failed to create HDF5 file");
        std::cout << "Done!" << std::endl;

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

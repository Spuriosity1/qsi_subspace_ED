#include <argparse/argparse.hpp>


#include <nlohmann/json.hpp>
#include "hamiltonian_setup.hpp"
#include "operator_mpi.hpp"
#include "lanczos_mpi.hpp"
#include "lanczos_cli.hpp"
#include "expectation_eval.hpp"
#include <fstream>
#include <filesystem>

#include "checkpoint_utils.hpp"

using json = nlohmann::json;
using namespace projED;


int main(int argc, char* argv[]) {
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file");
	prog.add_argument("-s", "--sector");
    prog.add_argument("--basis_file", "-b")
        .help("A basis file (HDF5 format). Defaults to ${lattice_file%.json}.h5");

    // Checkpointing
    prog.add_argument("--checkpoint", "-c")
        .help("Checkpoint file for resuming (default: auto-generated)")
        .default_value(std::string(""));

    prog.add_argument("--checkpoint_interval")
        .help("Checkpoint every N iterations (0=only on time limit)")
        .default_value(0)
        .scan<'i', int>();

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
            .help("Jom, units of Jzz")
            .scan<'g', double>();
    }

    prog.add_argument("-o", "--output_dir")
        .required()
        .help("output directory");

    provide_lanczos_options(prog);
    prog.add_argument("--n_spinons")
        .default_value(0)
        .scan<'i', int>();

//    prog.add_argument("--max_iters")
//        .help("Max steps for iterative solver")
//        .default_value(1000)
//        .scan<'i', int>();
	
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

	// Step 2: Load basis from H5
    std::cout<<"Loading basis..."<<std::endl;

    MPI_ZBasisBST basis;
    std::cout<<"[MPI_BST]  Loading basis..."<<std::endl;
    MPIContext ctx = load_basis(basis, prog);
    std::cout<<"[MPI_BST]  Done! local basis dim="<<basis.dim()<<std::endl;


	using coeff_t=double;
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
        // Auto-generate checkpoint filename based on parameters
        checkpoint_file = s.str() + ".checkpoint.h5";
    } else {
        checkpoint_file = prog.get<std::string>("--checkpoint");
    }
    
    if (ctx.my_rank == 0) {
        std::cout << "[Main] Checkpoint file: " << checkpoint_file << "\n";
    }

    //////////////////////////////////
    /// Build H

	auto H = MPILazyOpSum(basis, H_sym, ctx);

    ////////////////////////////////////////
    // Do the diagonalisation
    lanczos_mpi::Settings settings(ctx);
    parse_lanczos_settings(prog, settings);

    RealApplyFn evadd = [H, ctx](const coeff_t* x_local, coeff_t* y_local){
        H.evaluate_add(x_local, y_local);
    };

    double eigval;
    std::vector<double> local_v0(ctx.local_block_size());

//    auto res = lanczos_mpi::eigval0(evadd, eigval, local_v0, settings);

    std::vector<double> alphas, betas;

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
    std::vector tmp_alphas(alphas);
    std::vector tmp_betas(betas);
    tridiagonalise_one(tmp_alphas, tmp_betas, eigval, ritz);

    if(settings.calc_eigenvector) {
        std::cout << "[Lanczos] iterating to determine eigenvector\n";
        std::vector<double> evector(ctx.local_block_size());
        
        // Second pass for eigenvector 
        res = lanczos_mpi::lanczos_iterate_checkpoint(
            evadd, local_v0, alphas, betas, settings, 
            checkpoint_file + ".eigvec",
            &ritz, &evector
        );
        std::swap(evector, local_v0);


        // If we hit checkpoint and exited early, clean exit
        if (!res.eigval_converged) {
            if (ctx.my_rank == 0) {
            std::cout << "[Main] Exited second iteration at n="<<
                res.n_iterations<<" due to time limit. Restart to continue.\n";
            }
            MPI_Finalize();
            return 0; // Clean exit for restart
        }
    }

    std::string filename = s.str()+".eigs.h5";

    std::cout << "Eigenvalues:\n" << eigval << "\n\n";
	std::cout << "Writing to\n"<<filename<<std::endl;

    // Create parallel file access property list
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);
    if (file_id < 0) throw std::runtime_error("Failed to create HDF5 file");


    // dummy (here in case we calc more later
    std::vector<double> eigvals{eigval};

    // Write eigenvectors using parallel I/O: shape (dim, n_eigvecs)
    {
        hsize_t global_dims[2] = {static_cast<hsize_t>(ctx.global_basis_dim()), 1};
        hsize_t local_size = ctx.local_block_size();
        hsize_t offset = ctx.local_start_index();
        
        // Create dataspace for the full dataset
        hid_t filespace = H5Screate_simple(2, global_dims, NULL);
        
        // Create dataset
        hid_t dset_id = H5Dcreate(file_id, "eigenvectors", H5T_NATIVE_DOUBLE, 
                                  filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Select hyperslab for this process
        hsize_t count[2] = {local_size, 1};
        hsize_t start[2] = {offset, 0};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL);
        
        // Create memory dataspace
        hid_t memspace = H5Screate_simple(2, count, NULL);
        
        // Create property list for collective dataset write
        hid_t plist_xfer = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_xfer, H5FD_MPIO_COLLECTIVE);
        
        // Write data
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_xfer, local_v0.data());
        
        // Close resources
        H5Pclose(plist_xfer);
        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Dclose(dset_id);
    }

    /// Close the parallel file
    H5Fclose(file_id);

    // Reopen file in serial mode on rank 0 to write strings
    if(ctx.my_rank == 0)
    {
        hid_t serial_file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

        // write the eigenvalues
        hsize_t dims[1] = {static_cast<hsize_t>(eigvals.size())};
        write_dataset(serial_file_id, "eigenvalues", eigvals.data(), dims, 1);
    
   
        fs::path latfile = prog.get<std::string>("lattice_file");
        write_string_to_hdf5(serial_file_id, "latfile_json", 
                latfile.filename() );
        std::string dset_name = prog.is_used("--sector") ?
             prog.get<std::string>("--sector") : "basis";
        write_string_to_hdf5(serial_file_id, "dset_name", 
                dset_name );
        
        H5Fclose(serial_file_id);
    }

    MPI_Finalize();



	return 0;
}


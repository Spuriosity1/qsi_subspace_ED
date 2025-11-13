#include <argparse/argparse.hpp>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>
#include <fstream>
#include "physics/geometry.hpp"

//#include "matrix_diag_bits.hpp"
#include "expectation_eval.hpp"
#include "basis_format_bits.hpp"
#include "operator_mpi.hpp"
//#include "admin.hpp"


using json=nlohmann::json;
using namespace std;

void obtain_flags(
    bool& calc_ring,
    bool& calc_ring_ring,
    bool& calc_partial_vol,
    const argparse::ArgumentParser& prog){

    calc_ring=true;
    calc_ring_ring=true;
    calc_partial_vol=true;

    if (prog.is_used("--calculate")){
        calc_ring =        false;
        calc_ring_ring =   false;
        calc_partial_vol = false;
        auto calc_opts = prog.get<std::vector<std::string>>("--calculate");
        for (auto& s : calc_opts){
            if (s == "ring") calc_ring=true;
            if (s == "ring_ring") calc_ring_ring=true;
            if (s == "partial_vol") calc_partial_vol=true;
        }
    }
}

//typedef ZBasisBST_MPI basis_t;
//

void load_state(std::vector<double>& psi, const MPI_ZBasisBST& basis, const std::filesystem::path& infile, const MPIctx& ctx, int col=0){
    
	hid_t file_id = -1, dataset_id = -1, dataspace_id = -1;
	hid_t plist_id = -1, dxpl_id = -1;
	herr_t status;
    try{
        // Create property list for parallel file access
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        if (plist_id < 0) throw HDF5Error(-1, -1, -1, "load_state: Failed to create file access property list");
        
        status = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
        H5Pset_all_coll_metadata_ops(plist_id, true);
        if (status < 0) throw HDF5Error(-1, -1, -1, "load_state: Failed to set MPI-IO properties");
        
        // Open the file collectively
        file_id = H5Fopen(infile.c_str(), H5F_ACC_RDONLY, plist_id);
		if (file_id < 0) throw HDF5Error(file_id, -1, -1, "load_state: Failed to open file");
		
		// Open the dataset
		dataset_id = H5Dopen(file_id, "eigenvectors", H5P_DEFAULT);
		if (dataset_id < 0) throw HDF5Error(file_id, -1, dataset_id, "load_state: Failed to open dataset");
		
		// Get the dataspace to retrieve the dimensions
		dataspace_id = H5Dget_space(dataset_id);
		if (dataspace_id < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "load_state: Failed to get dataspace");
		
		// Get the dimensions
		int ndims = H5Sget_simple_extent_ndims(dataspace_id);
		if (ndims != 2) throw HDF5Error(file_id, dataspace_id, dataset_id, "load_state: Expected 2D data");
		
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);
		if (status < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "load_state: Failed to get dimensions");
        
        hsize_t row_width = dims[1];
        if (col >= row_width) {
            throw std::runtime_error("col selected >= row width ("+
                    std::to_string(row_width)+")");}
        hsize_t total_rows = dims[0];
        if (total_rows == 0){
            throw std::runtime_error("eigenvector is empty!");
        }
        
        hsize_t local_count = ctx.local_block_size();
        hsize_t local_start = ctx.local_start_index();
        
        // Allocate memory for the result
        psi.resize(local_count);
        
        // Create property list for collective I/O
        dxpl_id = H5Pcreate(H5P_DATASET_XFER);
        if (dxpl_id < 0) throw std::runtime_error("load_state: Failed to create transfer property list");
        
        status = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);
        if (status < 0) throw std::runtime_error("load_state: Failed to set collective I/O");
        
        // Select hyperslab in file dataspace (even if local_count == 0)
        hsize_t file_offset[2] = { local_start, static_cast<hsize_t>(col) };
        hsize_t file_count[2]  = { local_count, 1 };
        
        if (local_count > 0) {
            status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, file_offset, nullptr, file_count, nullptr);
        } else {
            // Select nothing for ranks with no data
            status = H5Sselect_none(dataspace_id);
        }
        if (status < 0) throw std::runtime_error("load_state: Failed to select hyperslab");
        
        // Create memory dataspace
        hid_t memspace = -1;
        if (local_count > 0) {
            memspace = H5Screate_simple(2, file_count, nullptr);
        } else {
            memspace = H5Screate(H5S_NULL);
        }
        if (memspace < 0) throw std::runtime_error("load_state: Failed to create memspace");
        
        // Collective read (all ranks must participate)
        void* data_ptr = (local_count > 0) ? reinterpret_cast<void*>(psi.data()) : nullptr;
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace, dataspace_id, dxpl_id, data_ptr);
        
        H5Sclose(memspace);
        
        if (status < 0) throw std::runtime_error("load_state: Failed to read data collectively");
        
        // Print diagnostics (only on rank 0 to avoid clutter)
        if (ctx.my_rank == 0) {
            assert(ctx.idx_partition.size() == ctx.state_partition.size());
            std::cout<<"Loaded basis chunk. Partition scheme:\n index\t state\n";
            for (size_t i=0; i<ctx.idx_partition.size(); i++){
                std::cout<<ctx.idx_partition[i]<<"\t";
                printHex(std::cout, ctx.state_partition[i])<<"\n";
            }
        }
		
		// Clean up
		H5Pclose(dxpl_id);
		H5Pclose(plist_id);
		H5Sclose(dataspace_id);
		H5Dclose(dataset_id);
		H5Fclose(file_id);
		
	} catch (const HDF5Error& e){
		if (dxpl_id >= 0) H5Pclose(dxpl_id);
		if (plist_id >= 0) H5Pclose(plist_id);
		if (dataset_id >= 0) H5Dclose(dataset_id);
		if (dataspace_id >= 0) H5Sclose(dataspace_id);
		if (file_id >= 0) H5Fclose(file_id);
		throw;
	}
}

    
int main(int argc, char* argv[]) {
	argparse::ArgumentParser prog("eval_observables");
	prog.add_argument("output_file");
	prog.add_argument("--latfile_dir")
        .default_value("../lattice_files");
	prog.add_argument("-s", "--n_spinons")
        .default_value(0)
        .scan<'i',int>();
    prog.add_argument("--calculate")
        .choices("ring", "ring_ring", "partial_vol")
        .nargs(argparse::nargs_pattern::at_least_one);
	prog.add_argument("--n_eigvecs", "-N")
		.help("Number of eigenvectors to check")
		.default_value(2)
		.scan<'i', int>();

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
		std::cerr << err.what() << std::endl;
		std::cerr << prog;
        std::exit(1);
    }


    bool calc_ring, calc_ring_ring, calc_partial_vol;

    obtain_flags(calc_ring, calc_ring_ring, calc_partial_vol, prog);

    auto in_datafile=fs::path(prog.get<std::string>("output_file"));

    hid_t in_fid = H5Fopen(in_datafile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (in_fid == H5I_INVALID_HID) {
        cerr<<"Invalid data file: " <<in_datafile;
        return 1;
    }
    
    // loading the small data
    std::vector<double> eigvals = read_vector_h5(in_fid, "eigenvalues");
    std::string latfile_name = read_string_from_hdf5(in_fid, "latfile_json");
    std::string dset_name = read_string_from_hdf5(in_fid, "dset_name");
    fs::path latfile_dir(prog.get<std::string>("--latfile_dir"));
    H5Fclose(in_fid);


	// Step 1: Load ring data from JSON
    fs::path latfile = latfile_dir/latfile_name;
	std::ifstream jfile(latfile);
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
    // NOTE n_spinons not handled properly
    MPIContext ctx = load_basis(basis, prog);
    std::cout<<"[MPI_BST]  Done! local basis dim="<<basis.dim()<<std::endl;

    // Step 3: Slab load of the state
    std::vector<double> psi;
    load_state(psi, basis, in_datafile, ctx, 0);

    // Step 4: evaluate the observables (matrix free)
    


    



            
    return 0;
}

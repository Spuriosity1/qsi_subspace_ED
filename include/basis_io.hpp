#pragma once
#include "admin.hpp"
#include "bittools.hpp"
#include <hdf5.h>
#include <string>
#include <vector>

namespace basis_io {

inline auto write_line(FILE* of, const Uint128& b){
	return std::fprintf(of, "0x%016llx%016llx\n", b.uint64[1],b.uint64[0]);
}

inline bool read_line(FILE *infile, Uint128& b) {
	char buffer[40];  // Enough to hold "0x" + 32 hex digits + null terminator
	if (!std::fgets(buffer, sizeof(buffer), infile)) {
		return false;  // Return false on failure (e.g., EOF)
	}
	
	return std::sscanf(buffer, "0x%016llx%016llx", &b.uint64[1], &b.uint64[0]) == 2;
}

inline void write_basis_csv(const std::vector<Uint128>& state_list, 
		const std::string &outfilename) {
	FILE *outfile = std::fopen((outfilename + ".csv").c_str(), "w");
	for (auto b : state_list) {
	  basis_io::write_line(outfile, b);
	}

	std::fclose(outfile);
}


inline void write_basis_hdf5(const std::vector<Uint128>& state_list, const std::string& outfilename){
	// do this C style because the C++ API is borked
	

	hsize_t dims[2] = {state_list.size(),2};

	hid_t file_id = -1, dataspace_id = -1, dataset_id = -1;
	herr_t status;

	try {
		// Create a new HDF5 file
		file_id = H5Fcreate((outfilename+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
				H5P_DEFAULT);
		if (file_id < 0) throw HDF5Error(file_id, -1, -1, "write_basis: Failed to create file");

		// Create a dataspace
		dataspace_id = H5Screate_simple(2, dims, nullptr);
		if (dataspace_id < 0) throw HDF5Error(file_id, dataspace_id, -1, "write_basis: Failed to create dataspace");

		// Create the dataset
		dataset_id = H5Dcreate(file_id, "basis", H5T_NATIVE_UINT64, dataspace_id,
				H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		if (dataset_id < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "write_basis: Failed to create dataset");

		// Write data to the dataset
		status = H5Dwrite(dataset_id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL,
				H5P_DEFAULT, state_list.data());
		if (status < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "write_basis: Failed to write data");

		// Cleanup and close everything
		H5Dclose(dataset_id);
		H5Sclose(dataspace_id);
		H5Fclose(file_id);
		return;
	} catch (const HDF5Error& e){
		if (dataset_id >= 0) H5Dclose(dataset_id);
		if (dataspace_id >= 0) H5Sclose(dataspace_id);
		if (file_id >= 0) H5Fclose(file_id);
		throw;
	}
}



inline std::vector<Uint128> read_basis_hdf5(const std::string& infile) {
    // Result vector to store the loaded data
    std::vector<Uint128> result;
    
    // HDF5 identifiers
    hid_t file_id = -1, dataset_id = -1, dataspace_id = -1;
    herr_t status;
    
    try {
        // Open the HDF5 file for reading
        file_id = H5Fopen((infile + ".h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) throw HDF5Error(file_id, -1, -1, "read_basis: Failed to open file");
        
        // Open the dataset
        dataset_id = H5Dopen(file_id, "basis", H5P_DEFAULT);
        if (dataset_id < 0) throw HDF5Error(file_id, -1, dataset_id, "read_basis: Failed to open dataset");
        
        // Get the dataspace to retrieve the dimensions
        dataspace_id = H5Dget_space(dataset_id);
        if (dataspace_id < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "read_basis: Failed to get dataspace");
        
        // Get the dimensions
        int ndims = H5Sget_simple_extent_ndims(dataspace_id);
        if (ndims != 2) throw HDF5Error(file_id, dataspace_id, dataset_id, "read_basis: Expected 2D data");
        
        hsize_t dims[2];
        status = H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);
        if (status < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "read_basis: Failed to get dimensions");
        
        // Allocate memory for the result
        result.resize(dims[0]);
        
        // Read the data
        status = H5Dread(dataset_id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());
        if (status < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "read_basis: Failed to read data");
        
        // Clean up
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        
        return result;
    }
    catch (const std::exception& e) {
        // Clean up in case of error
        if (dataspace_id >= 0) H5Sclose(dataspace_id);
        if (dataset_id >= 0) H5Dclose(dataset_id);
        if (file_id >= 0) H5Fclose(file_id);
        throw; // Rethrow the exception
    }
}


inline std::vector<Uint128> read_basis_csv(const std::string &infilename) {
	FILE *infile = std::fopen((infilename + ".csv").c_str(), "r");
	if (!infile) {
		throw std::runtime_error("Failed to open file: " + infilename + ".csv");
	}
	std::vector<Uint128> state_list;
	Uint128 b;
	while (read_line(infile, b)) {
		state_list.emplace_back(b);
	}

	std::fclose(infile);
	return state_list;
}


/*
Multi-write implementation (pointless, all is in ram anyway)
void pyro_vtree_parallel::write_basis_hdf5(const std::string& outfile){
	// do this C style because the C++ API is borked
	hsize_t dims[2] = {n_states(),2};

    hid_t file_id = -1, dataspace_id = -1, dataset_id = -1, memspace_id = -1;
    herr_t status;

	size_t thread_idx=0;

	// specifying slabs for thread-by-thread writes
	hsize_t row_offset=0;
	hsize_t block_rows = 0;	
	hsize_t start[2] = {row_offset, 0};
	hsize_t block[2] = {block_rows, 2};

    // Create a new HDF5 file
    file_id = H5Fcreate((outfile+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
			H5P_DEFAULT);
    if (file_id < 0) goto error;

    // Create a dataspace
    dataspace_id = H5Screate_simple(2, dims, nullptr);
    if (dataspace_id < 0) goto error;

    // Create the dataset
    dataset_id = H5Dcreate(file_id, "basis", H5T_NATIVE_UINT64, dataspace_id,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) goto error;

    // Write data to the dataset
	for (thread_idx=0; thread_idx < this->n_threads; thread_idx++){
		if (state_set[thread_idx].empty()) continue;

		// specifying the block to write
		start[0] = row_offset;
		block[0] = state_set[thread_idx].size();

		memspace_id = H5Screate_simple(2, block, nullptr); 
		if (memspace_id < 0) goto error;

		// select slab
        status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, nullptr, block, nullptr);
        if (status < 0) goto error;	
		
		// write slab
		status = H5Dwrite(dataset_id, H5T_NATIVE_UINT64, memspace_id,
				dataspace_id, H5P_DEFAULT, state_set[thread_idx].data());
		H5Sclose(memspace_id); memspace_id = -1;
		if (status < 0) goto error;

		row_offset += block[0];
	}

    // Cleanup and close everything
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    return;

error:
    if (memspace_id >= 0) H5Sclose(memspace_id);
    if (dataset_id >= 0) H5Dclose(dataset_id);
    if (dataspace_id >= 0) H5Sclose(dataspace_id);
    if (file_id >= 0) H5Fclose(file_id);
	std::cerr << "memspace id " << memspace_id; 
    throw HDF5Error(file_id, dataspace_id, dataset_id, "write_basis");
}

*/


}; // end namespace basis_io

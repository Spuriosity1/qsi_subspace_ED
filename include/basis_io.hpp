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
		if (dataset_id < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "write_basis: Failed to write data");

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


}; // end namespace basis_io

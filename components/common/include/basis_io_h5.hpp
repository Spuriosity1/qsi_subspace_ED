#pragma once
#include "bittools.hpp"
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <hdf5.h>


class HDF5Error : public std::runtime_error {
public:
    HDF5Error(hid_t file_id, hid_t dataspace_id, hid_t dataset_id, const std::string& message)
        : std::runtime_error(formatMessage(file_id, dataspace_id, dataset_id, message)) {}

private:
    static std::string formatMessage(hid_t file_id, hid_t dataspace_id, hid_t dataset_id, const std::string& message) {
        std::ostringstream oss;
        oss << "HDF5 Error: " << message << "\n"
            << "  File ID: " << file_id << "\n"
            << "  Dataspace ID: " << dataspace_id << "\n"
            << "  Dataset ID: " << dataset_id;
        return oss.str();
    }
};



namespace basis_io {

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


// sector-based operations


// Helper function to create sector string
inline std::string make_sector_string(const std::vector<int>& sector) {
    std::stringstream ss;
    ss << "basis_s" << sector[0] << "." << sector[1] << "." << sector[2] << "." << sector[3];
    return ss.str();
}

inline std::string make_sector_string(const std::array<int, 4>& sector) {
    std::stringstream ss;
    ss << "basis_s" << sector[0] << "." << sector[1] << "." << sector[2] << "." << sector[3];
    return ss.str();
}


inline std::vector<Uint128> read_basis_hdf5(const std::string& infile,
        const char*dset_name = "basis") {
	// Result vector to store the loaded data
	std::vector<Uint128> result;
	
	// HDF5 identifiers
	hid_t file_id = -1, dataset_id = -1, dataspace_id = -1;
	herr_t status;
	
	try {
		// Open the HDF5 file for reading
		file_id = H5Fopen((infile).c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file_id < 0) throw HDF5Error(file_id, -1, -1, "read_basis: Failed to open file");
		
		// Open the dataset
		dataset_id = H5Dopen(file_id, dset_name, H5P_DEFAULT);
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


}; // end namespace basis_io


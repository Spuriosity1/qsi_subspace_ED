#include <cstdio>
#include <pyro_tree.hpp>



inline std::vector<Uint128> read_basis_hdf5(const std::string& infile) {
    hid_t file_id = -1, dataspace_id = -1, dataset_id = -1;
    herr_t status;
    hsize_t dims[2];

	std::vector<Uint128> state_list;

    // Open the HDF5 file
    file_id = H5Fopen((infile + ".h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) goto error;

    // Open the dataset
    dataset_id = H5Dopen(file_id, "basis", H5P_DEFAULT);
    if (dataset_id < 0) goto error;

    // Get the dataspace and its dimensions
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) goto error;

    if (H5Sget_simple_extent_dims(dataspace_id, dims, nullptr) < 0) goto error;
    if (dims[1] != 2) goto error; // Ensure the second dimension is as expected

    // Resize vector to hold the data
    state_list.resize(dims[0]);

    // Read the data into the vector
    status = H5Dread(dataset_id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, state_list.data());
    if (status < 0) goto error;

    // Cleanup and close everything
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    return state_list;

error:
    if (dataset_id >= 0) H5Dclose(dataset_id);
    if (dataspace_id >= 0) H5Sclose(dataspace_id);
    if (file_id >= 0) H5Fclose(file_id);
    throw HDF5Error(file_id, dataspace_id, dataset_id, "read_basis");
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
};


int main (int argc, char *argv[]) {
	if (argc < 2){
		printf("USAGE: %s CSV_FILE",argv[0]);
		return 1;
	}



	return 0;
}

#include "expectation_eval.hpp"
#include <iostream>


std::tuple<std::vector<SymbolicPMROperator>,std::vector<SymbolicPMROperator>,
    std::vector<int>> 
get_ring_ops(
const nlohmann::json& jdata) {

    std::vector<SymbolicPMROperator> op_list;
    std::vector<SymbolicPMROperator> op_H_list;
    std::vector<int> sl_list;


	for (const auto& ring : jdata.at("rings")) {
		std::vector<int> spins = ring.at("member_spin_idx");

		std::vector<char> ops;
		std::vector<char> conj_ops;
		for (auto s : ring.at("signs")){
			ops.push_back( s == 1 ? '+' : '-');
			conj_ops.push_back( s == 1 ? '-' : '+');
		}
		
		int sl = ring.at("sl").get<int>();
		auto O   = SymbolicPMROperator(     ops, spins);
		auto O_h = SymbolicPMROperator(conj_ops, spins);
        op_list.push_back(O);
        op_H_list.push_back(O_h);
        sl_list.push_back(sl);
	}
    return std::make_tuple(op_list, op_H_list, sl_list);
}


std::tuple<std::vector<SymbolicPMROperator>,std::vector<SymbolicPMROperator>,
    std::vector<int>> 
get_vol_ops(
const nlohmann::json& jdata,
    const std::vector<SymbolicPMROperator>& ring_list,
    const std::vector<SymbolicPMROperator>& ring_H_list
) {

    std::vector<SymbolicPMROperator> op_list;
    std::vector<SymbolicPMROperator> op_H_list;
    std::vector<int> sl_list;
    
	for (const auto& vol : jdata.at("vols")) {
		std::vector<int> plaqi = vol.at("member_plaq_idx");
        SymbolicPMROperator volOp("");
        SymbolicPMROperator volOp_H("");

        for (auto J : plaqi){
            volOp *= ring_list[J];
            volOp_H *= ring_H_list[J];
        } 
	    
        op_list.push_back(volOp);
        op_H_list.push_back(volOp_H);
        sl_list.push_back(vol.at("sl").get<int>());

	}
    return std::make_tuple(op_list, op_H_list, sl_list);
}



double compute_expectation(
    const ZBasis& basis,
    const Eigen::MatrixXd& eigenvectors,
    const Eigen::SparseMatrix<double>& op,
    int i, int j
) {

    const auto& psi_i = eigenvectors.col(i);
    const auto& psi_j = eigenvectors.col(j);

    return  psi_j.dot( op * psi_i);
}


std::vector<double> compute_all_expectations(
    const ZBasis& basis,
    const Eigen::MatrixXd& eigenvectors,
    const std::vector<Eigen::SparseMatrix<double>>& ops
) {
    const int num_ops = ops.size();
    const int num_vecs = eigenvectors.cols();

    std::vector<double> result(num_ops * num_vecs * num_vecs);

    for (int l = 0; l < num_ops; ++l) {
        for (int i = 0; i < num_vecs; ++i) {
            for (int j = 0; j < num_vecs; ++j) {
                double val = compute_expectation(basis, eigenvectors, ops[l], i, j);
                result[l * num_vecs * num_vecs + i * num_vecs + j] = val;
            }
        }
    }

    return result;
}



void write_dataset(hid_t file_id, const char* name, const double* data, hsize_t* dims, int rank) {

    // Create the data space for the dataset
    hid_t dataspace_id = H5Screate_simple(rank, dims, NULL);
    if (dataspace_id < 0) throw std::runtime_error("Failed to create dataspace");

    // Create the dataset
    hid_t dataset_id = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE,
                            dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) throw std::runtime_error("Failed to create dataset");

    // Write the data to the dataset
    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data);
    if (status < 0) throw std::runtime_error("Failed to write dataset");

    // Clean up
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
};


void write_expectation_vals_h5(
    hid_t file_id,
    const char* name,
    const std::vector<double>& data,
    int num_ops,
    int num_vecs
) {

    // Define the dimensions of the dataset
    hsize_t dims[3] = {
        static_cast<hsize_t>(num_ops),
        static_cast<hsize_t>(num_vecs),
        static_cast<hsize_t>(num_vecs)
    };

    write_dataset(file_id, name, data.data(), dims, 3);
 
}


// helper HDF5 functions

void write_string_to_hdf5(hid_t file_id, const std::string& dataset_name, const std::string& value) {
    // Step 1: Create scalar dataspace (for a single string)
    hid_t dataspace_id = H5Screate(H5S_SCALAR);

    // Step 2: Create string datatype (variable-length string)
    hid_t dtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(dtype, H5T_VARIABLE);

    // Step 3: Create the dataset
    hid_t dset_id = H5Dcreate(file_id, dataset_name.c_str(), dtype, dataspace_id,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Step 4: Write the string
    const char* c_str = value.c_str();
    H5Dwrite(dset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &c_str);

    // Step 5: Close resources
    H5Dclose(dset_id);
    H5Tclose(dtype);
    H5Sclose(dataspace_id);
}




std::vector<double> read_vector_h5(hid_t file_id, const std::string& name) {
    hid_t dset_id = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
    hid_t space_id = H5Dget_space(dset_id);
    
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space_id, dims, nullptr);
    std::vector<double> data(dims[0]);

    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    H5Sclose(space_id);
    H5Dclose(dset_id);
    return data;
}


Eigen::MatrixXd read_matrix_h5(hid_t file_id, const std::string& name) {
    hid_t dset_id = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
    hid_t space_id = H5Dget_space(dset_id);

    hsize_t dims[2];
    H5Sget_simple_extent_dims(space_id, dims, nullptr);
    Eigen::MatrixXd mat(dims[0], dims[1]);

    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data());

    H5Sclose(space_id);
    H5Dclose(dset_id);
    return mat;
}


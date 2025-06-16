#include "expectation_eval.hpp"


Eigen::MatrixXd compute_expectation_values(
    const ZBasis& basis,
    const Eigen::MatrixXd& eigenvectors,
    const std::vector<SymbolicPMROperator>& ops
) {
    const int nStates = static_cast<int>(eigenvectors.cols());
    const int nOps = static_cast<int>(ops.size());

    Eigen::MatrixXd result(nOps, nStates);

    for (int k = 0; k < nStates; ++k) {
        const auto& psi_k = eigenvectors.col(k);

        for (int op_idx = 0; op_idx < nOps; ++op_idx) {
            Eigen::VectorXd temp = Eigen::VectorXd::Zero(psi_k.size());
            ops[op_idx].apply(basis, psi_k, temp);
            result(op_idx, k) = psi_k.dot(temp);
        }
    }

    return result;
}

Eigen::MatrixXd compute_cross_terms(
    const ZBasis& basis,
    const Eigen::MatrixXd& eigenvectors,
    const std::vector<SymbolicPMROperator>& ops,
    int i, int j
) {
    const int nOps = static_cast<int>(ops.size());
    Eigen::MatrixXd result(nOps, 2); // col 0: ⟨i|O|j⟩, col 1: ⟨j|O|i⟩

    const auto& psi_i = eigenvectors.col(i);
    const auto& psi_j = eigenvectors.col(j);

    for (int op_idx = 0; op_idx < nOps; ++op_idx) {
        Eigen::VectorXd temp_i = Eigen::VectorXd::Zero(psi_i.size());
        ops[op_idx].apply(basis, psi_j, temp_i);
        result(op_idx, 0) = psi_i.dot(temp_i);

        Eigen::VectorXd temp_j = Eigen::VectorXd::Zero(psi_j.size());
        ops[op_idx].apply(basis, psi_i, temp_j);
        result(op_idx, 1) = psi_j.dot(temp_j);
    }

    return result;
}


void save_expectation_data_to_hdf5(
    const std::string& filename,
    const Eigen::VectorXd& eigvals,
    const Eigen::MatrixXd& diag_vals,
    const Eigen::MatrixXd& cross_vals,
    const std::vector<int>& group_ids
) {
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) throw std::runtime_error("Failed to create HDF5 file");

    // Helper lambda to write a dataset
    auto write_dataset = [](hid_t file_id, const char* name, const double* data, hsize_t* dims, int rank) {
        hid_t dataspace_id = H5Screate_simple(rank, dims, NULL);
        hid_t dataset_id = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, dataspace_id,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    };

    // Write eigenvalues: shape (N,)
    {
        hsize_t dims[1] = {static_cast<hsize_t>(eigvals.size())};
        write_dataset(file_id, "eigenvalues", eigvals.data(), dims, 1);
    }

    // Write diag_vals: shape (n_ops, N)
    {
        hsize_t dims[2] = {static_cast<hsize_t>(diag_vals.rows()), static_cast<hsize_t>(diag_vals.cols())};
        write_dataset(file_id, "expectation_values", diag_vals.data(), dims, 2);
    }

    // Write cross_vals: shape (n_ops, 2)
    {
        hsize_t dims[2] = {static_cast<hsize_t>(cross_vals.rows()), static_cast<hsize_t>(cross_vals.cols())};
        write_dataset(file_id, "cross_terms", cross_vals.data(), dims, 2);
    }

    // Write group_ids (int): shape (n_ops,)
    {
        hsize_t dims[1] = {static_cast<hsize_t>(group_ids.size())};
        hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dataset_id = H5Dcreate2(file_id, "operator_group_ids", H5T_NATIVE_INT, dataspace_id,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, group_ids.data());
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    }

    H5Fclose(file_id);
}


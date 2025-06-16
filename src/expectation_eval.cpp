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
    using namespace HighFive;

    File file(filename, File::Overwrite);

    file.createDataSet("eigenvalues", DataSpace::From(eigvals)).write(eigvals);
    file.createDataSet("expectation_values", DataSpace::From(diag_vals)).write(diag_vals);
    file.createDataSet("cross_terms", DataSpace::From(cross_vals)).write(cross_vals);

    // Store operator group IDs
    file.createDataSet("operator_group_ids", DataSpace::From(group_ids)).write(group_ids);
}


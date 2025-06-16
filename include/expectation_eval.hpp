#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <hdf5.h>

#include "operator.hpp"

// Computes ⟨ψ_k| O |ψ_k⟩ for all eigenstates and operators
Eigen::MatrixXd compute_expectation_values(
    const ZBasis& basis,
    const Eigen::MatrixXd& eigenvectors,
    std::vector<SymbolicPMROperator>& grouped_ops
);

// Computes cross terms ⟨ψ_i| O |ψ_j⟩ for given i and j
Eigen::MatrixXd compute_cross_terms(
    const ZBasis& basis,
    const Eigen::MatrixXd& eigenvectors,
    std::vector<SymbolicPMROperator>& grouped_ops,
    int i, int j
);

// Saves results to an HDF5 file
void save_expectation_data_to_hdf5(
    const std::string& filename,
    const Eigen::VectorXd& eigvals,
    const Eigen::MatrixXd& diag_vals,
    const Eigen::MatrixXd& cross_vals,
    const std::vector<int>& group_ids
);


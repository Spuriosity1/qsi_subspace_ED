#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <string>
#include <hdf5.h>
#include "operator.hpp"

#include <nlohmann/json.hpp>


std::tuple<std::vector<SymbolicPMROperator>,std::vector<SymbolicPMROperator>,
    std::vector<int>> 
get_ring_ops(const nlohmann::json& jdata);


/*
std::pair<std::vector<SymbolicOpSum<double>>,
    std::vector<int>> 
get_vol_ops(
        const nlohmann::json& jdata,
        const std::vector<SymbolicPMROperator>& ring_list
);
*/

std::vector<SymbolicOpSum<double>>
get_partial_vol_ops(
        const nlohmann::json& jdata,
        const std::vector<SymbolicPMROperator>& ring_list,
        int sl
);




// Computes cross terms ⟨ψ_i| O |ψ_j⟩ for given i and j for operator O
double compute_expectation(
    const ZBasis& basis,
    const Eigen::MatrixXd& eigenvectors,
    const Eigen::SparseMatrix<double>& op,
    int i, int j
);


double compute_expectation(
    const ZBasis& basis,
    const Eigen::MatrixXd& eigenvectors,
    const SymbolicPMROperator& op,
    int i, int j
);


template<typename T>
std::vector<double>
compute_all_expectations(
    const Eigen::MatrixXd& eigenvectors,
    const std::vector<T>& ops
);


template<typename T>
std::vector<double>
compute_cross_correlation(
    const Eigen::MatrixXd& eigenvectors,
    const std::vector<T>& ops
);


// H5 IO niceties
//
std::vector<double> read_vector_h5(hid_t file_id, const std::string& name);
Eigen::MatrixXd read_matrix_h5(hid_t file_id, const std::string& name);


// Saves results to an HDF5 file
void write_expectation_vals_h5(
    hid_t file_id,
    const char* name,
    const std::vector<double>& data,
    int num_ops,
    int num_vecs
);

// Saves cross-correlations to h5
void write_cross_corr_vals_h5(
    hid_t file_id,
    const char* name,
    const std::vector<double>& data,
    int num_ops,
    int num_vecs
);



void write_string_to_hdf5(hid_t file_id, const std::string& dataset_name, const std::string& value);

void write_dataset(hid_t file_id, const char* name, const double* data, hsize_t* dims, int rank);


template<typename T>
requires std::convertible_to<T, int>
void write_integer(hid_t file_id, const char* name, T value);



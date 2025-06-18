#include <argparse/argparse.hpp>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "matrix_diag_bits.hpp"
#include "physics/Jring.hpp"

#include "Spectra/Util/SelectionRule.h"
#include "operator.hpp"

#include "expectation_eval.hpp"



// Helper function to split a string by a delimiter
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        tokens.push_back(item);
    }
    return tokens;
}

// Extracts key-value pairs of the form key=value from the filename
std::unordered_map<std::string, std::string> parse_parameters(const std::string& path) {
    std::unordered_map<std::string, std::string> result;

    // Find the start of the parameter section
    size_t start = path.find_last_of('/');
    std::string filename = (start != std::string::npos) ? path.substr(start + 1) : path;

    // Split on '%'
    std::vector<std::string> parts = split(filename, '%');
    for (const auto& part : parts) {
        size_t eq_pos = part.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = part.substr(0, eq_pos);
            std::string value = part.substr(eq_pos + 1);
            result[key] = value;
        }
    }

    return result;
}


std::string read_string_from_hdf5(hid_t file_id, const std::string& dataset_name) {
    hid_t dset_id = H5Dopen(file_id, dataset_name.c_str(), H5P_DEFAULT);
    hid_t dtype = H5Dget_type(dset_id);

    char* rdata;  // HDF5 will allocate memory for this
    H5Dread(dset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &rdata);

    std::string result(rdata);
    free(rdata);  // Free the memory allocated by HDF5

    H5Tclose(dtype);
    H5Dclose(dset_id);
    return result;
}


using json=nlohmann::json;
    
int main(int argc, char* argv[]) {
	argparse::ArgumentParser prog("build_ham");
	prog.add_argument("output_file");
	prog.add_argument("--latfile_dir")
        .default_value("../lattice_files");
	prog.add_argument("-s", "--sector");
	prog.add_argument("-s", "--n_spinons")
        .default_value(0)
        .scan<'i',int>();


    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
		std::cerr << err.what() << std::endl;
		std::cerr << prog;
        std::exit(1);
    }

    auto in_datafile=fs::path(prog.get<std::string>("output_file"));

    hid_t file_id = H5Fopen(in_datafile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
 
    std::vector<double> eigvals = read_vector_h5(file_id, "eigenvalues");
    Eigen::MatrixXd eigvecs = read_matrix_h5(file_id, "eigenvectors");
    std::string latfile_name = read_string_from_hdf5(file_id, "latfile_json");

    int n_eigvecs = eigvecs.cols();
    std::cout<<"Calculating expectation vals of "<<n_eigvecs<<" lowest |psi>\n";

    fs::path latfile_dir(prog.get<std::string>("--latfile_dir"));

    fs::path latfile = latfile_dir/latfile_name;
	std::ifstream jfile(latfile);
	if (!jfile) {
		std::cerr << "Failed to open JSON file at " << latfile <<std::endl;
		return 1;
	}
	json jdata;
	jfile >> jdata;


	ZBasis basis;

    fs::path basisfile = get_basis_file(latfile, prog);
    basis.load_from_file(basisfile);
    std::cout <<"Loading from "<<basisfile << std::endl;
 
    auto [ringL, ringR, sl] = get_ring_ops(jdata);

    std::vector<LazyOpSum<double>> l_ring_operators;
    std::vector<Eigen::SparseMatrix<double>> ring_operators;
    ring_operators.reserve(ringL.size());
    for (auto& O : ringL){
        l_ring_operators.emplace_back(basis, O);
        ring_operators.push_back(l_ring_operators.back().toSparseMatrix());
    }
 
    auto ring_expect = compute_all_expectations(basis, eigvecs, ring_operators);

    // Create a new file using default properties
    file_id = H5Fcreate(in_datafile.replace_extension(".out.h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) throw std::runtime_error("Failed to create HDF5 file");


    // save the eigvals for convenience
    {
        write_string_to_hdf5(file_id, "latfile_json", 
                latfile.filename() );
    }

    {
        hsize_t dims[1] = {static_cast<hsize_t>(eigvals.size())};
        write_dataset(file_id, "eigenvalues", eigvals.data(), dims, 1);
    }

    {
        write_expectation_vals_h5(file_id, "ring", ring_expect, 
                ringL.size(), n_eigvecs);
    }

            
    return 0;
}

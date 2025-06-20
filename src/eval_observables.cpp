#include <argparse/argparse.hpp>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "matrix_diag_bits.hpp"
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

    hid_t in_fid = H5Fopen(in_datafile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    
    // loading the relevant data
    std::vector<double> eigvals = read_vector_h5(in_fid, "eigenvalues");
    Eigen::MatrixXd eigvecs = read_matrix_h5(in_fid, "eigenvectors");
    std::string latfile_name = read_string_from_hdf5(in_fid, "latfile_json");
    std::string dset_name = read_string_from_hdf5(in_fid, "dset_name");
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
    // NOTE n_spinons not handled properly
    basis.load_from_file( get_basis_file(latfile, 0, dset_name!="basis"), 
            dset_name
            );
    
    


///////////////////////////////////////////////////////////////////////////////
/// Setup done, load the data

    // Create a new file using default properties
    hid_t out_fid = H5Fcreate(in_datafile.replace_extension(".out.h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (out_fid < 0) throw std::runtime_error("Failed to create HDF5 file");


    {
        // remember what the lattice is
        write_string_to_hdf5(out_fid, "latfile_json", 
                latfile.filename() );

        // save the eigvals for convenience
        hsize_t dims[1] = {static_cast<hsize_t>(eigvals.size())};
        write_dataset(out_fid, "eigenvalues", eigvals.data(), dims, 1);
    }
 
    auto [ringL, ringR, sl] = get_ring_ops(jdata);
    int n_eigvecs = eigvecs.cols();

    // convert the symols into actual matrices 
    std::vector<LazyOpSum<double>> lazy_ring_operators;
    std::vector<Eigen::SparseMatrix<double>> ring_operators;
    ring_operators.reserve(ringL.size());
    for (auto& O : ringL){
        lazy_ring_operators.emplace_back(basis, O);
        ring_operators.push_back(lazy_ring_operators.back().toSparseMatrix());
    }

    std::cout<<"Calculating expectation vals of "<<n_eigvecs<<" lowest |psi>\n";

    {
        // computing all one-ring expectation values
        auto ring_expect = compute_all_expectations(eigvecs, ring_operators);
 
        write_expectation_vals_h5(out_fid, "ring", ring_expect, 
                ringL.size(), n_eigvecs); 
    }

    {
        // computing all two-ring expectation values
        auto OO_expect = compute_cross_correlation(eigvecs, ring_operators);
        
        write_cross_corr_vals_h5(out_fid, "ring_ring", OO_expect, 
                ringL.size(), n_eigvecs); 

    }

    {
        // computing expectation values of the incomplete volumes 
        std::array<std::vector<double>, 4> partial_vol;
        for (int sl=0; sl<4; sl++){
            auto par_vol_operators = get_partial_vol_ops(jdata, ringL, 0); 
            std::vector<LazyOpSum<double>> lazy_par_vol_ops;
            for (auto& v : par_vol_operators){
                lazy_par_vol_ops.emplace_back(basis, v);
            }

            partial_vol[sl].reserve(lazy_par_vol_ops.size());
            partial_vol[sl]  = compute_all_expectations(eigvecs, lazy_par_vol_ops);
        }

        // save the incomplete vol operators (each sl)
        for (size_t i = 0; i < partial_vol.size(); ++i) {
            const auto& vec = partial_vol[i];
            hsize_t dims[1] = { vec.size() };
            std::string name = "partial_vol_sl" + std::to_string(i);
            write_dataset(out_fid, name.c_str(), vec.data(), dims, 1);
        }
    }



            
    return 0;
}

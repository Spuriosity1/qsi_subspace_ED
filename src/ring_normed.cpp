#include <argparse/argparse.hpp>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>
#include <fstream>
#include "physics/geometry.hpp"

//#include "matrix_diag_bits.hpp"
#include "expectation_eval.hpp"
#include "basis_format_bits.hpp"
#include "operator.hpp"
//#include "admin.hpp"


using json=nlohmann::json;
using namespace std;

void obtain_flags(
    bool& calc_ring,
    bool& calc_ring_ring,
    bool& calc_partial_vol,
    const argparse::ArgumentParser& prog){

    calc_ring=true;
    calc_ring_ring=true;
    calc_partial_vol=true;

}

typedef ZBasisBST basis_t;

    
int main(int argc, char* argv[]) {
	argparse::ArgumentParser prog("eval_observables");
	prog.add_argument("output_file");
	prog.add_argument("--latfile_dir")
        .default_value("../lattice_files");
	prog.add_argument("-s", "--n_spinons")
        .default_value(0)
        .scan<'i',int>();
	prog.add_argument("--n_eigvecs", "-N")
		.help("Number of eigenvectors to check")
		.default_value(2)
		.scan<'i', int>();

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
		std::cerr << err.what() << std::endl;
		std::cerr << prog;
        std::exit(1);
    }


    bool calc_ring, calc_ring_ring, calc_partial_vol;

    obtain_flags(calc_ring, calc_ring_ring, calc_partial_vol, prog);

    auto in_datafile=fs::path(prog.get<std::string>("output_file"));

    hid_t in_fid = H5Fopen(in_datafile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (in_fid == H5I_INVALID_HID) {
        cerr<<"Invalid data file: " <<in_datafile;
        return 1;
    }
    
    // loading the relevant data
    std::vector<double> eigvals = read_vector_h5(in_fid, "eigenvalues");
    Eigen::MatrixXd eigvecs = read_matrix_h5(in_fid, "eigenvectors");
    std::string latfile_name = read_string_from_hdf5(in_fid, "latfile_json");
    std::string dset_name = read_string_from_hdf5(in_fid, "dset_name");
    fs::path latfile_dir(prog.get<std::string>("--latfile_dir"));

    // truncate to the requested # of eigenvectors
    int n_eigvecs = std::min(static_cast<int>(eigvecs.cols()), prog.get<int>("--n_eigvecs"));
    if (eigvals.size() > 1){
    assert(eigvals[0] <= eigvals[1]); // make sure we don't so sth stupid
    }

    fs::path latfile = latfile_dir/latfile_name;
	std::ifstream jfile(latfile);
	if (!jfile) {
		std::cerr << "Failed to open JSON file at " << latfile <<std::endl;
		return 1;
	}
	json jdata;
	jfile >> jdata;

    cout<<"Importing basis... "<<flush;

	basis_t basis;
    // NOTE n_spinons not handled properly
    basis.load_from_file( get_basis_file(latfile, 0, dset_name!="basis"), 
            dset_name
            );
    
    cout<<"Done!"<<endl;
    

    // build the matrix free operators
    auto [ringL, ringR, sl] = get_ring_ops(jdata);
    std::vector<LazyOpSum<double, basis_t>> lazy_ringL_Op;
    std::vector<LazyOpSum<double, basis_t>> lazy_ringR_Op;
    std::vector<LazyOpSum<double, basis_t>> lazy_projector;
    std::vector<LazyOpSum<double, basis_t>> lazy_OO;

    for (size_t i=0; i<ringL.size(); i++){
        auto O = ringL[i];
        auto O_H = ringR[i];
        lazy_ringL_Op.emplace_back(basis, O);
        lazy_ringR_Op.emplace_back(basis, O_H);
        SymbolicOpSum<double> P; P += O*O_H; P += O_H*O;
        lazy_projector.emplace_back(basis, P);
        auto tmp = ringR[0] * O;
        SymbolicOpSum<double> OO ( tmp );
        lazy_OO.emplace_back(basis, OO);
    }

///////////////////////////////////////////////////////////////////////////////
/// Setup done, load the data

    // Create a new file using default properties
    hid_t out_fid = H5Fcreate(in_datafile.replace_extension(".obs.h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (out_fid < 0) throw std::runtime_error("Failed to create HDF5 file");
    {
        // remember what the lattice is
        write_string_to_hdf5(out_fid, "latfile_json", 
                latfile.filename() );

        // save the eigvals for convenience
        hsize_t dims[1] = {static_cast<hsize_t>(eigvals.size())};
        write_dataset(out_fid, "eigenvalues", eigvals.data(), dims, 1);
    }

    cout<<"Compute <O_i>... "<<endl;
    // computing all one-ring expectation values
    auto ring_expect = compute_all_expectations(eigvecs.leftCols(n_eigvecs), lazy_ringL_Op);
    write_expectation_vals_h5(out_fid, "ringL_i", ring_expect, ringL.size(), n_eigvecs); 

    cout<<"Compute <O_i O'_i + O'_i O_i>... "<<endl;
    auto total_expect = compute_all_expectations(eigvecs.leftCols(n_eigvecs), lazy_projector);
    write_expectation_vals_h5(out_fid, "norm_i", total_expect, total_expect.size(), n_eigvecs); 

    cout<<"Compute <O'_0 O_j>... "<<endl;
    auto ring_ring = compute_all_expectations(eigvecs.leftCols(n_eigvecs), lazy_OO);
    write_expectation_vals_h5(out_fid, "ringR_0 ringL_i", ring_ring, ring_ring.size(), n_eigvecs); 
    cout<<"Done!"<<endl;


            
    return 0;
}

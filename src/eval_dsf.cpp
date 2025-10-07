#include <argparse/argparse.hpp>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "matrix_diag_bits.hpp"
#include "operator.hpp"

#include "expectation_eval.hpp"
#include "admin.hpp"


using json=nlohmann::json;
using namespace std;


// calculates a diagonal vector rep of S^z_i
void calc_Sz_rep(
    std::vector<double>& Sz_diagvals,
    const ZBasisBST& basis,
    size_t spin_i
    ){
    for (ZBasisBase::idx_t i=0; i<basis.dim(); i++){
        auto state = basis[i];
        Sz_diagvals[i] = 1.0 * ((state & spin_i) != 0) -0.5;
        // note for later: could in principle shift by 1/2 for a factor of 2 speedup
    }
}

template<typename T1, typename T2, typename T3>
double triple_sum(
    T1& x,
    T2& y,
    T3& z){
    double tmp=0;
    for (size_t i=0; i<x.size(); i++) {
        tmp += x[i]*y[i]*z[i];
    }
    return tmp;
}

const double R3 =sqrt(3);

const std::array<Eigen::Vector3d, 4> local_sz = {
    Eigen::Vector3d( 1./R3, 1./R3, 1./R3),
    Eigen::Vector3d( 1./R3,-1./R3,-1./R3),
    Eigen::Vector3d(-1./R3, 1./R3,-1./R3),
    Eigen::Vector3d(-1./R3,-1./R3, 1./R3)
};

inline std::vector<std::complex<double>> compute_scalar_Szz_of_omega(
    const Eigen::MatrixXd& eigvecs,
    const ZBasisBST& basis,
    const nlohmann::json& jdata,
    const std::vector<Eigen::Vector3d>& k_vals
    )
{
    int n_vecs = eigvecs.cols();

    const auto& atom_list = jdata.at("atoms");

    // temporaries
    std::vector<Eigen::Vector3d> pos;
    std::vector<double> Sz(basis.dim());
    
    // Storage: sparse, intensities[k_idx, E_idx] (row-major)
    std::vector<std::complex<double>> S_Q(k_vals.size()*n_vecs, 0.0);

    cout << "Computing... \n" <<flush;
    for (int i=0; i<static_cast<int>(atom_list.size()); i++){
        cout<<"Atom i="<<i<<"\r"<<flush;

        auto atom = atom_list[i];
        auto sl = atom.at("sl");
        auto r_arr = atom.at("xyz");
        Eigen::Vector3d r(r_arr[0], r_arr[1], r_arr[2]);

        // Store a 1D vector rep of Sz_i in Sz
        calc_Sz_rep(Sz, basis, i);
        
        for (size_t ik=0; ik<k_vals.size(); ik++){
            auto& k = k_vals[ik];
            auto phi = exp( std::complex(0.,1.) * k.dot(r));
            for (size_t n=0; n<n_vecs; n++){
                S_Q[ik*n_vecs + n] += phi* triple_sum(
                        eigvecs.col(n), Sz, eigvecs.col(0)
                        );
            }
        }
        
    }

    cout << "Done.           \n" <<flush;

    return S_Q;
}





int main(int argc, char* argv[]) {
	argparse::ArgumentParser prog("build_ham");
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

    auto in_datafile=fs::path(prog.get<std::string>("output_file"));

    hid_t in_fid = H5Fopen(in_datafile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    
    // loading the relevant data
    std::vector<double> eigvals = read_vector_h5(in_fid, "eigenvalues");
    Eigen::MatrixXd eigvecs = read_matrix_h5(in_fid, "eigenvectors");
    std::string latfile_name = read_string_from_hdf5(in_fid, "latfile_json");
    std::string dset_name = read_string_from_hdf5(in_fid, "dset_name");
    fs::path latfile_dir(prog.get<std::string>("--latfile_dir"));

    // truncate to the requested # of eigenvectors
    int n_eigvecs = std::min(static_cast<int>(eigvecs.cols()), prog.get<int>("--n_eigvecs"));
    assert(eigvals[0] <= eigvals[1]); // make sure we don't so sth stupid

    fs::path latfile = latfile_dir/latfile_name;
	std::ifstream jfile(latfile);
	if (!jfile) {
		std::cerr << "Failed to open JSON file at " << latfile <<std::endl;
		return 1;
	}
	json jdata;
	jfile >> jdata;

    cout<<"Importing basis... "<<flush;

	ZBasisBST basis;
    // NOTE n_spinons not handled properly
    basis.load_from_file( get_basis_file(latfile, 0, dset_name!="basis"), 
            dset_name
            );
    
    cout<<"Done!"<<endl;

    
//
//    std::map<std::string, Eigen::Vector3d> fcc_kpoints = {
//        {"G", {0.0, 0.0, 0.0}},
//        {"X", {0.5, 0.0, 0.5}},
//        {"W", {0.5, 0.25, 0.75}},
//        {"K", {0.375, 0.375, 0.75}},
//        {"L", {0.5, 0.5, 0.5}},
//        {"U", {0.625, 0.25, 0.625}}
//    };

    
//std::vector<std::pair<std::string, std::string>> fcc_path = {
//    {"G", "X"}, {"X", "W"}, {"W", "K"}, {"K", "G"},
//    {"G", "L"}, {"L", "U"}, {"U", "W"}, {"L", "K"}, {"U", "X"}
//};
    
    std::vector<Eigen::Vector3d> k_vals;
    for (int i=0; i<16; i++){
        k_vals.push_back((1.0*i / 8) * Eigen::Vector3d({0.125, 0.125, 0.125}));
    }


///////////////////////////////////////////////////////////////////////////////
/// Setup done, calculate
    auto out_datafile = in_datafile.replace_extension(".szz.h5");

    // Create a new file using default properties
    hid_t out_fid = H5Fcreate(out_datafile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (out_fid < 0) throw std::runtime_error("Failed to create HDF5 file");


    {
        // remember what the lattice is
        write_string_to_hdf5(out_fid, "latfile_json", 
                latfile.filename() );

        // save the eigvals for convenience (only thise for which we have evectors)
        hsize_t dims[1] = {static_cast<hsize_t>(n_eigvecs)};
        write_dataset(out_fid, "eigenvalues", eigvals.data(), dims, 1);
    }
    auto Szz_intensities =
        compute_scalar_Szz_of_omega(eigvecs, basis, jdata, k_vals);
    {
        // Save raw Szz intensities
        hsize_t inten_dims[2] = {
            static_cast<hsize_t>(k_vals.size()),
            static_cast<hsize_t>(n_eigvecs)
        };
        write_cplx_dataset(out_fid, "Szz_intensities", Szz_intensities.data(), inten_dims, 2);

    }
    cout<<"Saved to\n"<<out_datafile<<endl; 
            
    return 0;
}

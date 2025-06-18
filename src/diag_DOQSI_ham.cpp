#include <argparse/argparse.hpp>


#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "matrix_diag_bits.hpp"
#include "physics/Jring.hpp"

#include "Spectra/Util/SelectionRule.h"
#include "operator.hpp"

#include "expectation_eval.hpp"



using json = nlohmann::json;


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

void build_hamiltonian(SymbolicOpSum<double>& H_sym, 
        const nlohmann::json& jdata, double Jpm, const Vector3d B){

	try {
		auto version=jdata.at("__version__");
		if ( stof(version.get<std::string>()) < 1.1 - 2e-5 ){
			throw std::runtime_error("JSON file is old, API version 1.1 is required");
		}
	} catch (const json::out_of_range& e){
		throw std::runtime_error("__version__ field missing, suspect an old file");
	} 	

    auto g= g_vals(Jpm, B);

    auto atoms = jdata.at("atoms");
    
    auto local_z = get_loc_z();

    auto [ringL, ringR, sl_list]  = get_ring_ops(jdata);

    for (size_t i=0; i<sl_list.size(); i++){
        auto sl = sl_list[i];
        auto R = ringR[i];
        auto L = ringL[i];

        H_sym.add_term(g[sl], R);
        H_sym.add_term(g[sl], L); 
    }
    
    for (const auto& bond : jdata.at("bonds")){
        auto i = bond.at("from_idx").get<int>();
        auto j = bond.at("to_idx").get<int>();


        auto si = std::stoi(jdata.at("atoms")[i].at("sl").get<std::string>());
        auto sj = std::stoi(jdata.at("atoms")[j].at("sl").get<std::string>());
        // int sl = stoi(atoms[i].at("sl").get<std::string>());

		// Convert row of local_z to Eigen::Vector3d
		double zi = B.dot(local_z.row(si));
		double zj = B.dot(local_z.row(sj));

		double zz_coeff = -4.0 * Jpm * zi * zj;

		H_sym.add_term(zz_coeff, SymbolicPMROperator({'z','z'},
                                                     {i, j}));
    }

}


int main(int argc, char* argv[]) {
	argparse::ArgumentParser prog("build_ham");
	prog.add_argument("lattice_file");
	prog.add_argument("-s", "--sector");
	prog.add_argument("--Jpm")
		.help("Jom, units of Jzz")
        .required()
        .scan<'g', double>();
    prog.add_argument("--B")
		.help("magnetic field, units of Jzz")
        .required()
        .nargs(3)
		.scan<'g', double>();

    prog.add_argument("-o", "--output_dir")
        .required()
        .help("output directory");

	prog.add_argument("--ncv", "-k")
		.help("Krylov dimension, shoufl be > 2*n_eigvals")
		.default_value(15)
		.scan<'i', int>();
	prog.add_argument("--n_eigvals", "-n")
		.help("Number of eigenvlaues to compute")
		.default_value(5)
		.scan<'i', int>();

	prog.add_argument("--save_matrix")
		.help("Flag to get the solver to export a rep of the matrix")
		.default_value(false)
		.implicit_value(true);
    prog.add_argument("--max_iters")
        .help("Max steps for iterative solver")
        .default_value(1000)
        .scan<'i', int>();
    prog.add_argument("--tol")
        .help("Tolerance iterative solver")
        .default_value(1e-10)
        .scan<'g', double>();

    prog.add_argument("--algorithm", "-a")
        .choices("dense","sparse","mfsparse")
        .default_value("sparse")
        .help("Variant of ED algorithm to run. dense is best for small problems, mfsparse is a matrix free method that trades off speed for memory.");
		
    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
		std::cerr << err.what() << std::endl;
		std::cerr << prog;
        std::exit(1);
    }

	// Step 1: Load basis from CSV
    std::cout<<"Loading basis..."<<std::endl;
	ZBasis basis;
	basis.load_from_file(get_basis_file(prog));
    std::cout<<"Done! Basis dim="<<basis.dim()<<std::endl;

	// Step 2: Load ring data from JSON
	std::ifstream jfile(prog.get<std::string>("lattice_file"));
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;


	using T=double;
	SymbolicOpSum<T> H_sym;

    auto Bv = prog.get<std::vector<double>>("B");

    Vector3d B;
    for (size_t i=0; i<3; i++) 
        B[i] = Bv[i];

    auto Jpm = prog.get<double>("Jpm");

    char outfilename_buf[1024];
    snprintf(outfilename_buf, 1024, "Jpm=%.4f%%Bx=%.4f%%By=%.4f%%Bz=%.4f%%",
            Jpm, B[0], B[1], B[2]);

    std::stringstream s;
    s << prog.get<std::string>("--output_dir") << "/" <<
        outfilename_buf;

	build_hamiltonian(H_sym, jdata, Jpm, B);

	auto H = LazyOpSum(basis, H_sym);

    if (prog.get<bool>("--save_matrix")) { 
        std::string H_path = s.str() + ".H.mtx";
        Eigen::saveMarket(H.toSparseMatrix(), H_path);
        std::cout << "Saved Hamiltonian to " << H_path << std::endl;
    }

    ////////////////////////////////////////
    // Do the diagonalisation
    auto [eigvals, v] = diagonalise_real(H, prog);

    std::cout << "Eigenvalues:\n" << eigvals << "\n\n";
    std::string filename = s.str()+".eigs.h5";

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

    // Write diag_vals: shape (N, n_eigvals)
    {
        hsize_t dims[2] = {static_cast<hsize_t>(v.rows()), static_cast<hsize_t>(v.cols())};
        write_dataset(file_id, "eigenvectors", v.data(), dims, 2);
    }


    /*
    
    auto [ringL, ringR, sl] = get_ring_ops(jdata);
    Eigen::MatrixXd diag_vals = compute_expectation_values(basis, v, ringL);
    Eigen::MatrixXd cross_vals = compute_cross_terms(basis, v, ringL, 0, 1);

    save_expectation_data_to_hdf5(s.str() + ".out.h5", 
            eigvals, diag_vals, cross_vals, sl);
            */

	return 0;
}


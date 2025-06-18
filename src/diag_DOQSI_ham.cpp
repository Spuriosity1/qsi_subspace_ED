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



void build_hamiltonian(SymbolicOpSum<double>& H_sym, 
        const nlohmann::json& jdata, const std::vector<double>& g){

    auto atoms = jdata.at("atoms");

    auto [ringL, ringR, sl_list]  = get_ring_ops(jdata);

    for (size_t i=0; i<sl_list.size(); i++){
        auto sl = sl_list[i];
        auto R = ringR[i];
        auto L = ringL[i];

        H_sym.add_term(g[sl], R);
        H_sym.add_term(g[sl], L); 
    }
    
}





int main(int argc, char* argv[]) {
	argparse::ArgumentParser prog("build_ham");
	prog.add_argument("lattice_file");
	prog.add_argument("-s", "--sector");

    // G specification
    {
        auto &group = prog.add_mutually_exclusive_group(true);
        group.add_argument("--B")
            .help("magnetic field, units of Jzz")
            .nargs(3)
            .scan<'g', double>();

        group.add_argument("--g")
            .help("raw ring exchange, units of Jzz")
            .nargs(4)
            .scan<'g', double>();


        prog.add_argument("--Jpm")
            .help("Jom, units of Jzz")
            .scan<'g', double>();
    }

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

	prog.add_argument("--n_eigvecs", "-N")
		.help("Number of eigenvectors to store (must be <= n_eigvals)")
		.default_value(4)
		.scan<'i', int>();
	prog.add_argument("--n_spinons")
        .default_value(0)
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


	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;

	// Step 2: Load basis from H5
    std::cout<<"Loading basis..."<<std::endl;
	ZBasis basis;

	if (prog.is_used("--sector")) {
        auto sector = prog.get<std::string>("--sector");
        basis.load_from_file(get_basis_file(lattice_file, prog), sector.c_str());
    } else {
        basis.load_from_file(get_basis_file(lattice_file, prog));
    }
    std::cout<<"Done! Basis dim="<<basis.dim()<<std::endl;



	using T=double;
	SymbolicOpSum<T> H_sym;


    char outfilename_buf[1024];

    std::stringstream s;



    if (prog.is_used("--g")){

        auto gv = prog.get<std::vector<double>>("--g");
        build_hamiltonian(H_sym, jdata, gv);

        snprintf(outfilename_buf, 1024, "g0=%.4f%%g1=%.4f%%g2=%.4f%%g3=%.4f%%",
                gv[0], gv[1], gv[2],gv[3]);

    } else {

        auto Jpm = prog.get<double>("--Jpm");
        auto Bv = prog.get<std::vector<double>>("B");

        Vector3d B;
        for (size_t i=0; i<3; i++) 
            B[i] = Bv[i];

        snprintf(outfilename_buf, 1024, "Jpm=%.4f%%Bx=%.4f%%By=%.4f%%Bz=%.4f%%",
                Jpm, B[0], B[1], B[2]);
        s << prog.get<std::string>("--output_dir") << "/" <<
            outfilename_buf;

        build_hamiltonian(H_sym, jdata, Jpm, B);
    }

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


	std::cout << "Writing to\n"<<filename<<std::endl;

    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) throw std::runtime_error("Failed to create HDF5 file");

    
    // Write eigenvalues: shape (n_eigvals,)
    {
        hsize_t dims[1] = {static_cast<hsize_t>(eigvals.size())};
        write_dataset(file_id, "eigenvalues", eigvals.data(), dims, 1);
    }

    // Write diag_vals: shape (dim, n_eigvecs)
    {
        hsize_t dims[2] = {static_cast<hsize_t>(v.rows()), static_cast<hsize_t>(v.cols())};
        write_dataset(file_id, "eigenvectors", v.data(), dims, 2);
    }

    {
        fs::path latfile = prog.get<std::string>("lattice_file");
        write_string_to_hdf5(file_id, "latfile_json", 
                latfile.filename() );
    }




	return 0;
}


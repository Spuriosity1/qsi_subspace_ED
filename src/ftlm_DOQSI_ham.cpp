#include <argparse/argparse.hpp>


#include <nlohmann/json.hpp>
#include <ostream>
#include "hamiltonain_setup.hpp"

#include "ftlm.hpp"


using json = nlohmann::json;

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

	prog.add_argument("--n_randvecs", "-R")
		.help("Number of random vectors for FTLM evaluation")
		.default_value(20)
		.scan<'i', int>();

	prog.add_argument("--n_temps", "-S")
		.help("Number of temperatures")
		.default_value(100)
		.scan<'i', int>();

    prog.add_argument("--T_min")
        .required()
        .scan<'g', double>();

    prog.add_argument("--T_max")
        .required()
        .scan<'g', double>();

	prog.add_argument("--n_spinons")
        .help("Number of allowed spinons in the basis")
        .default_value(0)
        .scan<'i', int>();

    prog.add_argument("--tol")
        .help("Tolerance iterative solver")
        .default_value(1e-10)
        .scan<'g', double>();
	
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

    load_basis(basis, prog);

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

        build_hamiltonian(H_sym, jdata, Jpm, B);
    }

    s << prog.get<std::string>("--output_dir") << "/" << outfilename_buf;

	auto H = LazyOpSum(basis, H_sym);

    ////////////////////////////////////////
    // run FTLM
    // Set up FTLM parameters
    int R = prog.get<int>("--n_randvecs");
    int ncv = prog.get<int>("--ncv");   // e.g. 100
    int N_T = prog.get<int>("--n_temps");   // e.g. 100
    double T_min = prog.get<double>("--T_min");
    double T_max = prog.get<double>("--T_max");

    // Output message
    std::cout << "Running FTLM: R=" << R << ", ncv=" << ncv << ", Temps=[" << T_min << ", " << T_max << "] with N_T=" << N_T << std::endl;

    // Create temperature grid
    // Allocate result containers
    std::vector<double> Cvs(N_T), Zs(N_T);

    // materialse H
    std::cout << "Materialising H..." << std::endl;
    auto H_sp = H.toSparseMatrix();

    // Run it
    
    using OpType = Eigen::SparseMatrix<double>;


    std::vector<OpType> observables;
    observables.push_back(H_sp);
    observables.push_back(H_sp*H_sp);

    ftlm_computer<OpType> calc(observables, H_sp, T_min, T_max, N_T);
    calc.set_numerical_params(ncv, prog.get<double>("--tol"));

    // 6. Run multiple samples
    std::random_device rd;
    std::mt19937 rng(rd());
    auto num_samples=prog.get<int>("--n_randvecs");
    for (int ns=0; ns<num_samples; ns++){
        std::cout<<"[ftlm] Sample "<<ns+1<<"/"<<num_samples << 
            std::endl;
        calc.evolve(rng);
    }
    std::cout<<std::endl;

    // finite_temperature_lanczos(H, basis.dim(), R, M, Ts, Cvs, Zs);

    std::string filename = s.str() + ".ftlm.h5";
    std::cout << "Writing FTLM output to " << filename << std::endl;
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) throw std::runtime_error("Failed to create HDF5 file");

    // Write temperatures
    {
        hsize_t dims[1] = {static_cast<hsize_t>(N_T)};
        write_dataset(file_id, "beta", calc.get_beta_grid().data(), dims, 1);
        calc.write_to_h5(file_id, 0, "H");
        calc.write_to_h5(file_id, 1, "H*H");
        write_integer(file_id, "num_samples", num_samples);
    }

	return 0;
}


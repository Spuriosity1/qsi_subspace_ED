#include <argparse/argparse.hpp>


#include <nlohmann/json.hpp>
#include "hamiltonian_setup.hpp"


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

	prog.add_argument("--n_spinons")
        .default_value(0)
        .scan<'i', int>();

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
	ZBasisBST basis;

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

    std::cout << "Materialising sparse matrix..." << std::endl;
    auto H_sparsemat = H.toSparseMatrix();

    std::string out = s.str()+".mat.mtx";
    Eigen::saveMarket(H_sparsemat, out);
    std::cout << "Saved to " << out << std::endl;

	return 0;
}


#pragma once

#include <stdexcept>
#include <vector>

#include "matrix_diag_bits.hpp"
#include "physics/Jring.hpp"

#include "operator.hpp"
#include "expectation_eval.hpp"


inline void build_hamiltonian(SymbolicOpSum<double>& H_sym, 
        const nlohmann::json& jdata, double Jpm, const Vector3d B){

	try {
		auto version=jdata.at("__version__");
		if ( stof(version.get<std::string>()) < 1.1 - 2e-5 ){
			throw std::runtime_error("JSON file is old, API version 1.1 is required");
		}
	} catch (const nlohmann::json::out_of_range& e){
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



inline void build_hamiltonian(SymbolicOpSum<double>& H_sym, 
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



inline void load_basis(ZBasis& basis, const argparse::ArgumentParser& prog){
    auto basisfile = get_basis_file(prog.get<std::string>("lattice_file"), 
            prog.get<int>("--n_spinons"),
            prog.is_used("--sector"));

    std::cout <<"Loading from "<<basisfile << std::endl;
	if (prog.is_used("--sector")) {
        auto sector = prog.get<std::string>("--sector");
        basis.load_from_file(basisfile, sector.c_str());
    } else {
        basis.load_from_file(basisfile);
    }
}





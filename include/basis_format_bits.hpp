#pragma once

#include <filesystem>
#include <argparse/argparse.hpp>
#include "operator.hpp"


inline std::filesystem::path get_basis_file(const std::filesystem::path& lattice_file,
        int n_spinons, bool subspace=false){
// Determine basis_file default if not set
	std::string basis_file;

    std::string ext = "." + std::to_string(n_spinons) + ".basis";
	if (subspace) {
        ext += ".partitioned";
    } 
    ext += ".h5";
   
    std::filesystem::path path(lattice_file);
    // Replace extension: json-> ext
    if (path.extension() == ".json") {
        path.replace_extension(ext);
    } else {
        // fallback if extension isn't ".json"
        path += ext;
    }

    return path;
}


template<Basis B>
inline auto load_basis(B& basis, const argparse::ArgumentParser& prog){
    auto basisfile = get_basis_file(prog.get<std::string>("lattice_file"), 
            prog.get<int>("--n_spinons"),
            prog.is_used("--sector"));

	if (prog.is_used("--sector")) {
        auto sector = prog.get<std::string>("--sector");
        return basis.load_from_file(basisfile, sector.c_str());
    } else {
        return basis.load_from_file(basisfile);
    }
}

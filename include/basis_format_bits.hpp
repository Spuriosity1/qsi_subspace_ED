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


inline std::pair<std::string, std::string>
get_basis_args(const argparse::ArgumentParser& prog){
    std::string basisfile;
    if (prog.is_used("--basis_file")) {
        basisfile = prog.get<std::string>("--basis_file");
    } else {
        basisfile = get_basis_file(prog.get<std::string>("lattice_file"),
            prog.get<int>("--n_spinons"),
            prog.is_used("--sector"));
    }
    std::string dataset = prog.is_used("--sector")
        ? prog.get<std::string>("--sector") : "basis";
    return {basisfile, dataset};
}

template<Basis B>
inline auto load_basis(B& basis, const argparse::ArgumentParser& prog){
    auto [basisfile, dataset] = get_basis_args(prog);
    return basis.load_from_file(basisfile, dataset);
}

// Load a raw slab without MPI redistribution.
// Follow with remove_null_states (optional) then basis.redistribute().
template<Basis B>
inline auto load_basis_raw(B& basis, const argparse::ArgumentParser& prog){
    auto [basisfile, dataset] = get_basis_args(prog);
    return basis.load_raw(basisfile, dataset);
}


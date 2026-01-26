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
    std::string basisfile;
    if (prog.is_used("--basis_file")) {
        basisfile = prog.get<std::string>("--basis_file");
    } else {
        basisfile = get_basis_file(prog.get<std::string>("lattice_file"), 
            prog.get<int>("--n_spinons"),
            prog.is_used("--sector"));
    }

	if (prog.is_used("--sector")) {
        auto sector = prog.get<std::string>("--sector");
        return basis.load_from_file(basisfile, sector.c_str());
    } else {
        return basis.load_from_file(basisfile);
    }
}



inline std::string replace_filename(const std::string& input) {
    std::string result = input;

    // Find the last dot before ".basis.h5"
    size_t basis_pos = result.rfind(".basis.h5");
    if (basis_pos == std::string::npos)
        throw std::runtime_error("Mangled basis name: expects .n.basis.csv for some int n");

    // Go backward to find the beginning of the decimal number
    size_t number_end = basis_pos - 1;
    size_t number_start = number_end;
    while (number_start > 0 && std::isdigit(result[number_start - 1]))
        --number_start;

    // Expect a dot before the number
    if (number_start == 0 || result[number_start - 1] != '.')
        throw std::runtime_error("Mangled basis name: expects .n.basis.csv for some int n");

    // Replace the entire ".NNN.basis.h5" with ".json"
    result.replace(number_start - 1, basis_pos - number_start + 10, ".json");
    return result;
}

#include <queue>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "argparse/argparse.hpp"
#include "bittools.hpp"
#include <algorithm>
#include <iostream>
#include "sort.hpp"


using json = nlohmann::json;
static const size_t GIGA= 1 <<30;
static const size_t MEGA= 1 <<20;


int main(int argc, char* argv[]) {
    // Initialize MPI
    argparse::ArgumentParser prog("merge_shards");
    prog.add_argument("manifest")
        .required()
        .help("Manifest JSON file produced by shard writer");
    prog.add_argument("--output", "-o")
        .required()
        .help("Output HDF5 filename (without .h5)");
    prog.add_argument("--batch_size", "-b")
        .default_value(MEGA)
        .scan<'i', size_t>()
        .help("Batch size for buffered merging");

    prog.add_argument("--chunk_size", "-c")
        .default_value(MEGA )
        .scan<'i', size_t>()
        .help("chunk size");

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << prog;
        
        return 1;
    }
    
    auto manifest_file = prog.get<std::string>("manifest");
    
    // Load manifest (all ranks read it)
    std::ifstream ifs(manifest_file);
    if (!ifs) {
        std::cerr << "Failed to open manifest file\n";
        return 1;
    }
    json manifest = json::parse(ifs);
    std::vector<std::string> shards = manifest["shards"].get<std::vector<std::string>>();
    
    std::string outfilename = prog.get<std::string>("--output");
    size_t batch_size = prog.get<size_t>("--batch_size");
    size_t chunk_size = prog.get<size_t>("--chunk_size");
    
    try {
        auto [file_id, dataset_id] = create_hdf5_dataset(
                outfilename, batch_size, chunk_size,
                0 // compression level, gzip
                );
        
        std::cout << "K-way merge.\n";
        size_t total_merged = merge_to_hdf5(shards, dataset_id, batch_size);
        H5Dclose(dataset_id);
        std::cout << "K-way merge complete. Total elements: " << total_merged << "\n";
        H5Fclose(file_id);
    } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

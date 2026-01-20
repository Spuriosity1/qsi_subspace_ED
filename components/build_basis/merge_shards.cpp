#include <mpi.h>
#include <queue>
#include <fstream>
#include <filesystem>
#include <hdf5.h>
#include <nlohmann/json.hpp>
#include "argparse/argparse.hpp"
#include "bittools.hpp"
#include <algorithm>
#include <iostream>
#include "sort.hpp"
//#include <numeric>

using json = nlohmann::json;

// Parallel sort of shard files using MPI
void parallel_sort_shards(const std::vector<std::string>& shard_files,
                          size_t memory_limit,
                          bool force_external_sort,
                          int rank,
                          int size) {
    // Each process sorts a subset of shards using round-robin distribution
    for (size_t i = rank; i < shard_files.size(); i += size) {
        if (rank == 0 || i == static_cast<size_t>(rank)) {
            std::cout << "Rank " << rank << " sorting shard " << i 
                      << ": " << shard_files[i] << "\n";
        }
        sort_shard_file(shard_files[i], memory_limit, force_external_sort);
    }
}

// MPI-parallel external merge sort across shard files -> HDF5
void external_mergesort_mpi(const std::vector<std::string> &shard_files,
                            const std::string &outfilename,
                            size_t batch_size = 1 << 28,
                            size_t memory_limit = 1 << 30,
                            bool force_external_sort = false) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // STEP 1: Distribute shard sorting across MPI processes
    if (rank == 0) {
        std::cout << "Step 1: Sorting individual shard files in parallel...\n";
        std::cout << "Using " << size << " MPI processes\n";
    }
    
    parallel_sort_shards(shard_files, memory_limit, force_external_sort, rank, size);
    
    // Synchronize all processes after sorting
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Individual shard sorting complete.\n\n";
        std::cout << "Step 2: Performing k-way merge on rank 0...\n";
    }
    
    // STEP 2: Only rank 0 performs the final merge
    if (rank == 0) {
        auto [file_id, dataset_id] = create_hdf5_dataset(outfilename, batch_size);
        size_t total_merged = merge_to_hdf5(shard_files, dataset_id, batch_size);
        H5Dclose(dataset_id);
        std::cout << "K-way merge complete. Total elements: " << total_merged << "\n";
        H5Fclose(file_id);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    argparse::ArgumentParser prog("merge_shards");
    prog.add_argument("manifest")
        .required()
        .help("Manifest JSON file produced by shard writer");
    prog.add_argument("--output", "-o")
        .required()
        .help("Output HDF5 filename (without .h5)");
    prog.add_argument("--batch_size", "-b")
        .default_value(static_cast<size_t>(1 << 26))
        .scan<'i', size_t>()
        .help("Batch size for buffered merging");

    prog.add_argument("--memory_limit_per_task", "-m")
        .default_value(static_cast<size_t>(1 << 30))
        .scan<'i', size_t>()
        .help("Batch size for buffered merging");
    
    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err) {
        if (rank == 0) {
            std::cerr << err.what() << std::endl;
            std::cerr << prog;
        }
        MPI_Finalize();
        return 1;
    }
    
    auto manifest_file = prog.get<std::string>("manifest");
    
    // Load manifest (all ranks read it)
    std::ifstream ifs(manifest_file);
    if (!ifs) {
        if (rank == 0) {
            std::cerr << "Failed to open manifest file\n";
        }
        MPI_Finalize();
        return 1;
    }
    json manifest = json::parse(ifs);
    std::vector<std::string> shards = manifest["shards"].get<std::vector<std::string>>();
    
    std::string outfilename = prog.get<std::string>("--output");
    size_t batch_size = prog.get<size_t>("--batch_size");
    size_t memory_limit = prog.get<size_t>("--memory_limit_per_task");
    
    try {
        external_mergesort_mpi(shards, outfilename, batch_size, memory_limit);
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    MPI_Finalize();
    return 0;
}

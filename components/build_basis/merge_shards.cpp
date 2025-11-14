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
//#include <numeric>

using json = nlohmann::json;

// Read a batch of Uint128s from a file
static size_t read_batch(std::ifstream& in,
                         std::vector<Uint128>& buffer,
                         size_t batch_size) {
    buffer.resize(batch_size);
    in.read(reinterpret_cast<char*>(buffer.data()),
            batch_size * sizeof(Uint128));
    size_t got = in.gcount() / sizeof(Uint128);
    buffer.resize(got);
    return got;
}

// Heap item for k-way merge
struct HeapItem {
    Uint128 val;
    size_t file_idx;
};

// Comparator for min-heap
auto heap_comparator = [](const HeapItem &a, const HeapItem &b) {
    return a.val > b.val;
};

using MergeHeap = std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(heap_comparator)>;

// Open all shard files and return streams
std::vector<std::ifstream> open_shard_files(const std::vector<std::string>& shard_files) {
    std::vector<std::ifstream> streams;
    streams.reserve(shard_files.size());
    
    for (const auto &fname : shard_files) {
        streams.emplace_back(fname, std::ios::binary);
        if (!streams.back()) {
            throw std::runtime_error("Failed to open " + fname);
        }
    }
    
    return streams;
}

// Initialize heap with first element from each file
void prime_merge_heap(MergeHeap& heap,
                      std::vector<std::ifstream>& streams,
                      std::vector<std::vector<Uint128>>& buffers,
                      std::vector<size_t>& positions,
                      size_t batch_size) {
    for (size_t i = 0; i < streams.size(); ++i) {
        if (read_batch(streams[i], buffers[i], batch_size) > 0) {
            positions[i] = 0;
            heap.push({buffers[i][0], i});
        }
    }
}

// Advance to next element in a file's buffer
bool advance_file_buffer(size_t file_idx,
                         MergeHeap& heap,
                         std::vector<std::ifstream>& streams,
                         std::vector<std::vector<Uint128>>& buffers,
                         std::vector<size_t>& positions,
                         size_t batch_size) {
    positions[file_idx]++;
    
    if (positions[file_idx] < buffers[file_idx].size()) {
        heap.push({buffers[file_idx][positions[file_idx]], file_idx});
        return true;
    } else {
        // Buffer exhausted, try to refill
        if (read_batch(streams[file_idx], buffers[file_idx], batch_size) > 0) {
            positions[file_idx] = 0;
            heap.push({buffers[file_idx][0], file_idx});
            return true;
        }
    }
    
    return false; // No more data from this file
}

// Write buffer to binary output file
void flush_binary_buffer(std::ofstream& output,
                         std::vector<Uint128>& out_buffer) {
    if (!out_buffer.empty()) {
        output.write(reinterpret_cast<const char*>(out_buffer.data()),
                     out_buffer.size() * sizeof(Uint128));
        out_buffer.clear();
    }
}

// K-way merge temp files directly to output file
void external_mergesort_to_file(const std::vector<std::string>& shard_files,
                                const std::string& output_filename,
                                size_t batch_size = 1 << 16) {
    MergeHeap heap(heap_comparator);
    
    // Open all shard files
    auto streams = open_shard_files(shard_files);
    
    // Buffers per stream
    std::vector<std::vector<Uint128>> buffers(shard_files.size());
    std::vector<size_t> positions(shard_files.size(), 0);
    
    // Prime heap
    prime_merge_heap(heap, streams, buffers, positions, batch_size);
    
    // Open output file
    std::ofstream output(output_filename, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to create output file: " + output_filename);
    }
    
    std::vector<Uint128> out_buffer;
    out_buffer.reserve(batch_size);
    
    // Main merge loop
    while (!heap.empty()) {
        auto [val, fidx] = heap.top();
        heap.pop();
        
        out_buffer.push_back(val);
        
        // Advance to next element
        advance_file_buffer(fidx, heap, streams, buffers, positions, batch_size);
        
        // Write output buffer when full
        if (out_buffer.size() >= batch_size || heap.empty()) {
            flush_binary_buffer(output, out_buffer);
        }
    }
    
    output.close();
}

// In-memory sort of a file
void sort_file_in_memory(const std::string& filename, size_t num_elements) {
    size_t file_size = num_elements * sizeof(Uint128);
    std::vector<Uint128> data(num_elements);
    
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data.data()), file_size);
    in.close();
    
    std::sort(data.begin(), data.end());
    
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), file_size);
    out.close();
    
    std::cout << "Sorted " << filename << " (in-memory, " << num_elements << " elements)\n";
}

// External sort for large files
void sort_file_external(const std::string& filename,
                        size_t num_elements,
                        size_t elements_per_chunk) {
    namespace fs = std::filesystem;
    
    std::cout << "Sorting " << filename << " (external sort, " << num_elements 
              << " elements in " << ((num_elements + elements_per_chunk - 1) / elements_per_chunk) 
              << " chunks)\n";
    
    std::vector<std::string> temp_files;
    std::ifstream in(filename, std::ios::binary);
    
    // Phase 1: Sort chunks and write to temp files
    size_t chunk_num = 0;
    std::vector<Uint128> chunk;
    
    while (read_batch(in, chunk, elements_per_chunk) > 0) {
        std::sort(chunk.begin(), chunk.end());
        
        std::string temp_name = filename + ".tmp." + std::to_string(chunk_num++);
        temp_files.push_back(temp_name);
        
        std::ofstream temp_out(temp_name, std::ios::binary);
        temp_out.write(reinterpret_cast<const char*>(chunk.data()), 
                      chunk.size() * sizeof(Uint128));
        temp_out.close();
    }
    in.close();
    
    // Phase 2: K-way merge the temp files back to original file
    external_mergesort_to_file(temp_files, filename);
    
    // Clean up temp files
    for (const auto& temp : temp_files) {
        fs::remove(temp);
    }
}

// Sort a single shard file in-place using external sort if needed
void sort_shard_file(const std::string& filename,
                     size_t memory_limit = 1 << 20,
                     bool force_multi_thread = false) {
    // Get file size
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file for sorting: " + filename);
    }
    
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    file.close();
    
    size_t num_elements = file_size / sizeof(Uint128);
    size_t elements_per_chunk = memory_limit / sizeof(Uint128);
    
    if (num_elements <= elements_per_chunk && !force_multi_thread) {
        sort_file_in_memory(filename, num_elements);
    } else {
        sort_file_external(filename, num_elements, elements_per_chunk);
    }
}

// Print HDF5 error stack and throw
inline void h5_fail(const char* msg) {
    // Prints entire error stack to stderr
    H5Eprint2(H5E_DEFAULT, stderr);
    throw std::runtime_error(msg);
}

// Check an HDF5 return ID
inline void h5_expect_id(hid_t id, const char* msg) {
    if (id < 0) h5_fail(msg);
}

// Check an HDF5 return status (herr_t)
inline void h5_expect_ok(herr_t status, const char* msg) {
    if (status < 0) h5_fail(msg);
}


// Create HDF5 dataset with unlimited first dimension
hid_t create_hdf5_dataset(const std::string& filename, size_t batch_size) {
    hsize_t dims[2]       = {0, 2};
    hsize_t maxdims[2]    = {H5S_UNLIMITED, 2};
    hsize_t chunk_dims[2] = {batch_size, 2};
    
   std::string out_file_with_ext = filename + ".h5";

    std::cout<<"Creating file: "<<out_file_with_ext<<std::endl;
    hid_t file_id = H5Fcreate(out_file_with_ext.c_str(),
                              H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    h5_expect_id(file_id, "File creation");
    
    hid_t dataspace_id = H5Screate_simple(2, dims, maxdims);
    h5_expect_id(dataspace_id, "dataspace creation");

    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    h5_expect_id(plist_id, "plist creation");


    h5_expect_ok(
            H5Pset_chunk(plist_id, 2, chunk_dims),
            "H5Pset_chunk failed");

    hid_t dataset_id = H5Dcreate(file_id, "basis", H5T_NATIVE_UINT64,
                                 dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    h5_expect_id(dataset_id, "dataset creation");

    H5Pclose(plist_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    
    return dataset_id;
}

// Write buffer to HDF5 dataset
void flush_hdf5_buffer(hid_t dataset_id,
                       std::vector<Uint128>& out_buffer,
                       hsize_t& offset) {
    if (out_buffer.empty()) return;
    
    hsize_t new_dims[2] = {offset + out_buffer.size(), 2};
    H5Dset_extent(dataset_id, new_dims);
    
    hid_t filespace = H5Dget_space(dataset_id);
    hsize_t start[2] = {offset, 0};
    hsize_t count[2] = {out_buffer.size(), 2};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
    
    hid_t memspace = H5Screate_simple(2, count, nullptr);
    H5Dwrite(dataset_id, H5T_NATIVE_UINT64, memspace, filespace,
             H5P_DEFAULT, out_buffer.data());
    
    H5Sclose(memspace);
    H5Sclose(filespace);
    
    offset += out_buffer.size();
    out_buffer.clear();
}

// Perform k-way merge and write to HDF5
size_t merge_to_hdf5(const std::vector<std::string>& shard_files,
                     hid_t dataset_id,
                     size_t batch_size) {
    MergeHeap heap(heap_comparator);
    
    // Open all shard files
    auto streams = open_shard_files(shard_files);
    
    // Buffers per stream
    std::vector<std::vector<Uint128>> buffers(shard_files.size());
    std::vector<size_t> positions(shard_files.size(), 0);
    
    // Prime heap
    prime_merge_heap(heap, streams, buffers, positions, batch_size);
    
    std::vector<Uint128> out_buffer;
    out_buffer.reserve(batch_size);
    hsize_t offset = 0;
    
    size_t total_merged = 0;
    
    // Main merge loop
    while (!heap.empty()) {
        auto [val, fidx] = heap.top();
        heap.pop();
        
        out_buffer.push_back(val);
        total_merged++;
        
        // Advance to next element
        advance_file_buffer(fidx, heap, streams, buffers, positions, batch_size);
        
        // Write out_buffer in batches
        if (out_buffer.size() >= batch_size || heap.empty()) {
            flush_hdf5_buffer(dataset_id, out_buffer, offset);
        }
    }
    
    return total_merged;
}

// Parallel sort of shard files using MPI
void parallel_sort_shards(const std::vector<std::string>& shard_files,
                          size_t memory_limit,
                          bool force_external_sort,
                          int rank,
                          int size) {
    // Each process sorts a subset of shards using round-robin distribution
    for (size_t i = rank; i < shard_files.size(); i += size) {
        if (rank == 0 || i == rank) {
            std::cout << "Rank " << rank << " sorting shard " << i 
                      << ": " << shard_files[i] << "\n";
        }
        sort_shard_file(shard_files[i], memory_limit, force_external_sort);
    }
}

// MPI-parallel external merge sort across shard files -> HDF5
void external_mergesort_mpi(const std::vector<std::string> &shard_files,
                            const std::string &outfilename,
                            size_t batch_size = 1 << 16,
                            bool force_external_sort = false) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    static const size_t memory_limit = 1 << 20;
    
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
        hid_t dataset_id = create_hdf5_dataset(outfilename, batch_size);
        size_t total_merged = merge_to_hdf5(shard_files, dataset_id, batch_size);
        H5Dclose(dataset_id);
        
        std::cout << "K-way merge complete. Total elements: " << total_merged << "\n";
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
        .default_value(static_cast<size_t>(1 << 16))
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
    
    try {
        external_mergesort_mpi(shards, outfilename, batch_size);
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

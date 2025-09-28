#include <queue>
#include <fstream>
#include <filesystem>
#include <hdf5.h>
#include <nlohmann/json.hpp>
#include "argparse/argparse.hpp"
#include "bittools.hpp"
#include <algorithm>
#include <iostream>

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



// Sort a single shard file in-place using external sort if needed
void sort_shard_file(const std::string& filename, size_t memory_limit = 1 << 20) {
    namespace fs = std::filesystem;
    
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
    
    if (num_elements <= elements_per_chunk) {
        // File fits in memory - simple in-memory sort
        std::vector<Uint128> data(num_elements);
        
        std::ifstream in(filename, std::ios::binary);
        in.read(reinterpret_cast<char*>(data.data()), file_size);
        in.close();
        
        std::sort(data.begin(), data.end());
        
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), file_size);
        out.close();
        
        std::cout << "Sorted " << filename << " (in-memory, " << num_elements << " elements)\n";
    } else {
        throw std::runtime_error("Not Implemented");
        // File too large - external sort
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
        //external_mergesort_to_file(temp_files, filename);
        
        // Clean up temp files
        for (const auto& temp : temp_files) {
            fs::remove(temp);
        }
    }
}

// K-way merge temp files directly to output file
void external_mergesort_to_file(const std::vector<std::string>& shard_files,
                                const std::string& output_filename,
                                size_t batch_size = 1 << 16) {
    struct HeapItem {
        Uint128 val;
        size_t file_idx;
    };
    
    auto cmp = [](const HeapItem &a, const HeapItem &b) {
        return a.val > b.val; // min-heap
    };
    std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(cmp)> heap(cmp);
    
    // Open all shard files
    std::vector<std::ifstream> streams;
    streams.reserve(shard_files.size());
    for (const auto &fname : shard_files) {
        streams.emplace_back(fname, std::ios::binary);
        if (!streams.back()) {
            throw std::runtime_error("Failed to open " + fname);
        }
    }
    
    // Buffers per stream
    std::vector<std::vector<Uint128>> buffers(shard_files.size());
    std::vector<size_t> positions(shard_files.size(), 0);
    
    // Prime heap
    for (size_t i = 0; i < streams.size(); ++i) {
        if (read_batch(streams[i], buffers[i], batch_size) > 0) {
            positions[i] = 0;
            heap.push({buffers[i][0], i});
        }
    }
    
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
        
        // Advance position in the current file's buffer
        positions[fidx]++;
        
        // Check if we need more data from this file
        if (positions[fidx] < buffers[fidx].size()) {
            heap.push({buffers[fidx][positions[fidx]], fidx});
        } else {
            // Buffer exhausted, try to refill
            if (read_batch(streams[fidx], buffers[fidx], batch_size) > 0) {
                positions[fidx] = 0;
                heap.push({buffers[fidx][0], fidx});
            }
        }
        
        // Write output buffer when full
        if (out_buffer.size() >= batch_size || heap.empty()) {
            if (!out_buffer.empty()) {
                output.write(reinterpret_cast<const char*>(out_buffer.data()),
                           out_buffer.size() * sizeof(Uint128));
                out_buffer.clear();
            }
        }
    }
    
    output.close();
}

// External merge sort across shard files -> HDF5
void external_mergesort(const std::vector<std::string> &shard_files,
                        const std::string &outfilename,
                        size_t batch_size = 1 << 16) {
    namespace fs = std::filesystem;
    
    // STEP 1: Sort each shard file individually
    std::cout << "Step 1: Sorting individual shard files...\n";
    for (const auto& shard : shard_files) {
        sort_shard_file(shard);
    }
    std::cout << "Individual shard sorting complete.\n\n";
    
    // STEP 2: K-way merge the now-sorted shard files
    std::cout << "Step 2: Performing k-way merge...\n";
    
    struct HeapItem {
        Uint128 val;
        size_t file_idx;
    };
    
    auto cmp = [](const HeapItem &a, const HeapItem &b) {
        return a.val > b.val; // min-heap
    };
    std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(cmp)> heap(cmp);
    
    // Open all shard files
    std::vector<std::ifstream> streams;
    streams.reserve(shard_files.size());
    for (const auto &fname : shard_files) {
        streams.emplace_back(fname, std::ios::binary);
        if (!streams.back()) {
            throw std::runtime_error("Failed to open " + fname);
        }
    }
    
    // Buffers per stream
    std::vector<std::vector<Uint128>> buffers(shard_files.size());
    std::vector<size_t> positions(shard_files.size(), 0);
    
    // Prime heap
    for (size_t i = 0; i < streams.size(); ++i) {
        if (read_batch(streams[i], buffers[i], batch_size) > 0) {
            positions[i] = 0;
            heap.push({buffers[i][0], i});
        }
    }
    
    // Prepare HDF5 file with unlimited first dimension
    hsize_t dims[2]       = {0, 2};
    hsize_t maxdims[2]    = {H5S_UNLIMITED, 2};
    hsize_t chunk_dims[2] = {batch_size, 2};
    hid_t file_id = H5Fcreate((outfilename + ".h5").c_str(),
                              H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataspace_id = H5Screate_simple(2, dims, maxdims);
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, 2, chunk_dims);
    hid_t dataset_id = H5Dcreate(file_id, "basis", H5T_NATIVE_UINT64,
                                 dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Pclose(plist_id);
    H5Sclose(dataspace_id);
    
    std::vector<Uint128> out_buffer;
    out_buffer.reserve(batch_size);
    hsize_t offset[2] = {0, 0};
    
    size_t total_merged = 0;
    
    // Main merge loop
    while (!heap.empty()) {
        auto [val, fidx] = heap.top();
        heap.pop();
        
        out_buffer.push_back(val);
        total_merged++;
        
        // Advance position in the current file's buffer
        positions[fidx]++;
        
        // Check if we need more data from this file
        if (positions[fidx] < buffers[fidx].size()) {
            heap.push({buffers[fidx][positions[fidx]], fidx});
        } else {
            // Buffer exhausted, try to refill
            if (read_batch(streams[fidx], buffers[fidx], batch_size) > 0) {
                positions[fidx] = 0;
                heap.push({buffers[fidx][0], fidx});
            }
        }
        
        // Write out_buffer in batches
        if (out_buffer.size() >= batch_size || heap.empty()) {
            if (!out_buffer.empty()) {
                hsize_t new_dims[2] = {offset[0] + out_buffer.size(), 2};
                H5Dset_extent(dataset_id, new_dims);
                hid_t filespace = H5Dget_space(dataset_id);
                hsize_t start[2] = {offset[0], 0};
                hsize_t count[2] = {out_buffer.size(), 2};
                H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
                hid_t memspace = H5Screate_simple(2, count, nullptr);
                H5Dwrite(dataset_id, H5T_NATIVE_UINT64, memspace, filespace,
                         H5P_DEFAULT, out_buffer.data());
                H5Sclose(memspace);
                H5Sclose(filespace);
                offset[0] += out_buffer.size();
                out_buffer.clear();
            }
        }
    }
    
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    
    std::cout << "K-way merge complete. Total elements: " << total_merged << "\n";
    
    // Remove shard files (optional)
    // for (auto &f : shard_files) fs::remove(f);
}

int main(int argc, char* argv[]) {
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
        std::cerr << err.what() << std::endl;
        std::cerr << prog;
        return 1;
    }

    auto manifest_file = prog.get<std::string>("manifest");

    // Load manifest
    std::ifstream ifs(manifest_file);
    if (!ifs) {
        std::cerr << "Failed to open manifest file\n";
        return 1;
    }
    json manifest = json::parse(ifs);

    std::vector<std::string> shards = manifest["shards"].get<std::vector<std::string>>();
    std::string outfilename = prog.get<std::string>("--output");
    size_t batch_size = prog.get<size_t>("--batch_size");

    external_mergesort(shards, outfilename, batch_size);
    return 0;
}

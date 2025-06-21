#include <iostream>
#include <basis_io.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>
#include "bittools.hpp"
#include <unordered_map>
#include <sstream>
#include <algorithm>

using namespace basis_io;

inline void partition_basis_hdf5(const std::string& infile, const std::array<Uint128, 4>& sl_mask) {
    // HDF5 identifiers for input file
    hid_t input_file_id = -1, input_dataset_id = -1, input_dataspace_id = -1;
    hid_t output_file_id = -1;
    
    // Maps to store output datasets and their current sizes
    std::unordered_map<std::string, hid_t> output_datasets;
    std::unordered_map<std::string, hsize_t> dataset_sizes;
    std::unordered_map<std::string, std::vector<Uint128>> sector_buffers;
    
    const hsize_t CHUNK_SIZE = 10000; // Process 10k entries at a time
    const hsize_t BUFFER_SIZE = 1000; // Buffer 1k entries per sector before writing
    
    herr_t status;
    
    try {
        // Open input file
        input_file_id = H5Fopen(infile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (input_file_id < 0) throw HDF5Error(input_file_id, -1, -1, "partition_basis: Failed to open input file");
        
        // Open input dataset
        input_dataset_id = H5Dopen(input_file_id, "basis", H5P_DEFAULT);
        if (input_dataset_id < 0) throw HDF5Error(input_file_id, -1, input_dataset_id, "partition_basis: Failed to open input dataset");
        
        // Get input dataspace
        input_dataspace_id = H5Dget_space(input_dataset_id);
        if (input_dataspace_id < 0) throw HDF5Error(input_file_id, input_dataspace_id, input_dataset_id, "partition_basis: Failed to get input dataspace");
        
        // Get dimensions
        int ndims = H5Sget_simple_extent_ndims(input_dataspace_id);
        if (ndims != 2) throw HDF5Error(input_file_id, input_dataspace_id, input_dataset_id, "partition_basis: Expected 2D data");
        
        hsize_t dims[2];
        status = H5Sget_simple_extent_dims(input_dataspace_id, dims, nullptr);
        if (status < 0) throw HDF5Error(input_file_id, input_dataspace_id, input_dataset_id, "partition_basis: Failed to get dimensions");
        
        // Create output file
        std::string outfile = infile;
        size_t dot_pos = outfile.find_last_of('.');
        if (dot_pos != std::string::npos) {
            outfile = outfile.substr(0, dot_pos) + ".partitioned.h5";
        } else {
            outfile += ".partitioned.h5";
        }
        
        output_file_id = H5Fcreate(outfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (output_file_id < 0) throw HDF5Error(output_file_id, -1, -1, "partition_basis: Failed to create output file");
        
        // Process data in chunks
        std::vector<Uint128> chunk_buffer(CHUNK_SIZE);
        hsize_t total_rows = dims[0];
        
        for (hsize_t start_row = 0; start_row < total_rows; start_row += CHUNK_SIZE) {
            printf("[ part ] row %12ld / %12ld (%.0f %%) \r", start_row, total_rows, 100.0*start_row/total_rows);
            hsize_t current_chunk_size = std::min(CHUNK_SIZE, total_rows - start_row);
            
            // Define hyperslab for current chunk
            hsize_t offset[2] = {start_row, 0};
            hsize_t count[2] = {current_chunk_size, dims[1]};
            
            status = H5Sselect_hyperslab(input_dataspace_id, H5S_SELECT_SET, offset, nullptr, count, nullptr);
            if (status < 0) throw HDF5Error(input_file_id, input_dataspace_id, input_dataset_id, "partition_basis: Failed to select hyperslab");
            
            // Create memory dataspace for chunk
            hid_t mem_dataspace_id = H5Screate_simple(2, count, nullptr);
            if (mem_dataspace_id < 0) throw HDF5Error(input_file_id, input_dataspace_id, input_dataset_id, "partition_basis: Failed to create memory dataspace");
            
            // Read chunk
            status = H5Dread(input_dataset_id, H5T_NATIVE_UINT64, mem_dataspace_id, input_dataspace_id, H5P_DEFAULT, chunk_buffer.data());
            if (status < 0) {
                H5Sclose(mem_dataspace_id);
                throw HDF5Error(input_file_id, input_dataspace_id, input_dataset_id, "partition_basis: Failed to read chunk");
            }
            
            H5Sclose(mem_dataspace_id);
            
            // Process each entry in the chunk
            for (hsize_t i = 0; i < current_chunk_size; ++i) {
                // Calculate sector for this basis state
                std::vector<int> sector(4);
                for (int j = 0; j < 4; ++j) {
                    Uint128 masked = sl_mask[j] & chunk_buffer[i];
                    sector[j] = popcnt_u128(masked);
                }
                
                std::string sector_name = make_sector_string(sector);
                
                // Add to sector buffer
                sector_buffers[sector_name].push_back(chunk_buffer[i]);
                
                // If buffer is full, flush to disk
                if (sector_buffers[sector_name].size() >= BUFFER_SIZE) {
                    // Create dataset if it doesn't exist
                    if (output_datasets.find(sector_name) == output_datasets.end()) {
                        // Create extensible dataset
                        hsize_t initial_dims[2] = {0, dims[1]};
                        hsize_t max_dims[2] = {H5S_UNLIMITED, dims[1]};
                        hsize_t chunk_dims[2] = {BUFFER_SIZE, dims[1]};
                        
                        hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
                        H5Pset_chunk(plist_id, 2, chunk_dims);
                        
                        hid_t file_dataspace_id = H5Screate_simple(2, initial_dims, max_dims);
                        hid_t dataset_id = H5Dcreate2(output_file_id, sector_name.c_str(), H5T_NATIVE_UINT64, 
                                                     file_dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
                        
                        H5Sclose(file_dataspace_id);
                        H5Pclose(plist_id);
                        
                        if (dataset_id < 0) throw HDF5Error(output_file_id, -1, dataset_id, "partition_basis: Failed to create output dataset");
                        
                        output_datasets[sector_name] = dataset_id;
                        dataset_sizes[sector_name] = 0;
                    }
                    
                    // Extend dataset
                    hsize_t new_size[2] = {dataset_sizes[sector_name] + sector_buffers[sector_name].size(), dims[1]};
                    status = H5Dset_extent(output_datasets[sector_name], new_size);
                    if (status < 0) throw HDF5Error(output_file_id, -1, output_datasets[sector_name], "partition_basis: Failed to extend dataset");
                    
                    // Write buffer to dataset
                    hid_t file_space = H5Dget_space(output_datasets[sector_name]);
                    hsize_t start[2] = {dataset_sizes[sector_name], 0};
                    hsize_t count[2] = {sector_buffers[sector_name].size(), dims[1]};
                    
                    status = H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr);
                    if (status < 0) {
                        H5Sclose(file_space);
                        throw HDF5Error(output_file_id, file_space, output_datasets[sector_name], "partition_basis: Failed to select output hyperslab");
                    }
                    
                    hid_t mem_space = H5Screate_simple(2, count, nullptr);
                    status = H5Dwrite(output_datasets[sector_name], H5T_NATIVE_UINT64, mem_space, file_space, 
                                     H5P_DEFAULT, sector_buffers[sector_name].data());
                    
                    H5Sclose(file_space);
                    H5Sclose(mem_space);
                    
                    if (status < 0) throw HDF5Error(output_file_id, -1, output_datasets[sector_name], "partition_basis: Failed to write buffer");
                    
                    dataset_sizes[sector_name] += sector_buffers[sector_name].size();
                    sector_buffers[sector_name].clear();
                }
            }
        }
        
        // Flush remaining buffers
        for (auto& [sector_name, buffer] : sector_buffers) {
            if (!buffer.empty()) {
                // Create dataset if it doesn't exist
                if (output_datasets.find(sector_name) == output_datasets.end()) {
                    hsize_t initial_dims[2] = {0, dims[1]};
                    hsize_t max_dims[2] = {H5S_UNLIMITED, dims[1]};
                    hsize_t chunk_dims[2] = {std::min(BUFFER_SIZE, 
                            static_cast<hsize_t>(buffer.size())), dims[1]};
                    
                    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
                    H5Pset_chunk(plist_id, 2, chunk_dims);
                    
                    hid_t file_dataspace_id = H5Screate_simple(2, initial_dims, max_dims);
                    hid_t dataset_id = H5Dcreate2(output_file_id, sector_name.c_str(), H5T_NATIVE_UINT64, 
                                                 file_dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
                    
                    H5Sclose(file_dataspace_id);
                    H5Pclose(plist_id);
                    
                    if (dataset_id < 0) throw HDF5Error(output_file_id, -1, dataset_id, "partition_basis: Failed to create final output dataset");
                    
                    output_datasets[sector_name] = dataset_id;
                    dataset_sizes[sector_name] = 0;
                }
                
                // Extend and write final buffer
                hsize_t new_size[2] = {dataset_sizes[sector_name] + buffer.size(), dims[1]};
                status = H5Dset_extent(output_datasets[sector_name], new_size);
                if (status < 0) throw HDF5Error(output_file_id, -1, output_datasets[sector_name], "partition_basis: Failed to extend final dataset");
                
                hid_t file_space = H5Dget_space(output_datasets[sector_name]);
                hsize_t start[2] = {dataset_sizes[sector_name], 0};
                hsize_t count[2] = {buffer.size(), dims[1]};
                
                status = H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr);
                if (status < 0) {
                    H5Sclose(file_space);
                    throw HDF5Error(output_file_id, file_space, output_datasets[sector_name], "partition_basis: Failed to select final output hyperslab");
                }
                
                hid_t mem_space = H5Screate_simple(2, count, nullptr);
                status = H5Dwrite(output_datasets[sector_name], H5T_NATIVE_UINT64, mem_space, file_space, 
                                 H5P_DEFAULT, buffer.data());
                
                H5Sclose(file_space);
                H5Sclose(mem_space);
                
                if (status < 0) throw HDF5Error(output_file_id, -1, output_datasets[sector_name], "partition_basis: Failed to write final buffer");
            }
        }
        
        // Clean up
        for (auto& [name, dataset_id] : output_datasets) {
            H5Dclose(dataset_id);
        }
        H5Sclose(input_dataspace_id);
        H5Dclose(input_dataset_id);
        H5Fclose(input_file_id);
        H5Fclose(output_file_id);
        
    } catch (const std::exception& e) {
        // Clean up in case of error
        for (auto& [name, dataset_id] : output_datasets) {
            if (dataset_id >= 0) H5Dclose(dataset_id);
        }
        if (input_dataspace_id >= 0) H5Sclose(input_dataspace_id);
        if (input_dataset_id >= 0) H5Dclose(input_dataset_id);
        if (input_file_id >= 0) H5Fclose(input_file_id);
        if (output_file_id >= 0) H5Fclose(output_file_id);
        throw; // Rethrow the exception
    }
}


using json=nlohmann::json;

std::string replace_filename(const std::string& input) {
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

int main (int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: "<<std::string(argv[0])<<" <latfile_stem.N.basis.h5>\n";
        return 1;
    }

    std::string latfile = replace_filename(argv[1]);

	// Load ring data from JSON
	std::ifstream jfile(latfile);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;
    // sublattice masks
    std::array<Uint128, 4> sl_masks = {0,0,0,0};
    
    const auto& atoms = jdata.at("atoms");
    for (size_t i=0; i<atoms.size(); i++){
        or_bit(sl_masks[ stoi(atoms[i].at("sl").get<std::string>()) ], i);
    }


    partition_basis_hdf5(argv[1], sl_masks);
    
    printf("\n\n Done! \n\n");


    return 0;
}

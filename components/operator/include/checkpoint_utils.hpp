#pragma once

#include <string>
#include <vector>
#include <complex>
#include <ctime>
#include <mpi.h>
#include <hdf5.h>
#include <iostream>
#include <cstdlib>
#include <fstream>

#include "operator_mpi.hpp"

namespace projED {
namespace checkpoint {

// Parse SLURM environment to get job end time
inline time_t get_slurm_end_time() {
    const char* end_str = std::getenv("SLURM_JOB_END_TIME");
    if (end_str) {
        return static_cast<time_t>(std::atol(end_str));
    }
    
    // Fallback: calculate from start time and time limit
    const char* start_str = std::getenv("SLURM_JOB_START_TIME");
    const char* limit_str = std::getenv("SLURM_JOB_TIME_LIMIT");
    
    if (start_str && limit_str) {
        time_t start = static_cast<time_t>(std::atol(start_str));
        long limit_minutes = std::atol(limit_str);
        return start + (limit_minutes * 60);
    }
    
    return 0; // No SLURM environment detected
}

// Check if we should checkpoint (less than threshold seconds remaining)
inline bool should_checkpoint(time_t end_time, int threshold_seconds = 3600) {
    if (end_time == 0) return false;
    
    time_t now = std::time(nullptr);
    time_t remaining = end_time - now;
    
    return remaining < threshold_seconds;
}

// Structure to hold checkpoint data
template<typename _S>
struct CheckpointData {
    std::vector<_S> v;           // Current Lanczos vector
    std::vector<_S> u;           // Previous Lanczos vector
    std::vector<double> alphas;  // Alpha coefficients
    std::vector<double> betas;   // Beta coefficients
    double beta;                 // Current beta value
    size_t iteration;            // Current iteration number
    double eigval;               // Current eigenvalue estimate
    int random_seed;             // For reproducibility
};

// Save checkpoint to HDF5 (parallel I/O)
template<typename _S>
void save_checkpoint(const std::string& filename,
                     const CheckpointData<_S>& data,
                     const MPIctx& ctx) {
    int rank = ctx.my_rank;
    
    // Create parallel file access
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);
    
    if (file_id < 0) {
        throw std::runtime_error("Failed to create checkpoint file");
    }
    
    // Write vectors v and u (distributed across ranks)
    auto write_distributed_vector = [&](const std::vector<_S>& vec, const char* name) {
        hsize_t global_dims[1] = {static_cast<hsize_t>(ctx.global_basis_dim())};
        hsize_t local_size = ctx.local_block_size();
        hsize_t offset = ctx.local_start_index();
        
        hid_t filespace = H5Screate_simple(1, global_dims, NULL);
        
        hid_t type_id;
        if constexpr (std::is_same_v<_S, double>) {
            type_id = H5T_NATIVE_DOUBLE;
        } else {
            // Complex type
            type_id = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
            H5Tinsert(type_id, "real", 0, H5T_NATIVE_DOUBLE);
            H5Tinsert(type_id, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
        }
        
        hid_t dset_id = H5Dcreate(file_id, name, type_id, filespace,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        hsize_t count[1] = {local_size};
        hsize_t start[1] = {offset};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL);
        
        hid_t memspace = H5Screate_simple(1, count, NULL);
        hid_t plist_xfer = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_xfer, H5FD_MPIO_COLLECTIVE);
        
        H5Dwrite(dset_id, type_id, memspace, filespace, plist_xfer, vec.data());
        
        H5Pclose(plist_xfer);
        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Dclose(dset_id);
        
        if constexpr (!std::is_same_v<_S, double>) {
            H5Tclose(type_id);
        }
    };
    
    write_distributed_vector(data.v, "v");
    write_distributed_vector(data.u, "u");
    
    H5Fclose(file_id);
    
    // Write scalar data from rank 0 only
    if (rank == 0) {
        hid_t serial_file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        
        // Write alphas
        hsize_t alpha_dims[1] = {data.alphas.size()};
        hid_t alpha_space = H5Screate_simple(1, alpha_dims, NULL);
        hid_t alpha_dset = H5Dcreate(serial_file, "alphas", H5T_NATIVE_DOUBLE,
                                     alpha_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(alpha_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.alphas.data());
        H5Dclose(alpha_dset);
        H5Sclose(alpha_space);
        
        // Write betas
        hsize_t beta_dims[1] = {data.betas.size()};
        hid_t beta_space = H5Screate_simple(1, beta_dims, NULL);
        hid_t beta_dset = H5Dcreate(serial_file, "betas", H5T_NATIVE_DOUBLE,
                                    beta_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(beta_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.betas.data());
        H5Dclose(beta_dset);
        H5Sclose(beta_space);
        
        // Write scalars
        hsize_t scalar_dims[1] = {1};
        hid_t scalar_space = H5Screate_simple(1, scalar_dims, NULL);
        
        hid_t beta_scalar = H5Dcreate(serial_file, "beta", H5T_NATIVE_DOUBLE,
                                      scalar_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(beta_scalar, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.beta);
        H5Dclose(beta_scalar);
        
        hid_t iter_dset = H5Dcreate(serial_file, "iteration", H5T_NATIVE_ULONG,
                                    scalar_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(iter_dset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.iteration);
        H5Dclose(iter_dset);
        
        hid_t eigval_dset = H5Dcreate(serial_file, "eigval", H5T_NATIVE_DOUBLE,
                                      scalar_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(eigval_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.eigval);
        H5Dclose(eigval_dset);
        
        hid_t seed_dset = H5Dcreate(serial_file, "random_seed", H5T_NATIVE_INT,
                                    scalar_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(seed_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.random_seed);
        H5Dclose(seed_dset);
        
        H5Sclose(scalar_space);
        H5Fclose(serial_file);
        
        std::cout << "[Checkpoint] Saved at iteration " << data.iteration << std::endl;
    }
}

// Load checkpoint from HDF5
template<typename _S>
bool load_checkpoint(const std::string& filename,
                     CheckpointData<_S>& data,
                     const MPIctx& ctx) {
    // Check if file exists
    if (ctx.my_rank == 0) {
        std::ifstream f(filename);
        bool exists = f.good();
        int exists_flag = exists ? 1 : 0;
        MPI_Bcast(&exists_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!exists) return false;
    } else {
        int exists_flag;
        MPI_Bcast(&exists_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!exists_flag) return false;
    }
    
    // Open parallel file
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);
    
    if (file_id < 0) return false;
    
    // Read distributed vectors
    auto read_distributed_vector = [&](std::vector<_S>& vec, const char* name) {
        hsize_t local_size = ctx.local_block_size();
        hsize_t offset = ctx.local_start_index();
        vec.resize(local_size);
        
        hid_t dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
        hid_t filespace = H5Dget_space(dset_id);
        
        hsize_t count[1] = {local_size};
        hsize_t start[1] = {offset};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL);
        
        hid_t memspace = H5Screate_simple(1, count, NULL);
        hid_t plist_xfer = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_xfer, H5FD_MPIO_COLLECTIVE);
        
        hid_t type_id;
        if constexpr (std::is_same_v<_S, double>) {
            type_id = H5T_NATIVE_DOUBLE;
        } else {
            type_id = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
            H5Tinsert(type_id, "real", 0, H5T_NATIVE_DOUBLE);
            H5Tinsert(type_id, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
        }
        
        H5Dread(dset_id, type_id, memspace, filespace, plist_xfer, vec.data());
        
        H5Pclose(plist_xfer);
        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Dclose(dset_id);
        
        if constexpr (!std::is_same_v<_S, double>) {
            H5Tclose(type_id);
        }
    };
    
    read_distributed_vector(data.v, "v");
    read_distributed_vector(data.u, "u");
    
    H5Fclose(file_id);
    
    // Read scalar data from rank 0, then broadcast
    if (ctx.my_rank == 0) {
        hid_t serial_file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        
        // Read alphas
        hid_t alpha_dset = H5Dopen(serial_file, "alphas", H5P_DEFAULT);
        hid_t alpha_space = H5Dget_space(alpha_dset);
        hsize_t alpha_size;
        H5Sget_simple_extent_dims(alpha_space, &alpha_size, NULL);
        data.alphas.resize(alpha_size);
        H5Dread(alpha_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.alphas.data());
        H5Dclose(alpha_dset);
        H5Sclose(alpha_space);
        
        // Read betas
        hid_t beta_dset = H5Dopen(serial_file, "betas", H5P_DEFAULT);
        hid_t beta_space = H5Dget_space(beta_dset);
        hsize_t beta_size;
        H5Sget_simple_extent_dims(beta_space, &beta_size, NULL);
        data.betas.resize(beta_size);
        H5Dread(beta_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.betas.data());
        H5Dclose(beta_dset);
        H5Sclose(beta_space);
        
        // Read scalars
        hid_t beta_scalar = H5Dopen(serial_file, "beta", H5P_DEFAULT);
        H5Dread(beta_scalar, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.beta);
        H5Dclose(beta_scalar);
        
        hid_t iter_dset = H5Dopen(serial_file, "iteration", H5P_DEFAULT);
        H5Dread(iter_dset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.iteration);
        H5Dclose(iter_dset);
        
        hid_t eigval_dset = H5Dopen(serial_file, "eigval", H5P_DEFAULT);
        H5Dread(eigval_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.eigval);
        H5Dclose(eigval_dset);
        
        hid_t seed_dset = H5Dopen(serial_file, "random_seed", H5P_DEFAULT);
        H5Dread(seed_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.random_seed);
        H5Dclose(seed_dset);
        
        H5Fclose(serial_file);
        
        std::cout << "[Checkpoint] Loaded from iteration " << data.iteration << std::endl;
    }
    
    // Broadcast scalar data to all ranks
    int alpha_size = data.alphas.size();
    int beta_size = data.betas.size();
    MPI_Bcast(&alpha_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (ctx.my_rank != 0) {
        data.alphas.resize(alpha_size);
        data.betas.resize(beta_size);
    }
    
    MPI_Bcast(data.alphas.data(), alpha_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(data.betas.data(), beta_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data.beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data.iteration, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data.eigval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data.random_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    return true;
}

} // namespace checkpoint
} // namespace projED




#pragma once

#include <vector>
#include <string>
#include <complex>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include "mpi.h"
#include "mpi_context.hpp"

namespace projED {

// Forward declaration assumed from elsewhere
struct MPIContext {
    int my_rank;
    int n_ranks;
};

namespace checkpoint {

// -----------------------------------------------------------------------
// Data bundle saved/restored at each checkpoint
// -----------------------------------------------------------------------
template<typename _S>
struct CheckpointData {
    std::vector<_S>    v;           // current Lanczos vector v_j
    std::vector<_S>    u;           // previous/scratch vector
    std::vector<double> alphas;     // diagonal elements collected so far
    std::vector<double> betas;      // off-diagonal elements collected so far
    double              beta   = 0.0;
    size_t              iteration = 0;
    double              eigval = std::numeric_limits<double>::max();
    unsigned int        random_seed = 0;
};

// -----------------------------------------------------------------------
// SLURM helpers
// -----------------------------------------------------------------------

/// Returns the SLURM job end time from $SLURM_JOB_END_TIME (seconds since
/// epoch), or 0 if not running under SLURM.
inline time_t get_slurm_end_time()
{
    const char* env = std::getenv("SLURM_JOB_END_TIME");
    if (!env) return 0;
    return static_cast<time_t>(std::atoll(env));
}

/// Returns true when there are fewer than `safety_seconds` seconds left
/// before `job_end_time`.
inline bool should_checkpoint(time_t job_end_time, int safety_seconds = 3600)
{
    if (job_end_time == 0) return false;
    time_t now = std::time(nullptr);
    return (job_end_time - now) < static_cast<time_t>(safety_seconds);
}

// -----------------------------------------------------------------------
// Binary I/O helpers
// -----------------------------------------------------------------------
namespace detail {

template<typename T>
inline void write_pod(std::ofstream& f, const T& val)
{
    f.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template<typename T>
inline bool read_pod(std::ifstream& f, T& val)
{
    return static_cast<bool>(f.read(reinterpret_cast<char*>(&val), sizeof(T)));
}

template<typename T>
inline void write_vec(std::ofstream& f, const std::vector<T>& v)
{
    size_t n = v.size();
    write_pod(f, n);
    if (n > 0)
        f.write(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
}

template<typename T>
inline bool read_vec(std::ifstream& f, std::vector<T>& v)
{
    size_t n = 0;
    if (!read_pod(f, n)) return false;
    v.resize(n);
    if (n > 0)
        if (!f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T))) return false;
    return true;
}

} // namespace detail

// -----------------------------------------------------------------------
// File-name helper: each rank writes its own shard
//   <base>.<rank_of_N_ranks>.bin
// -----------------------------------------------------------------------
inline std::string shard_filename(const std::string& base, int rank, int n_ranks)
{
    std::ostringstream oss;
    oss << base << "." << rank << "_of_" << n_ranks << ".bin";
    return oss.str();
}

// -----------------------------------------------------------------------
// Magic number / version for basic sanity checking
// -----------------------------------------------------------------------
static constexpr uint64_t CKPT_MAGIC   = 0x4C434B5054000001ULL; // "LCKPT\0\0\1"
static constexpr uint32_t CKPT_VERSION = 1;

// -----------------------------------------------------------------------
// save_checkpoint
//
// Each MPI rank independently writes its own shard to
//   <checkpoint_file>.<rank>_of_<n_ranks>.bin
//
// File layout (binary, host byte-order):
//   uint64  magic
//   uint32  version
//   int32   rank
//   int32   n_ranks
//   size_t  iteration
//   double  beta
//   double  eigval
//   uint32  random_seed
//   vector<_S>     v       (size_t n, then n elements)
//   vector<_S>     u
//   vector<double> alphas
//   vector<double> betas
// -----------------------------------------------------------------------
template<typename _S>
void save_checkpoint(const std::string& checkpoint_file,
                     const CheckpointData<_S>& data,
                     const MPIHashContext& ctx)
{
    int n_ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    std::string fname = shard_filename(checkpoint_file, ctx.my_rank, n_ranks);

    // Write to a tmp file first, then rename atomically
    std::string tmp_fname = fname + ".tmp";
    {
        std::ofstream f(tmp_fname, std::ios::binary | std::ios::trunc);
        if (!f) {
            std::cerr << "[Checkpoint] rank " << ctx.my_rank
                      << " could not open " << tmp_fname << " for writing\n";
            return;
        }

        detail::write_pod(f, CKPT_MAGIC);
        detail::write_pod(f, CKPT_VERSION);
        detail::write_pod(f, static_cast<int32_t>(ctx.my_rank));
        detail::write_pod(f, static_cast<int32_t>(n_ranks));
        detail::write_pod(f, data.iteration);
        detail::write_pod(f, data.beta);
        detail::write_pod(f, data.eigval);
        detail::write_pod(f, data.random_seed);

        detail::write_vec(f, data.v);
        detail::write_vec(f, data.u);
        detail::write_vec(f, data.alphas);
        detail::write_vec(f, data.betas);

        if (!f) {
            std::cerr << "[Checkpoint] rank " << ctx.my_rank
                      << " write error on " << tmp_fname << "\n";
            return;
        }
    }

    // Atomic rename
    if (std::rename(tmp_fname.c_str(), fname.c_str()) != 0) {
        std::cerr << "[Checkpoint] rank " << ctx.my_rank
                  << " rename failed for " << fname << "\n";
        return;
    }

    if (ctx.my_rank == 0) {
        std::cout << "[Checkpoint] Saved checkpoint to " << checkpoint_file
                  << " (iter=" << data.iteration << ")\n";
    }
}

// -----------------------------------------------------------------------
// load_checkpoint
//
// Each rank loads its own shard.  Returns true only if the shard for
// *this* rank is present, readable, consistent (magic/version), and the
// local vector dimension matches the current v.size().
//
// On failure the function returns false and leaves `data` untouched so
// the caller can start fresh.
// -----------------------------------------------------------------------
template<typename _S>
bool load_checkpoint(const std::string& checkpoint_file,
                     CheckpointData<_S>& data,
                     const MPIHashContext& ctx)
{
    int n_ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    std::string fname = shard_filename(checkpoint_file, ctx.my_rank, n_ranks);

    std::ifstream f(fname, std::ios::binary);
    if (!f) {
        // No checkpoint shard for this rank – treat as fresh start
        if (ctx.my_rank == 0) {
            std::cout << "[Checkpoint] No checkpoint found at " << checkpoint_file
                      << ", starting fresh.\n";
        }
        return false;
    }

    // --- magic & version ---
    uint64_t magic = 0;
    uint32_t version = 0;
    if (!detail::read_pod(f, magic) || magic != CKPT_MAGIC) {
        std::cerr << "[Checkpoint] rank " << ctx.my_rank
                  << ": bad magic in " << fname << "\n";
        return false;
    }
    if (!detail::read_pod(f, version) || version != CKPT_VERSION) {
        std::cerr << "[Checkpoint] rank " << ctx.my_rank
                  << ": unsupported version " << version << " in " << fname << "\n";
        return false;
    }

    // --- rank / n_ranks consistency ---
    int32_t saved_rank = -1, saved_n_ranks = -1;
    if (!detail::read_pod(f, saved_rank) || !detail::read_pod(f, saved_n_ranks)) {
        std::cerr << "[Checkpoint] rank " << ctx.my_rank
                  << ": truncated header in " << fname << "\n";
        return false;
    }
    if (saved_rank != ctx.my_rank || saved_n_ranks != n_ranks) {
        std::cerr << "[Checkpoint] rank " << ctx.my_rank
                  << ": rank/n_ranks mismatch in " << fname
                  << " (saved " << saved_rank << "/" << saved_n_ranks << ")\n";
        return false;
    }

    // --- scalar fields ---
    CheckpointData<_S> tmp;
    if (!detail::read_pod(f, tmp.iteration)) return false;
    if (!detail::read_pod(f, tmp.beta))      return false;
    if (!detail::read_pod(f, tmp.eigval))    return false;
    if (!detail::read_pod(f, tmp.random_seed)) return false;

    // --- vectors ---
    if (!detail::read_vec(f, tmp.v))      return false;
    if (!detail::read_vec(f, tmp.u))      return false;
    if (!detail::read_vec(f, tmp.alphas)) return false;
    if (!detail::read_vec(f, tmp.betas))  return false;

    // --- dimension check ---
    // data.v is pre-sized to the local dimension by the caller before
    // calling load_checkpoint; we check that the shard matches.
    if (!data.v.empty() && tmp.v.size() != data.v.size()) {
        std::cerr << "[Checkpoint] rank " << ctx.my_rank
                  << ": local dimension mismatch: expected " << data.v.size()
                  << " got " << tmp.v.size() << " in " << fname << "\n";
        return false;
    }

    data = std::move(tmp);

    if (ctx.my_rank == 0) {
        std::cout << "[Checkpoint] Loaded checkpoint from " << checkpoint_file
                  << " at iteration " << data.iteration << "\n";
    }
    return true;
}

} // namespace checkpoint
} // namespace projED





#pragma once
// Shared resolution of the fused pipelines' per-rank in-RAM shard soft cap
// (see mpi_par_searcher::set_local_memory_limit / --local-memory).
#include <cstdlib>
#include <cstddef>
#include <string>
#include "logging.hpp"

namespace projED {

// Resolve the per-rank in-RAM shard soft cap, in bytes, from the --local-memory
// CLI value:
//   cli_gib  > 0 : explicit cap in GiB.
//   cli_gib == 0 : explicitly unlimited.
//   cli_gib  < 0 : "auto" -- derive a sensible default from the SLURM
//                  allocation: half of this rank's memory share, leaving the
//                  other half for the comm buffers, MPI/UCX, and the OS. The
//                  share is taken from SLURM_MEM_PER_CPU x SLURM_CPUS_PER_TASK
//                  (--mem-per-cpu jobs), falling back to SLURM_MEM_PER_NODE /
//                  tasks-per-node (--mem jobs). Falls back to unlimited (0) when
//                  neither is available (not under SLURM), so local/non-SLURM
//                  runs and the test suite are unaffected.
// Logs the chosen value at INFO (rank 0 only).
inline std::size_t resolve_local_memory_bytes(double cli_gib) {
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    if (cli_gib > 0.0)  return static_cast<std::size_t>(cli_gib * GiB);
    if (cli_gib == 0.0) return 0;

    auto env_ull = [](const char* name, unsigned long long& out) -> bool {
        const char* s = std::getenv(name);
        if (!s || !*s) return false;
        char* end = nullptr;
        unsigned long long v = std::strtoull(s, &end, 10);
        if (end == s) return false;
        out = v;
        return true;
    };

    // Tasks (ranks) per node: SLURM_NTASKS_PER_NODE if set, else the leading
    // count of SLURM_TASKS_PER_NODE (e.g. "192(x8)" -> 192; strtoull stops at
    // the '(' or ',').
    auto tasks_per_node = [&](unsigned long long& out) -> bool {
        if (env_ull("SLURM_NTASKS_PER_NODE", out) && out > 0) return true;
        return env_ull("SLURM_TASKS_PER_NODE", out) && out > 0;
    };

    long double rank_bytes = 0.0L;
    std::string detail;

    // Preferred: per-CPU memory x CPUs/task (SLURM_CPUS_PER_TASK, or the
    // OMP_NUM_THREADS the job scripts export from it).
    unsigned long long mem_per_cpu_mb = 0, cpus_per_task = 0;
    const bool have_cpus = env_ull("SLURM_CPUS_PER_TASK", cpus_per_task) ||
                           env_ull("OMP_NUM_THREADS", cpus_per_task);
    if (env_ull("SLURM_MEM_PER_CPU", mem_per_cpu_mb) && have_cpus &&
            mem_per_cpu_mb > 0 && cpus_per_task > 0) {
        rank_bytes = static_cast<long double>(mem_per_cpu_mb) * cpus_per_task *
                1024.0L * 1024.0L;
        detail = "SLURM_MEM_PER_CPU " + std::to_string(mem_per_cpu_mb) +
                 " MB x " + std::to_string(cpus_per_task) + " CPUs/task";
    } else {
        // Fallback: per-node memory / ranks-per-node (--mem jobs).
        unsigned long long mem_per_node_mb = 0, ntasks_per_node = 0;
        if (env_ull("SLURM_MEM_PER_NODE", mem_per_node_mb) &&
                tasks_per_node(ntasks_per_node) &&
                mem_per_node_mb > 0 && ntasks_per_node > 0) {
            rank_bytes = static_cast<long double>(mem_per_node_mb) /
                    ntasks_per_node * 1024.0L * 1024.0L;
            detail = "SLURM_MEM_PER_NODE " + std::to_string(mem_per_node_mb) +
                     " MB / " + std::to_string(ntasks_per_node) + " tasks/node";
        }
    }

    if (rank_bytes > 0.0L) {
        const std::size_t cap = static_cast<std::size_t>(0.5L * rank_bytes);
        logging::log(logging::INFO)
            << "[local-memory] auto cap = " << (cap >> 20) << " MiB/rank"
            << " (0.5 x " << detail << ")\n";
        return cap;
    }

    logging::log(logging::INFO)
        << "[local-memory] auto default requested but SLURM memory env "
           "(SLURM_MEM_PER_CPU / SLURM_MEM_PER_NODE) unavailable -- shard cap "
           "disabled (unlimited)\n";
    return 0;
}

} // namespace projED

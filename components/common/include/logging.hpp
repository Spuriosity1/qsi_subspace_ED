#pragma once
#include <iostream>
#include <ostream>
#include <streambuf>

// Lightweight, process-wide logging control.
//
// In an MPI job there is one process per rank, so a single global config is
// enough and avoids threading a logger object through every constructor.
// Messages carry a level; `log(level)` returns std::cout when this rank should
// emit at that level and a discarding stream otherwise. Crucially, no log
// files are ever opened: by default only rank 0 writes, straight to stdout.
namespace logging {

enum Level : int {
    SILENT = 0,  // nothing
    INFO   = 1,  // high-level progress / milestones (default)
    DEBUG  = 2,  // per-rank detail, sizes, plans
    TRACE  = 3,  // per-iteration chatter (work stealing, ring shutdown, ...)
};

// A streambuf that swallows everything written to it.
class NullBuffer : public std::streambuf {
public:
    int overflow(int c) override { return c; }
};

inline std::ostream& null_stream() {
    static NullBuffer buf;
    static std::ostream os(&buf);
    return os;
}

struct Config {
    int  rank      = 0;      // this process's MPI rank
    int  verbosity = INFO;   // highest level that is emitted
    bool all_ranks = false;  // if false, only rank 0 emits (no files either way)
};

inline Config& config() {
    static Config c;
    return c;
}

inline void configure(int rank, int verbosity, bool all_ranks) {
    Config& c = config();
    c.rank      = rank;
    c.verbosity = verbosity;
    c.all_ranks = all_ranks;
}

inline bool enabled(int level = INFO) {
    const Config& c = config();
    return level <= c.verbosity && (c.rank == 0 || c.all_ranks);
}

// Returns std::cout if a message at `level` should be emitted by this rank,
// otherwise a stream that discards its input.
inline std::ostream& log(int level = INFO) {
    return enabled(level) ? std::cout : null_stream();
}

} // namespace logging

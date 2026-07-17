#pragma once
#include <argparse/argparse.hpp>
#include "logging.hpp"

inline void provide_logging_options(argparse::ArgumentParser& prog) {
    prog.add_argument("--verbosity", "-v")
        .help("Logging verbosity: 0=silent, 1=info, 2=debug, 3=trace")
        .default_value(static_cast<int>(logging::INFO))
        .scan<'i', int>();
    prog.add_argument("--log-all-ranks")
        .help("Emit log output from every MPI rank (default: only rank 0). "
              "No per-rank log files are created; output is interleaved on stdout.")
        .default_value(false)
        .implicit_value(true);
}

// Push the parsed verbosity/rank settings into the process-wide logging config.
inline void configure_logging(const argparse::ArgumentParser& prog, int rank) {
    logging::configure(rank, prog.get<int>("--verbosity"),
                       prog.get<bool>("--log-all-ranks"));
}

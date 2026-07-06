// shard.hpp
#pragma once
#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <unistd.h> // for fsync
#include <sys/stat.h>
#include "bittools.hpp"


class ShardWriter {
    std::string inpath;
    std::string donepath;
    FILE* f = nullptr;
    std::vector<Uint128> buf;
    size_t buf_limit_entries;

public:
    ShardWriter(const std::string& basepath, size_t buf_limit_entries_ = (1<<20))
     : inpath(basepath + ".inprogress"),
       donepath(basepath + ".done"),
       buf_limit_entries(buf_limit_entries_)
    {
        // open in append mode, so resume writes will append
        f = fopen(inpath.c_str(), "ab");
        if (!f) throw std::runtime_error("ShardWriter: fopen failed for " + inpath);
        buf.reserve(std::min<size_t>(buf_limit_entries, 1<<16));
    }

    ~ShardWriter(){
        try { finalize(false); } catch(...) {}
    }

    void push(const Uint128 &v){
        buf.push_back(v);
        if (buf.size() >= buf_limit_entries) flush();
    }

    void flush(bool do_fsync = false) {
        if (buf.empty() || !f) return;
        size_t written = fwrite(buf.data(), sizeof(Uint128), buf.size(), f);
        if (written != buf.size()) throw std::runtime_error("ShardWriter: incomplete fwrite");
        buf.clear();
        fflush(f);
        if (do_fsync){
            int fd = fileno(f);
            if (fd >= 0) fsync(fd);
        }
    }

    // finalize: flush and atomically rename .inprogress -> .done
    // If move_to_done==false, we keep .inprogress for resume.
    void finalize(bool move_to_done = true) {
        if (f) {
            flush(true);
            fclose(f);
            f = nullptr;
        }
        if (move_to_done) {
            // atomic on POSIX
            rename(inpath.c_str(), donepath.c_str());
        }
    }

    const std::string &in_progress_path() const { return inpath; }
    const std::string &done_path() const { return donepath; }
};


// In-memory drop-in replacement for ShardWriter: accumulates states in RAM
// instead of streaming them to disk. The path argument is accepted (and
// ignored) so path-constructing callers like mpi_par_searcher need no changes.
//
// States are stored in fixed-size blocks rather than one growing vector:
// vector doubling would transiently hold old + new capacity (~3x the data)
// at each reallocation, whereas blocks keep the search-phase high-water mark
// at (data + one block). Blocks can be filtered/shrunk in place before
// take_states() assembles them into a single contiguous vector.
class MemoryShard {
    std::vector<std::vector<Uint128>> blocks;
    size_t block_entries;
    std::string fake_path; // returned by the *_path() accessors

public:
    MemoryShard(const std::string& basepath, size_t block_entries_ = (1<<20))
     : block_entries(block_entries_ > 0 ? block_entries_ : (1<<20)),
       fake_path(basepath) {}

    void push(const Uint128 &v) {
        if (blocks.empty() || blocks.back().size() >= block_entries) {
            blocks.emplace_back();
            blocks.back().reserve(block_entries);
        }
        blocks.back().push_back(v);
    }
    void flush(bool = false) {}
    void finalize(bool = true) {}

    const std::string &in_progress_path() const { return fake_path; }
    const std::string &done_path() const { return fake_path; }

    size_t size() const {
        size_t n = 0;
        for (const auto& b : blocks) n += b.size();
        return n;
    }

    // Direct access to the underlying blocks, e.g. to filter states out
    // before assembly (shrink_to_fit filtered blocks to reclaim the space).
    std::vector<std::vector<Uint128>>& mutable_blocks() { return blocks; }

    // Assemble the accumulated states into one contiguous vector, freeing
    // each block as it is copied. Leaves the shard empty.
    std::vector<Uint128> take_states() {
        std::vector<Uint128> out;
        out.reserve(size());
        for (auto& b : blocks) {
            out.insert(out.end(), b.begin(), b.end());
            std::vector<Uint128>().swap(b);
        }
        blocks.clear();
        return out;
    }
};





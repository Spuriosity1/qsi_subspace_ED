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


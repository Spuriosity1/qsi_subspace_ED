#include <queue>
#include <fstream>
#include <filesystem>
#include "bittools.hpp"
#include <algorithm>
#include <iostream>
#include "sort.hpp"
#include "argparse/argparse.hpp"

int main(int argc, char* argv[]) {
    
    argparse::ArgumentParser prog("sort_shard");
    prog.add_argument("shardfile")
        .required()
        .help("shard to sort in-place");

    prog.add_argument("--memory_limit", "-m")
        .default_value(static_cast<size_t>(1 << 30))
        .scan<'i', size_t>()
        .help("RAM limit for working memory");
    
    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err) {
            std::cerr << err.what() << std::endl;
            std::cerr << prog;
        return 1;
    }
    
    auto shardfile = prog.get<std::string>("shardfile");
    
    size_t memory_limit = prog.get<size_t>("--memory_limit");
    
    try {
        sort_shard_file(shardfile, memory_limit, false);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

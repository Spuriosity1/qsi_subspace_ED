#include <mpi.h>
#include <fstream>
#include <filesystem>
#include <hdf5.h>
#include <nlohmann/json.hpp>
#include "argparse/argparse.hpp"
#include "bittools.hpp"
#include <algorithm>

#include <array>
#include <iostream>

using namespace nlohmann;

std::string make_sector_string(const std::array<int, 4> s)
{
    std::ostringstream oss;
    oss << s[0]<<'.'<<s[1]<<'.'<<s[2]<<'.'<<s[3];
    return oss.str();
}

std::string make_sector_string_from_id(uint32_t sid)
{
    std::array<int, 4> s;
    for (int i=0; i<4; i++){
        s[i] = sid&0xff;
        sid >>=8;
    }
    return make_sector_string(s);
}

struct SectorStats {
    uint64_t count = 0;
    std::string shard_file;
};

struct ShardContribution {
    uint32_t sector_id;
    uint64_t count;
    std::string shard_path;
};

std::vector<ShardContribution> partition_shard(const std::filesystem::path& shard_file,
                     const std::string& job_tag,
                     const std::array<Uint128,4>& sl_masks,
                     size_t BUFFER_CAPACITY = 6'250'000) // ~100 MB
{
    std::vector<Uint128> chunk(BUFFER_CAPACITY);

    // Each sector corresponds to a buffer + output file + stats
    struct SectorBin {
        std::vector<Uint128> buf;
        std::ofstream out;
        uint64_t count = 0;
    };

    // Map sector id → index in `bins`
    std::unordered_map<uint32_t, size_t> sector_index;
    sector_index.reserve(128);

    std::vector<SectorBin> bins;
    bins.reserve(128);

    // Track all sectors encountered
    std::vector<uint32_t> encountered_sectors;

    std::vector<ShardContribution> contrib;

    std::ifstream in(shard_file, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open shard file: " + shard_file.string());

    auto work_dir = shard_file.parent_path();
    auto filename = shard_file.filename();

    auto get_sector_bin = [&](uint32_t sid) -> SectorBin& {
        auto it = sector_index.find(sid);
        if (it != sector_index.end())
            return bins[it->second];

        // New sector encountered → create directory + open file + allocate buffer
        size_t new_idx = bins.size();
        sector_index[sid] = new_idx;
        bins.emplace_back();
        encountered_sectors.push_back(sid);
        auto &bin = bins.back();

        // Pre-allocate buffer
        bin.buf.reserve(BUFFER_CAPACITY);

        // Construct output path
        std::string sector_name = make_sector_string_from_id(sid);
        std::filesystem::create_directories(work_dir / sector_name);
        auto outpath = work_dir / sector_name / filename;

        bin.out.open(outpath, std::ios::binary | std::ios::app);
        if (!bin.out) {
            throw std::runtime_error("Cannot open output file: " + outpath.string());
        }
        return bin;
    };

    while (true) {
        in.read(reinterpret_cast<char*>(chunk.data()),
                BUFFER_CAPACITY * sizeof(Uint128));
        size_t got = in.gcount() / sizeof(Uint128);
        if (got == 0) break;

        for (size_t k = 0; k < got; k++) {
            Uint128 x = chunk[k];

            // Popcnt evaluation (keep branchless)
            uint32_t s0 = popcnt_u128(sl_masks[0] & x);
            uint32_t s1 = popcnt_u128(sl_masks[1] & x);
            uint32_t s2 = popcnt_u128(sl_masks[2] & x);
            uint32_t s3 = popcnt_u128(sl_masks[3] & x);

            // Pack into single sector key
            uint32_t sid = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);

            // Fetch or create output bin
            auto &bin = get_sector_bin(sid);

            bin.buf.push_back(x);
            bin.count++;

            if (bin.buf.size() >= BUFFER_CAPACITY) {
                bin.out.write(reinterpret_cast<char*>(bin.buf.data()),
                              bin.buf.size() * sizeof(Uint128));
                bin.buf.clear();
            }
        }
    }

    // Final writes and manifest creation
    for (uint32_t sid : encountered_sectors) {
        size_t idx = sector_index[sid];
        auto &bin = bins[idx];
        
        if (!bin.buf.empty()) {
            bin.out.write(reinterpret_cast<char*>(bin.buf.data()),
                          bin.buf.size() * sizeof(Uint128));
        }
        bin.out.close();
        
        contrib.emplace_back(sid, bin.count, shard_file);
    }
    return contrib;
}


void update_sector_manifest(uint32_t sid, uint64_t count,
        const std::string& shard_file, const std::string& job_tag){

        // Update sector manifest
        std::string sector_name = make_sector_string_from_id(sid);
        auto sector_dir = std::filesystem::path(shard_file).parent_path() / sector_name;
        auto manifest_path = sector_dir / "partition_manifest.json";

        json sector_manifest;
        
        // Load existing manifest if it exists
        if (std::filesystem::exists(manifest_path)) {
            std::ifstream mf(manifest_path);
            mf >> sector_manifest;
        } else {
            sector_manifest["sector"] = sector_name;
            sector_manifest["sector_id"] = sid;
            sector_manifest["shards"] = json::array();
            sector_manifest["job_tag"] = job_tag;
            sector_manifest["n_shards"] = 0;
            sector_manifest["size"] = 0;
        }

        // Add this shard's contribution
//        json shard_entry;
//        shard_entry["shard_id"] = shard_id;
//        shard_entry["shard_file"] = filename.string();
//        shard_entry["count"] = bin.count;
        
        sector_manifest["shards"].push_back(shard_file);
        sector_manifest["n_shards"] = sector_manifest["n_shards"].get<int>() + 1;
        sector_manifest["size"] = sector_manifest["size"].get<int>() + count;

        
        // Write manifest
        std::ofstream mf(manifest_path);
        mf << sector_manifest.dump(2);


}

int main(int argc, char* argv[]) {
       
    argparse::ArgumentParser prog("partition_shard");
    prog.add_argument("manifest")
        .required()
        .help("Manifest JSON file produced by shard writer");
    prog.add_argument("--batch_size", "-b")
        .default_value(static_cast<size_t>(1'000'000))
        .scan<'i', size_t>()
        .help("Batch size for buffered merging");
    prog.add_argument("--latfile_dir")
        .default_value("../lattice_files")
        .help("direcory of the original lattice json");
    
    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << prog;
        return 1;
    }

        
    auto manifest_file = prog.get<std::string>("manifest");
    
    // Load manifest (all ranks read it)
    std::ifstream ifs(manifest_file);
    if (!ifs) {
        std::cerr << "Failed to open manifest file\n";
        return 1;
    }
    json manifest = json::parse(ifs);
    std::vector<std::string> shards = manifest["shards"].get<std::vector<std::string>>();
    
    size_t batch_size = prog.get<size_t>("--batch_size");

    std::string latfile = prog.get<std::string>("--latfile_dir") + "/";
    latfile += manifest["job_tag"];
    latfile += ".json";

    // Load ring data from JSON
    std::cout<<"Opening "<<latfile<<"...\n";
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

    std::vector<ShardContribution> contributions;


    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != shards.size()){
        std::cerr<<"Error: call with "<<shards.size()<<
            " MPI tasks\n";
        MPI_Finalize();
        return 1;
    }

    try {
        contributions = partition_shard(shards[rank], manifest["job_tag"], sl_masks, batch_size);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Serialize contributions to a string for MPI
    json msg = json::array();
    for (auto &c : contributions) {
        msg.push_back({
            {"sector_id", c.sector_id},
            {"count", c.count},
            {"file", c.shard_path}
        });
    }
    std::string payload = msg.dump();
    int len = payload.size();

    // send to rank 0, the scribe
    //gather sizes
    std::vector<int> lengths(world_size);
    MPI_Gather(&len, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    // gather data
    // Gather payloads at rank 0
    std::vector<char> recvbuf;
    std::vector<int> displs(world_size);
    if (rank == 0) {
        size_t total = 0;
        for (int r=0; r<world_size; r++){
            displs[r] = total;
            total += lengths[r];
        }
        recvbuf.resize(total);
        std::cout << "[rank 0] receiving ";
        for (auto& l : lengths){
            std::cout << l <<" | ";
        }
        std::cout<<std::endl;
    }

    MPI_Gatherv(payload.data(), len, MPI_CHAR, 
            recvbuf.data(), lengths.data(),
            displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        std::string tmp_jt = manifest["job_tag"];

        size_t offset = 0;
        for (int r = 0; r < world_size; r++) {
            std::string s(recvbuf.begin() + offset,
                          recvbuf.begin() + offset + lengths[r]);
            offset += lengths[r];

            json arr = json::parse(s);
            for (auto &entry : arr) {
                uint32_t sid = entry["sector_id"];
                uint64_t count = entry["count"];
                std::string file = entry["file"];

                // Same code you already have — but only rank 0 executes it.
                update_sector_manifest(sid, count, file, tmp_jt);
            }
        }
    }

    MPI_Finalize();   
    return 0;
}

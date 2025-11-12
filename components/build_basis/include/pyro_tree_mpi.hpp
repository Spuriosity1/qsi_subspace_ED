#pragma once
#include <mpi.h>
#include <random>
#include <vector>
#include <csignal>
#include "mpi_context.hpp"
#include "pyro_tree.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>

inline int get_mpi_rank(){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    return myrank;
}

inline int get_mpi_world_size(){
    int x;
    MPI_Comm_size(MPI_COMM_WORLD, &x);
    return x;
}

class CheckpointWriter {
    std::filesystem::path ckpt_file;

    public:
    CheckpointWriter(const std::filesystem::path& file) : ckpt_file(file) {
        std::cout<<"Initialised checkpoint: "<<ckpt_file<<std::endl;
    }

    inline void save_stack(const lat_container::cust_stack& stack) {
        FILE* f = fopen(ckpt_file.c_str(), "wb");
        if (!f) throw std::runtime_error("save_stack: failed to open " + ckpt_file.string());

        size_t n = stack.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(stack.data(), sizeof(vtree_node_t), n, f);
        fclose(f);
    }

    inline void load_stack(lat_container::cust_stack& stack) {
        FILE* f = fopen(ckpt_file.c_str(), "rb");
        if (!f) return; // No restart available = start normally.
        std::cout <<"reading checkpoint data: "<<ckpt_file<<"\n";

        size_t n;
        fread(&n, sizeof(n), 1, f);
        stack.resize(n);
        fread(stack.data(), sizeof(vtree_node_t), n, f);
        fclose(f);
    }

    void finalize() {
        // deletes the checkpoints
        remove(ckpt_file.c_str());
    }
};

volatile extern sig_atomic_t GLOBAL_SHUTDOWN_REQUEST;

//#pragma pack(push,1)
//struct packet {
//    vtree_node_t state;
//    int32_t available;
//};

MPI_Datatype create_vtree_node_type();

//MPI_Datatype create_packet_type();

template<typename T>
requires std::derived_from<T, lat_container>
class mpi_par_searcher : public T {
//    static mpi_par_searcher<T>* global_self;
    int world_size;
    int my_rank;

    std::filesystem::path workdir;
    const std::string job_tag;
    
    const std::vector<size_t>& perm;


    static constexpr unsigned INITIAL_DEPTH_FACTOR = 5;
    static constexpr int CHECK_INTERVAL = 100000;
    static constexpr int PRINT_INTERVAL = 50; // print this many checks

    // MPI message tags
    static constexpr int TAG_WORK_REQUEST = 1;
    static constexpr int TAG_WORK_RESPONSE = 2;

    static constexpr int TAG_SHUTDOWN_RING = 300;
//    static constexpr int TAG_SHUTDOWN_COMPLETE = 301;
    static constexpr int NUM_TERMINATE_LOOPS = 3;


    // Work request status
    static constexpr int WORK_AVAILABLE = 1;
    static constexpr int WORK_UNAVAILABLE = 0;

    ShardWriter shard;
    CheckpointWriter checkpoint;

    lat_container::cust_stack my_job_stack;

    vtree_node_t pop_hardest_job();
    int lowest_spon_id_on_stack();

	void _build_state_dfs(lat_container::cust_stack &node_stack,
			unsigned long max_stack_size = (1ul << 40));
	void _build_state_bfs(std::queue<vtree_node_t>& node_stack, 
		unsigned long max_queue_len);

       // MPI-specific helper methods
    void state_tree_init();
    void distribute_initial_work(std::queue<vtree_node_t>& starting_nodes);
    void receive_initial_work();
    bool request_work_from_shuffled();
    bool request_work_from(int target_rank);
    bool check_work_requests(bool allow_steal=true);

    bool check_termination_requests(MPI_Request* send);
    void initiate_termination_check(MPI_Request* send);

    std::mt19937 rng;


    RankLogger db_log;

    std::ostream& db_print(const std::string& msg=""){
        return db_log << msg;
    }

    int continue_exit;


public:


mpi_par_searcher(const lattice& lat, unsigned num_spinon_pairs,
        const std::vector<size_t>& perm_,
        const std::filesystem::path& workdir_,
        const std::string &job_tag_,
        size_t buf_entries = 1<<20) :
    T(lat, num_spinon_pairs),
    world_size(get_mpi_world_size()),
    my_rank(get_mpi_rank()),
    workdir(workdir_),
    job_tag(job_tag_),
    perm(perm_),
    shard( workdir / ("shard-" + job_tag + "-" + std::to_string(my_rank) + ".bin"), buf_entries ),
    checkpoint( workdir / ("checkpoint-" + job_tag + "-" + std::to_string(my_rank) + ".bin") ),
    db_log(my_rank)
    {
        GLOBAL_SHUTDOWN_REQUEST=0;
        signal(SIGINT, sig_handler);
        signal(SIGTERM, sig_handler);
    }

    static void sig_handler(int){
        const char msg[] = "Exit requested...\n";
        write(STDERR_FILENO, msg, sizeof(msg) - 1);  // Async-signal-safe
        GLOBAL_SHUTDOWN_REQUEST=1;
    }

/////

    void build_state_tree();

    void finalise_shards(){
        shard.finalize(true);
        checkpoint.finalize();
        static const int filename_bufsize = 4096;
        char* sendbuf = new char[filename_bufsize];
        const std::string& my_name = shard.done_path();
        if (my_name.size() > filename_bufsize-1){
            throw std::runtime_error("filename too long!");
        }
        for (size_t i=0; i<filename_bufsize; i++){
            sendbuf[i] = (i < my_name.size()) ? my_name[i] : '\0';
        }
        char* namebuf = nullptr;
        if (my_rank == 0) {
            namebuf = new char[world_size*filename_bufsize];
        }
        MPI_Gather(sendbuf, filename_bufsize, MPI_CHAR, 
                namebuf, filename_bufsize, MPI_CHAR, 0,
                MPI_COMM_WORLD);
        delete[] sendbuf;
        
        if (my_rank != 0) return;
        
        nlohmann::json manifest;
        manifest["job_tag"] = job_tag;
        manifest["n_shards"] = world_size;
        manifest["shards"] = nlohmann::json::array();
        for (int t = 0; t < world_size; ++t) {
            std::string tmp(namebuf + t*filename_bufsize);
            manifest["shards"].push_back(tmp);
        }
        delete[] namebuf;
        
        // write manifest to file
        std::string manifest_path = workdir / ("manifest-" + job_tag + ".json");
        std::ofstream ofs(manifest_path);
        ofs << manifest.dump(2) << std::endl;
        ofs.close();
    }


};

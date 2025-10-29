#pragma once
#include <mpi.h>
#include <vector>
#include "pyro_tree.hpp"

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


class mpi_par_searcher : public lat_container {

    int world_size;
    int my_rank;

    std::filesystem::path workdir;
    const std::string job_tag;
    
    const std::vector<size_t>& perm;
    static constexpr unsigned INITIAL_DEPTH_FACTOR = 5;
    static constexpr int CHECKIN_INTERVAL = 50000000;
    static constexpr int WORK_REQUEST_TAG = 1;
    static constexpr int WORK_RESPONSE_TAG = 2;
    static constexpr int TERMINATION_TAG = 3;
    ShardWriter shard;

    cust_stack my_job_stack;

	void _build_state_dfs(cust_stack &node_stack,
			unsigned long max_stack_size = (1ul << 40));
	void _build_state_bfs(std::queue<vtree_node_t>& node_stack, 
		unsigned long max_queue_len);

       // MPI-specific helper methods
    void distribute_initial_work(std::queue<vtree_node_t>& starting_nodes);
    bool check_for_work_requests();
    bool request_work_from_others();
    void send_work_to_requester(int requester_rank);
    bool global_termination_check();

public:

mpi_par_searcher(const lattice& lat, unsigned num_spinon_pairs,
        const std::vector<size_t>& perm_,
        const std::filesystem::path& workdir_,
        const std::string &job_tag_,
        size_t buf_entries = 1<<20) :
    lat_container(lat, num_spinon_pairs),
    world_size(get_mpi_world_size()),
    my_rank(get_mpi_rank()),
    workdir(workdir_),
    job_tag(job_tag_),
    perm(perm_),
    shard( workdir / ("shard-" + job_tag + "-" + std::to_string(my_rank) + ".bin"), buf_entries )
    {

    }

    void build_state_tree();

    void finalise_shards(){
        shard.finalize(true);
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
        for (unsigned t = 0; t < world_size; ++t) {
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

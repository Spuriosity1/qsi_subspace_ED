#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>
#include "admin.hpp"
#include "bittools.hpp"
#include "basis_io.hpp"
#include <unordered_set>
#include "mpi_context.hpp"
#include "operator.hpp"
#include <mpi.h>

using namespace std;
using json=nlohmann::json;

struct IntDefaultedToZero
{
    int i = 0;
    int& operator()(){return i;}
    int operator()() const {return i;}
};



using state_t = Uint128;
typedef std::unordered_set<state_t, Uint128Hash, Uint128Eq> state_uset_t;

inline std::vector<int> close_packed_displ(const std::vector<int>& sendcounts){
    const int world_size = sendcounts.size();
    std::vector<int> send_displ(world_size);

    if (world_size > 0) {
        send_displ[0]=0;    
        for (int r=1; r<world_size; r++){
            send_displ[r] = send_displ[r-1] + sendcounts[r-1];
        }
    }
    return send_displ;
}


void get_all_ring_ops(std::vector<SymbolicPMROperator>& opset, const nlohmann::json& jdata){
    // obtains all operators in the ring-spec
	for (const auto& ring : jdata.at("rings")) {
		std::vector<int> spins = ring.at("member_spin_idx");

		std::vector<char> ops;
		std::vector<char> conj_ops;
		for (auto s : ring.at("signs")){
			ops.push_back( s == 1 ? '+' : '-');
			conj_ops.push_back( s == 1 ? '-' : '+');
		}
		
		auto O   = SymbolicPMROperator(     ops, spins);
		auto O_h = SymbolicPMROperator(conj_ops, spins);

		opset.push_back(O);
		opset.push_back(O_h);
	}
}


struct constr_explorer_mpi {
	constr_explorer_mpi(const nlohmann::json& data,
            const std::filesystem::path& workdir_,
            const std::string& job_tag_
            ) : 
        workdir(workdir_),
        job_tag(job_tag_),
        db_log(my_rank)
    {
        get_all_ring_ops(opset, data);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        n_states = data.at("atoms").size();
        buckets.resize(world_size);
   }

	void build_states(state_t init);

    auto get_out_file(){
        return  workdir / ("shard-" + job_tag + "-" + std::to_string(my_rank) + ".bin.done");
    }

    std::string write_shard(
            ){

        auto out_file = get_out_file();

        FILE* f = fopen(out_file.c_str(), "wb");
        if (!f) throw std::runtime_error("save: failed to open " + out_file.string());
        
        const int block_size = 1<<14;
        std::vector<state_t> buf;
        buf.reserve(block_size);
        
        db_print("Writing "+std::to_string(local_states.size())+" states to file\n");

        for (auto& state : local_states){
            buf.push_back(state);
            if(buf.size() >= block_size){
                // flush to file
                size_t written = fwrite(buf.data(), sizeof(Uint128), buf.size(), f);
                if (written != buf.size()) throw std::runtime_error("save: incomplete fwrite");
                buf.clear();
            }
        }
        // flush the last states
        size_t written = fwrite(buf.data(), sizeof(Uint128), buf.size(), f);
        if (written != buf.size()) throw std::runtime_error("save: incomplete fwrite");

        fclose(f);
        return out_file.string();
    }



    void finalise_shards(){
        auto out_file = get_out_file();
        std::string my_name(out_file);

        static const int filename_bufsize = 4096;
        char* sendbuf = new char[filename_bufsize];
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


        if (my_rank == 0) {
        
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
    }

    size_t insert_states(state_uset_t& states_to_insert){
        // tries to add states_to_insert to all ranks, 
        // returning the total number of inserted states (global)
        // and leaving only the new states in states_to_insert

        // stage 1: distribute between buckets
        for (auto& b : buckets){
            b.clear();
        }
        for (auto& state : states_to_insert) {
            buckets[hash_f(state) % world_size].push_back(state);
        }

        // MPI broadcast the sizes;
        std::vector<int> sendcounts;
        std::vector<int> recvcounts(world_size);
        for (int r=0; r<world_size; r++){
            sendcounts.push_back(buckets[r].size());
        }
        MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1,
                     MPI_INT, MPI_COMM_WORLD);

        // allocate memory to reveive the states
        size_t n_received = std::reduce(recvcounts.begin(), recvcounts.end(), 0);
        std::vector<state_t> recvbuf(n_received);

        sendbuf.clear();
        sendbuf.reserve(states_to_insert.size());
        // stage 2: move into buffer (states_to_insert);
        for (int r=0; r<world_size; r++){
            sendbuf.insert(sendbuf.end(), buckets[r].begin(), buckets[r].end());
        }

        auto send_displ = close_packed_displ(sendcounts); 
        auto recv_displ = close_packed_displ(recvcounts);
        MPI_Alltoallv(sendbuf.data(), sendcounts.data(), send_displ.data(),
                      get_mpi_type<state_t>(),
                      recvbuf.data(), recvcounts.data(), recv_displ.data(),
                      get_mpi_type<state_t>(), MPI_COMM_WORLD);
    
        // Repurpose states_to_insert
        states_to_insert.clear();
    
        // GET IN THE HOLE
        size_t l_insertions=0;
        for (auto& s : recvbuf){
            auto [_, inserted] = local_states.insert(s);
            if (inserted){
                l_insertions++;
                states_to_insert.insert(s);
            }
        }
        size_t g_insertions=0;
        MPI_Allreduce(&l_insertions, &g_insertions, 1, get_mpi_type<size_t>(),
                MPI_SUM, MPI_COMM_WORLD);

        return g_insertions;
    }

    int rank_of_state(const state_t& psi){
        // hash the state and wrap
        return hash_f(psi) % world_size;
    }

protected:
    int world_size;
    int my_rank;
    int n_states;
    std::filesystem::path workdir;
    std::string job_tag;

    RankLogger db_log;

    std::ostream& db_print(const std::string& msg=""){
        return db_log << msg;
    }

    Uint128Hash hash_f;

    std::vector<std::vector<state_t>> buckets;

    state_uset_t local_states;
    std::vector<state_t> sendbuf;
	std::vector<SymbolicPMROperator> opset;
};


void distribute_initial_work(ZBasisBST::state_t init){

}


// Sketch of the algo:
// Starting with some collection of seed states,
// Apply all possible rings and insert of any new states are found.
// All ranks know all ring ops.
// Need a heuristic to decide which rnak owns what state: Sufficient to use the top N bits?
// Repeat until we stop finding new states.
void constr_explorer_mpi::build_states(ZBasisBST::state_t init) {
	state_uset_t tmp;

	state_uset_t prev_set;

    if (my_rank == 0){
        prev_set.insert(init);
        insert_states(prev_set);
    }

	size_t iter_no = 0;

	size_t insertions = 0;
	std::cout << std::endl;

	do {
        // prev_set contains the states on _this_ rank

        // For each state in the previous set of new states, apply all possible operators
		for (const auto& psi : prev_set){
			for (const auto &o : opset) {
                auto chi = psi;
				if (o.applyState(chi) != 0) {
#ifndef NDEBUG
                    std::cout << "Generated psi=";
                    printHex(std::cout, chi) << "\n";
#endif
					tmp.insert(chi);
				}
			}
		}
        
        prev_set.clear();
		// insert the new keys
		insertions = insert_states(tmp); 
        std::swap(tmp, prev_set);
		iter_no++;

		std::cout << "[rank "<<my_rank<<"] Iteration " << iter_no << ", " << insertions << " insertions, "
			<< "basis dim " << local_states.size() 
			<< std::endl;
	} while (insertions != 0);
}

int main (int argc, char *argv[]) {
	if (argc < 4) {
		printf("USAGE: %s <latfile: json> seed_state tmp_outdir\n", argv[0]);
		return 1;
	}

	std::string infilename(argv[1]);
    std::filesystem::path workdir(argv[3]);
    if(!std::filesystem::exists(workdir)){
        throw std::runtime_error("work dir " + workdir.string() + "does not exist");
    }

	Uint128 seed_state;
	size_t nchar = std::sscanf(argv[2], "0x%" PRIx64 ",0x%" PRIx64, &seed_state.uint64[1], &seed_state.uint64[0]);
	if (nchar != 2){
		cerr<<"Failed to parse argv[2] as a seed state: got "<<argv[2]<<endl;
	}
    std::cout << "Loaded seed state";
    printHex(std::cout, seed_state) << "\n";


	std::string ext = ".0";
	ext += (argc >= 5) ? argv[4] : ".basis";

	auto outfilename=as_basis_file(infilename, ext );

	ifstream ifs(infilename);
	json jdata = json::parse(ifs);
	ifs.close();


    const auto& atoms = jdata.at("atoms");
    std::vector<int> top_sector;

    std::map<std::string, IntDefaultedToZero > sector;
    for (size_t i=0; i<atoms.size(); i++){
        sector[atoms[i].at("sl")]()+= (readbit(seed_state, i) ? 1 : 0 );
    }

    for (const auto& [k, v] : sector){
        std::cout<<k<<"-> "<<v()<<"\n";
    }

    std::filesystem::path inpath(infilename);

    MPI_Init(&argc, &argv);

	constr_explorer_mpi L(jdata, workdir, inpath.stem().string() +"-"+ to_string(seed_state)
            );
	

	printf("Building states...\n");
	L.build_states(seed_state);

    L.write_shard();
    L.finalise_shards();
    MPI_Finalize();

	return 0;
}


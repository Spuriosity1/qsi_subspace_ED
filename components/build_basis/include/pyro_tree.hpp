#pragma once
#include "basis_io.hpp"
#include "basis_io_h5.hpp"
#include "bittools.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include "tetra_graph_io.hpp"
#include <array>
#include <cstdio>
#include <thread>
#include <queue>
#include <stack>
#include <vector>
#include <filesystem>

#include "shard.hpp"


//#pragma pack(push, 1)
struct vtree_node_t {
	Uint128 state_thus_far;
	unsigned curr_spin;
	unsigned num_spinon_pairs;
	// curr_spin is the bit ID of the rightmost unknown spin
	// i.e. (1<<curr_spin) & state_thus_far is guaranteed to be 0
};

//#pragma pack(pop)


inline void print_node(std::ostream& os, const vtree_node_t& node){
    printHex(os, node.state_thus_far) << " [spin " << node.curr_spin<<"]\n";
}


typedef std::array<int, 4> global_sz_sector_t;


template<typename T>
struct vstack : public std::vector<T> {
    T top() const { return this->back(); }
    T pop() { 
        T top = this->back();
        this->pop_back();
        return top;
    }
    void push(const T& x) { this->push_back(x); }
};


struct lat_container {
	lat_container(const lattice& _lat, unsigned num_spinon_pairs)
		: num_spinon_pairs(num_spinon_pairs), lat(_lat) {
			//auto natoms = data.at("atoms").size();
			auto natoms = _lat.spins.size();
			masks.resize(natoms+1);
			for (size_t i = 0; i < natoms+1; i++) {
				masks[i] = make_mask(i);
			}
		}

        // state is only initialised up to (but not including) bit 1<<idx
	// returns possible states of state&(1<<idx)

	// return values:
	// 0b00 -> no spin state valid
	// 0b01 -> spin down (0) state valid
	// 0b10 -> spin up (1) state valid
	// 0b11 -> both up and down valid
	char possible_spin_states(const vtree_node_t& curr) const;
	//char possible_spin_states(const Uint128& state, unsigned idx) const ;

	const unsigned num_spinon_pairs;

    using cust_stack = vstack<vtree_node_t>;

	void fork_state(cust_stack& to_examine);
	void fork_state(std::queue<vtree_node_t>& to_examine);

	const lattice& lat;
protected:
	std::vector<Uint128> masks; // bitmasks filled by make_mask
    std::string h5_dset_name="basis";
};

struct lat_container_with_sector : public lat_container {
    lat_container_with_sector(const lattice& _lat, unsigned num_spinon_pairs)
        : lat_container(_lat, num_spinon_pairs){
        }

    void set_sector(const std::vector<int>& _sector){
        make_sl_masks(_sector);
        std::ostringstream oss("basis");
        char delim='_';
        for (auto s : _sector){
            oss << delim << s;
            delim='.';
        }
        h5_dset_name = oss.str();
    }

	void fork_state(cust_stack& to_examine);
	void fork_state(std::queue<vtree_node_t>& to_examine);

    char possible_spin_states(const vtree_node_t& curr) const;


    protected:

    std::string h5_dset_name="basis";

    void make_sl_masks(const std::vector<int>& sector){ 
        int max_sl=0;
        for (const auto& s : this->lat.spins){
            max_sl = std::max(max_sl, s.sl);
        }
        if (max_sl+1 != static_cast<int>(sector.size())){
            throw std::logic_error("Bad sector secification: expected " + std::to_string(max_sl+1) + " integers");
        }
        sl_masks.resize(max_sl+1);
        for (size_t si=0; si<lat.spins.size(); si++){
            auto spin = lat.spins[si];
            or_bit(sl_masks[spin.sl].first, si);
        }
        for (int mu=0; mu<=max_sl; mu++){
            sl_masks[mu].second = sector[mu];
        }
    }
	std::vector<std::pair<Uint128, int>> sl_masks; // pairs such that state &sl_masks == integer
};



template<typename LatContainer>
requires std::derived_from<LatContainer, lat_container>
struct pyro_vtree : public LatContainer {
	pyro_vtree(const lattice& lat, unsigned num_spinon_pairs) :
		LatContainer(lat, num_spinon_pairs) {
			is_sorted = false;
		}

    char possible_spin_states(const vtree_node_t& curr) const {
        return static_cast<const LatContainer*>(this)->possible_spin_states(curr);
    }

	void build_state_tree();
	void sort();
	// Applies bittools::permute to all elements of the basis
	void permute_spins(const std::vector<size_t>& perm);

	void write_basis_csv(const std::string &outfilename) {
        this->sort();
        std::cout<<"Outfile: "<<outfilename<<std::endl;
        basis_io::write_basis_csv(state_list, outfilename);
    }

    void write_basis_hdf5(const std::string& outfilename) {
        this->sort();
        std::cout<<"Outfile: "<<outfilename<<std::endl;
        basis_io::write_basis_hdf5(this->state_list, outfilename, this->h5_dset_name.c_str());
    }
protected:
	void save_state(const Uint128& state) {
			state_list.push_back(state);
	}
	// Repository of ice states for perusal
	std::vector<Uint128> state_list;

	bool is_sorted;

	// auxiliary variable for printing
	unsigned counter = 0;
};


template<typename LatContainer>
requires std::derived_from<LatContainer, lat_container>
struct pyro_vtree_parallel : public LatContainer {
	pyro_vtree_parallel(const lattice &lat, unsigned num_spinon_pairs, 
			unsigned n_threads = 1)
		: LatContainer(lat, num_spinon_pairs), n_threads(n_threads) {
		is_sorted = false;
		}


    char possible_spin_states(const vtree_node_t& curr) const {
        return static_cast<const LatContainer*>(this)->possible_spin_states(curr);
    }

	void build_state_tree();
	void sort();

	// Applies bittools::permute to all elements of the basis
	void permute_spins(const std::vector<size_t>& perm);

	void write_basis_csv(const std::string& outfilename) {
        this->sort();
        for (size_t i=1; i<state_set.size(); i++){
            if(state_set[i].size() != 0){
                throw std::logic_error("Error in write_basis_csv - basis was not sorted properly");
            }
        }
        std::cout<<"Outfile: "<<outfilename<<std::endl;
        basis_io::write_basis_csv(state_set[0], outfilename);
    }
	void write_basis_hdf5(const std::string& outfilename){
        this->sort();
        for (size_t i=1; i<state_set.size(); i++){
            if(state_set[i].size() != 0){
                throw std::logic_error("Error in write_basis_hdf5 - basis was not sorted properly");
            }
        }
        std::cout<<"Outfile: "<<outfilename<<std::endl;
        basis_io::write_basis_hdf5(state_set[0], outfilename);
    }


protected:
	void _build_state_dfs(lat_container::cust_stack &node_stack, unsigned thread_id,
			unsigned long max_stack_size = (1ul << 40));
	void _build_state_bfs(std::queue<vtree_node_t>& node_stack, 
		unsigned long max_queue_len);
	unsigned n_threads;


//    template <typename StackT>
//    void rebalance_stacks(std::vector<StackT>& stacks);

	bool is_sorted;

	size_t n_states() const {
		size_t acc=0;
		for (auto v : state_set){
			acc += v.size();
		}
		return acc;
	}

	// first index is the thread ID
	std::vector<std::vector<Uint128>> state_set;
	std::vector<std::thread> threads;
	std::vector<lat_container::cust_stack> job_stacks;

    static constexpr unsigned INITIAL_DEPTH_FACTOR = 5;
};





// similar to pyro_vtree_parallel, but only does the search part
struct par_searcher : public lat_container {
    par_searcher(const lattice& lat, unsigned num_spinon_pairs,
            unsigned n_threads, const std::vector<size_t>& perm
            )
        : lat_container(lat, num_spinon_pairs), perm(perm), n_threads(n_threads)
    {}

    void initialise_shards(
            const std::string &out_dir, const std::string &job_tag,
            size_t buf_entries = (1<<20)
            ){
        // ensure that out_dir exists
        std::filesystem::create_directories(out_dir);

        std::string shard_prefix;
        shard_prefix = out_dir + "/shard-" + job_tag + "-";
        shards.clear();
        shards.reserve(n_threads);
        for (unsigned t = 0; t < n_threads; ++t) {
            auto path = shard_prefix + std::to_string(t) + ".bin";
            shards.emplace_back(new ShardWriter(path, buf_entries));
        }
        this->job_tag = job_tag;
        this->out_dir = out_dir;

    }

    void build_state_tree();

    void finalise_shards(){
        for (auto &w : shards) {
            if (w) w->finalize(true);
        }
 // construct manifest file
        nlohmann::json manifest;
        manifest["job_tag"] = job_tag;
        manifest["n_shards"] = shards.size();
        manifest["shards"] = nlohmann::json::array();

         std::string shard_prefix = out_dir + "/shard-" + job_tag + "-";
        for (unsigned t = 0; t < shards.size(); ++t) {
            manifest["shards"].push_back(shard_prefix + std::to_string(t) + ".bin.done");
        }

        // write manifest to file
        std::string manifest_path = out_dir + "/manifest-" + job_tag + ".json";
        std::ofstream ofs(manifest_path);
        ofs << manifest.dump(2) << std::endl;
        ofs.close();
    }

//    const std::string shard_prefix;

protected:
    const std::vector<size_t>& perm;
    std::string job_tag, out_dir;

	void _build_state_dfs(cust_stack &node_stack, unsigned thread_id,
			unsigned long max_stack_size = (1ul << 40));
	void _build_state_bfs(std::queue<vtree_node_t>& node_stack, 
		unsigned long max_queue_len);

	unsigned n_threads;
    std::vector<std::unique_ptr<ShardWriter>> shards;
	std::vector<std::thread> threads;
	std::vector<cust_stack> job_stacks;

    static constexpr unsigned INITIAL_DEPTH_FACTOR = 5;

};



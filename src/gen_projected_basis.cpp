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
#include "operator.hpp"


using namespace std;
using json=nlohmann::json;


struct constr_explorer : public ZBasis {
	constr_explorer(const nlohmann::json& data);

	void build_states(ZBasis::state_t init);

	void sort() {
		std::sort(states.begin(), states.end());
		state_to_index.clear();
		for (idx_t i=0; i<states.size(); i++){
			state_to_index[states[i]] = i;
		}

	}

	void write_basis_csv(const std::string &outfilename) {
		this->sort();
		basis_io::write_basis_csv(states, outfilename);
	}
    void write_basis_hdf5(const std::string& outfilename) {
		this->sort();
		basis_io::write_basis_hdf5(states, outfilename);
	}

	protected:
	std::vector<SymbolicPMROperator> opset;
};

constr_explorer::constr_explorer(const nlohmann::json& jdata){

	// strategy: apply ring exchanges, and prune
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


// Sketch of the algo:
// Starting with some collection of seeds states,
// Apply all possible rings and insert of any new states are found.
// Repeaat until we stop finding new states.
// Simple enough to parallelise
void constr_explorer::build_states(ZBasis::state_t init) {
	std::vector<ZBasis::state_t> tmp;
	std::vector<ZBasis::state_t> prev_set = {init};

	size_t iter_no = 0;

	size_t insertions = 0;
	std::cout << std::endl;

	do {
		tmp.resize(0);
		for (auto& psi : prev_set){
			for (const auto &o : opset) {
				if (o.applyState(psi) != 0) {
					tmp.push_back(psi);
				}
			}
		}
		// insert the new keys
		insertions = this->insert_states(tmp, prev_set);
		iter_no++;

		std::cout << "Iteration " << iter_no << ", " << insertions << " insertions, "
			<< "basis dim " << states.size() 
			<< std::endl;
	} while (insertions != 0);
}

int main (int argc, char *argv[]) {
	if (argc < 3) {
		printf("USAGE: %s <latfile: json> seed_state (<ext>=.basis)\n", argv[0]);
		return 1;
	}

	std::string infilename(argv[1]);

	Uint128 seed_state;
	size_t nchar = std::sscanf(argv[2], "0x%" PRIx64 "0x%" PRIx64, &seed_state.uint64[1], &seed_state.uint64[0]);
	if (nchar != 2){
		cerr<<"Failed to parse argv[2] as a seed state: got "<<argv[2]<<endl;
	}


	std::string ext = ".0";
	ext += (argc >= 4) ? argv[3] : ".basis";

	auto outfilename=as_basis_file(infilename, ext );

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	constr_explorer L(data);
	
	printf("Building states...\n");
	L.build_states(seed_state);

	L.write_basis_csv(outfilename);
	L.write_basis_hdf5(outfilename);

	return 0;
}


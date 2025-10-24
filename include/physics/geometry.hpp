#pragma once

#include <vector>
#include "operator.hpp"
#include <nlohmann/json.hpp>

std::tuple<std::vector<SymbolicPMROperator>,std::vector<SymbolicPMROperator>,
    std::vector<int>> 
get_ring_ops(const nlohmann::json& jdata, bool incl_partial=false);

/*
std::pair<std::vector<SymbolicOpSum<double>>,
    std::vector<int>> 
get_vol_ops(
        const nlohmann::json& jdata,
        const std::vector<SymbolicPMROperator>& ring_list
);
*/

std::vector<SymbolicOpSum<double>>
get_partial_vol_ops(
        const nlohmann::json& jdata,
        const std::vector<SymbolicPMROperator>& ring_list,
        int sl
);



std::tuple<std::vector<SymbolicPMROperator>,std::vector<SymbolicPMROperator>,
    std::vector<int>> 
get_ring_ops(
const nlohmann::json& jdata, bool incl_partial) {

    std::vector<SymbolicPMROperator> op_list;
    std::vector<SymbolicPMROperator> op_H_list;
    std::vector<int> sl_list;



	for (const auto& ring : jdata.at("rings")) {
		std::vector<int> spins = ring.at("member_spin_idx");

        if (!incl_partial && spins.size() != 6){
            continue;
        }

		std::vector<char> ops;
		std::vector<char> conj_ops;
		for (auto s : ring.at("signs")){
			ops.push_back( s == 1 ? '+' : '-');
			conj_ops.push_back( s == 1 ? '-' : '+');
		}
		
		int sl = ring.at("sl").get<int>();
		auto O   = SymbolicPMROperator(     ops, spins);
		auto O_h = SymbolicPMROperator(conj_ops, spins);
        op_list.push_back(O);
        op_H_list.push_back(O_h);
        sl_list.push_back(sl);
	}
    return std::make_tuple(op_list, op_H_list, sl_list);
}




using optype=SymbolicOpSum<double>;


/*
std::pair<std::vector<optype>,
    std::vector<int>> 
get_vol_ops(
        const nlohmann::json& jdata,
        const std::vector<SymbolicPMROperator>& ring_list
) {

    std::vector<optype> op_list;
    std::vector<int> sl_list;
    
	for (const auto& vol : jdata.at("vols")) {
		std::vector<int> plaqi = vol.at("member_plaq_idx");
        SymbolicPMROperator volOp("");

        for (auto J : plaqi){
            volOp *= ring_list[J];
        } 
	    
        op_list.push_back(volOp);
        sl_list.push_back(vol.at("sl").get<int>());

	}
    return std::make_pair(op_list, sl_list);
}
*/


const int perm3[6][3] = {
    {0,1,2},
    {1,2,0},
    {2,0,1},
    {0,2,1},
    {2,1,0},
    {1,0,2}
};

std::vector<optype>
get_partial_vol_ops(
        const nlohmann::json& jdata,
        const std::vector<SymbolicPMROperator>& ring_list,
        int missing_plaq_sl
) {

    std::vector<optype> op_list;
    std::vector<int> sl_list;
    
	for (const auto& vol : jdata.at("vols")) {
		std::vector<int> plaqi = vol.at("member_plaq_idx");
        
        std::vector<SymbolicPMROperator> ops;
        for (int i=0; i<4; i++) {
            if (i != missing_plaq_sl){
                int J = plaqi[i];
                ops.push_back(ring_list[J]);
            }
        } 
        assert( ops.size() == 3);
        // Symmetrise
        //
        SymbolicOpSum<double> OOO;
        for (int i=0; i<6; i++){
            SymbolicPMROperator volOp("");
            for (int j=0; j<3; j++){
                volOp *= ops[perm3[i][j]];
            }
            OOO += volOp; 
        } 
        op_list.push_back(OOO);
        sl_list.push_back(vol.at("sl").get<int>());

	}
    return op_list;
}







#pragma once
#include<string>

inline std::string as_basis_file(const std::string& input_jsonfile){
	return input_jsonfile.substr(0,input_jsonfile.find_last_of('.'))+".basis.csv";
}

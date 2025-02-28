#pragma once
#include<string>
#include <hdf5.h>
#include <vector>
#include <stdexcept>
#include <sstream>

inline std::string as_basis_file(const std::string& input_jsonfile, const std::string& ext=".basis" ){
	return input_jsonfile.substr(0,input_jsonfile.find_last_of('.'))+ext;
}


class HDF5Error : public std::runtime_error {
public:
    HDF5Error(hid_t file_id, hid_t dataspace_id, hid_t uint128_datatype, hid_t dataset_id, const std::string& message)
        : std::runtime_error(formatMessage(file_id, dataspace_id, uint128_datatype, dataset_id, message)) {}

private:
    static std::string formatMessage(hid_t file_id, hid_t dataspace_id, hid_t uint128_datatype, hid_t dataset_id, const std::string& message) {
        std::ostringstream oss;
        oss << "HDF5 Error: " << message << "\n"
            << "  File ID: " << file_id << "\n"
            << "  Dataspace ID: " << dataspace_id << "\n"
            << "  Datatype ID: " << uint128_datatype << "\n"
            << "  Dataset ID: " << dataset_id;
        return oss.str();
    }
};


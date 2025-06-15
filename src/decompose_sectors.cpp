#include <iostream>
#include <basis_io.hpp>

int main (int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: "<<std::string(argv[0])<<" <*.basis.h5>\n";
        return 1;
    }
    auto basis_list = basis_io::read_basis_hdf5(argv[1]);


    return 0;
}

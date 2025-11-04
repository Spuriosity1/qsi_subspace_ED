#include "bittools.hpp"

static const int bucket_bits=8;

struct node_t;

struct node_t {
    node_t* leaves[1<<bucket_bits];
};

class custom_map {
    std::vector<Uint128> v;

    node_t* root;

    public:
    int64_t find_index(const Uint128& x);
};

int main (int argc, char *argv[]) {

    return 0;
}

#include "bittools.hpp"
#include <iostream>
#include <vector>
#include "operator.hpp"
#include "group_theory.hpp"

size_t total_failures;

class AssertException : public std::runtime_error {
public:
    explicit AssertException(const std::string& what_arg)
        : std::runtime_error(what_arg) { total_failures++; }
};


#define ASSERT_EQ(x, y, msg)                                                     \
    do {                                                                         \
        auto _ax = (x);                                                          \
        auto _ay = (y);                                                          \
        if (!(_ax == _ay)) {                                                     \
            std::ostringstream _assert_eq_os;                                    \
            _assert_eq_os << "ASSERT_EQ failed: " << msg                         \
                           << "\n  Expected: " << #x << " == " << #y             \
                           << "\n  Actual:   " << _ax << " vs " << _ay           \
                           << "\n  Location: " << __FILE__ << ":" << __LINE__;   \
            throw AssertException(_assert_eq_os.str());                          \
        }                                                                        \
    } while (0)


#define EXPECT_EQ(x, y, msg)                                                     \
    do {                                                                         \
        auto _ax = (x);                                                          \
        auto _ay = (y);                                                          \
        if (!(_ax == _ay)) {                                                     \
            cerr << "ASSERT_EQ failed: " << msg                         \
                           << "\n  Expected: " << #x << " == " << #y             \
                           << "\n  Actual:   " << _ax << " vs " << _ay           \
                           << "\n  Location: " << __FILE__ << ":" << __LINE__;   \
            total_failures++;                          \
        }                                                                        \
    } while (0)

using state_t = Uint128;
using perm_t = PermutationGroup<state_t>::perm_t;
using PermGroup = PermutationGroup<state_t>;
using namespace std;

void ensure_component_consistent(const PermGroup& G, state_t psi_0){
    G.make_representative(psi_0);

    cout<<"\n";
    for (int iG=0; iG<G.size(); iG++){
        state_t chi = permute(psi_0, G[iG]);
        cout<<chi<<" -> ";
        G.make_representative(chi);
        cout<<chi << endl;
        EXPECT_EQ(chi, psi_0, "Representative making");
    }
}
    


int main(int argc, char** argv) {
    const static int N=128;
    static_assert(N<=8*sizeof(state_t), "N is wider than width of state_t");

    perm_t transl; 

    for (int i=0; i<N; i++)
        transl.push_back((i+1) % N);

    std::vector<perm_t> generators {transl};

    PermGroup G(generators);


    
    std::vector<state_t> psi_0_set = {1, 3, 5, 9, 0xA, 0xAA};
    for (const auto& psi_0: psi_0_set){
//        cout << "[component find] testing representatives of " << psi_0 <<endl;
//        ensure_component_consistent(G, psi_0);
    }

    cout<<"Done! ("<<total_failures<< " failures)"<<endl;
    return total_failures;

}

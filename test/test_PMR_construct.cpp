#include "bittools.hpp"
#include <iostream>
#include <vector>
#include "operator.hpp"


int test_counter = 0;
int fail_counter = 0;

void require(bool condition, const std::string& msg) {
    ++test_counter;
    if (!condition) {
        ++fail_counter;
        std::cerr << "FAILED: " << msg << std::endl;
    }
}

template<typename T>
void require_eq(T A, T B, const std::string& msg) {
    ++test_counter;
    if (!(A == B)) {
        ++fail_counter;
        std::cerr << "FAILED: " << msg << std::endl;
        std::cerr <<"Expected " << A <<" == "<< B <<std::endl;
    }
}
    
void check_equiv(const std::vector<char>& Aop, const std::vector<char>& Bop,
                 int expected_sign, Uint128 initial, int site = 0) {

    ZBasisBase::state_t sa{initial}, sb{initial};
    std::vector<int> idsA, idsB;
    for (size_t i = 0; i < Aop.size(); ++i) idsA.push_back(site);
    for (size_t i = 0; i < Bop.size(); ++i) idsB.push_back(site);

    SymbolicPMROperator A(Aop, idsA);
    SymbolicPMROperator B(Bop, idsB);

    int signA = A.applyState(sa);
    int signB = B.applyState(sb);

    std::string label = std::string(Aop.begin(), Aop.end()) + " == " +
                        (expected_sign == -1 ? "-" : "") +
                        std::string(Bop.begin(), Bop.end());

    require(A==expected_sign*B, label+ " [op mismatch]");
    require_eq(sa.uint64[site/64], sb.uint64[site/64], label + " [state mismatch]");
    require_eq(signA, expected_sign * signB, label + " [sign mismatch]");
}


void check_equiv_str(const std::string& A, const std::string& B, int expected_sign, Uint128 initial) {
    ZBasisBase::state_t sa{initial}, sb{initial};
    SymbolicPMROperator opA(A), opB(B);
    int signA = opA.applyState(sa);
    int signB = opB.applyState(sb);
    require_eq(sa.uint64[0], sb.uint64[0], A + " == " + B + " [lo state mismatch]");
    require_eq(sa.uint64[1], sb.uint64[1], A + " == " + B + " [hi state mismatch]");
    require_eq(signA, expected_sign * signB, A + " == " + B + " [sign mismatch]");
}

void check_zero(const std::string& A, Uint128 initial) {
    ZBasisBase::state_t s{initial};
    SymbolicPMROperator op(A);
    require(op.applyState(s) == 0, A + " == 0");
}

void check_application(const std::string& O, int expected_sign, Uint128 initial, Uint128 final) {

    // basic sanity check
    SymbolicPMROperator Op(O);
    ZBasisBase::state_t s{initial};
    int sign1 = Op.applyState(s); 

    require_eq(s.uint64[0], final.uint64[0], O + " [ lo expected final state mismatch]");
    require_eq(s.uint64[1], final.uint64[1], O + " [ hi expected final state mismatch]");
    require_eq(sign1, expected_sign, O + " [sign mismatch]");
}


void test_apply() {

    Uint128 up =   {0,1ULL};     // site 0 is up
    Uint128 up64 = {1ULL, 0};    // site 64 is up
    Uint128 down = 0;             // all sites down

    // Basic sanity test
    check_application("0+", 1, down, up);
    check_application("0-", 1, up, down);
    check_application("64-", 1, up64, down);

    check_application("0X", 1, down, up);
    check_application("0X", 1, up, down);

    check_application("0Z", 1, up, up);
    check_application("0Z", -1, down, down);

    auto psi = up | 7263729ULL;
    check_application("0Z", 1, psi, psi);

    check_application("64Z",  1, up64, up64);
    check_application("64Z", -1, down, down);

    Uint128 L = {0, 0b111010};
    Uint128 R = {0, 0b110101};
    // Ring application
    check_application("0+ 1- 2+ 3-", 1, L, R);
    check_application("0- 1+ 2- 3+", 1, R, L);
                                  
    // Commutation / Anticommutation
    check_equiv_str("0Z 1X", "1X 0Z", +1, up);
    check_equiv_str("0Z 0X", "0X 0Z", -1, up);
    check_equiv_str("0Z 1Z", "1z 0z", +1, up);
    check_equiv_str("1Z 1Z", "", +1, up);
    check_equiv_str("0X 0X", "", +1, up);

    // Z with +/-
    check_equiv_str("1+ 1Z", "1Z 1+", -1, up);
    check_equiv_str("1- 1Z", "1Z 1-", -1, down);
    check_equiv_str("1+ 1Z", "1Z 1+", -1, up);  // +Z == -Z+

    // X with +/-
    check_equiv_str("1+ 1X", "1+ 1-", +1, down);
    check_equiv_str("1- 1X", "1- 1+", +1, up);
    check_equiv_str("1X 1+", "1- 1+", +1, up);
    check_equiv_str("1X 1-", "1+ 1-", +1, down);

    // Double application of ++ or -- should vanish
    check_zero("1+ 1+", up);
    check_zero("1- 1-", down);
    check_zero("0+ 0+", up);
    check_zero("0- 0-", down);

    // Z X Z = -X
    check_equiv_str("0Z 0X 0Z", "0X", -1, up);
    // X Z X = -Z
    check_equiv_str("0X 0Z 0X", "0Z", -1, up);

    // + X + = +-+ = +
    check_equiv_str("0+ 0X 0+", "0+", +1, down);
    check_equiv_str("0+ 0X 0+", "0+ 0- 0+", +1, down);

    check_equiv_str("0+ 0X 0+", "0+", +1, up);
    check_equiv_str("0+ 0X 0+", "0+ 0- 0+", +1, up);

    // - X - = -+- = -
    check_equiv_str("0- 0X 0-", "0-", +1, up);
    check_equiv_str("0- 0X 0-", "0- 0+ 0-", +1, up);

    // +Z == -Z+
    check_equiv_str("0+ 0Z", "0Z 0+", -1, up);
    check_equiv_str("64+ 64Z", "64Z 64+", -1, up64);

    // X+ == -+
    check_equiv_str("0X 0+", "0- 0+", +1, up);
    check_equiv_str("0X 0+", "0- 0+", +1, up);

    // X- == +-
    check_equiv_str("0X 0-", "0+ 0-", +1, down);


}

void test_multiply() {
    SymbolicPMROperator a("0Z 1X");
    SymbolicPMROperator b("1Z");
    SymbolicPMROperator ab("0Z 1X 1Z");

    SymbolicPMROperator c = a * b;
    require(c == ab, "a * b matches expected");
    require((a *= b) == c, "a *= b produces same result");

    // test killing condition
    SymbolicPMROperator kill1("0+");
    SymbolicPMROperator kill2("0-");
    SymbolicPMROperator proj1 = kill1 * kill2;
    require_eq(proj1.get_sign() , 1, "+ * - gives projector");

    SymbolicPMROperator dead1 = kill1 * kill1;
    require_eq(dead1.get_sign() , 0, "+ * + gives projector");

    SymbolicPMROperator dead2 = kill2 * kill2;
    require_eq(dead2.get_sign() , 0, "+ * + gives projector");


    SymbolicPMROperator O1("0+ 1- 2+ 3-");
    SymbolicPMROperator O2("0- 1+ 4- 5+");
    SymbolicPMROperator O1O2("0+ 0- 1- 1+ 2+ 3- 4- 5+");
    require(O1*O2 == O1O2, "O1 * O2 behaves as expected");
    
}




// === Entry point with CLI ===

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "all") {
        test_apply();
        test_multiply();
    } else if (std::string(argv[1]) == "apply") {
        test_apply();
    } else if (std::string(argv[1]) == "multiply") {
        test_multiply();
    } else {
        std::cerr << "Unknown test group: " << argv[1] << "\n";
        return 1;
    }

    std::cout << "Tests passed: " << (test_counter - fail_counter)
              << " / " << test_counter << "\n";
    return fail_counter == 0 ? 0 : 1;
}

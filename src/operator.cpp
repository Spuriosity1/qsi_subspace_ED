#include "operator.hpp"




idx_t ZBasis::idx_of_state(const comp_basis_state_t& state) const {
	// do binary search, throw if not found

	idx_t low = 0;
	idx_t high = states.size()-1;

	while (low <= high) {
		idx_t mid = (high - low)/2+ low;
		if (states[mid] == state) return mid;

		if (states[mid] < state)
			low = mid + 1;
		else
			high = mid -1;
	}
	// not found
	throw state_not_found_error(state);
}

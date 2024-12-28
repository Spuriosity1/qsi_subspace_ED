#include "pyro_tree.hpp"
#include <queue>
#include <stack>






char pyro_tree::possible_spin_states(const Uint128& state, int idx) const {
	// state is only initialised up to (but not including) bit 1<<idx
	// returns possible states of state&(1<<idx)

	// return values:
	// 0b00 -> no spin state valid
	// 0b01 -> spin down (0) state valid
	// 0b10 -> spin up (1) state valid
	// 0b11 -> both up and down valid
	char res=0b11;

	Uint128 state_new = state; // new spin is already a 0
	assert( !readbit(state, idx) );
	assert( idx < spins.size() );

	const auto known_mask = make_mask(idx);


	for (__uint128_t updown=0; updown<2; updown++){
		if (updown == 1){
			or_bit(state_new, idx);
		}

		for (auto t : spins[idx].tetra_neighbours){
			// calculate the partial tetra charges
			// NOTE: state_new is all zeros for bits > idx
			int Q = popcnt_u128( state_new & t->bitmask );
			// we know the state of all previous bits, and the one we just set
			int num_known_spins = popcnt_u128( t->bitmask & known_mask )+1;

			int num_spins = t->member_spin_ids.size();
			int max_spins_up = (num_spins +1) /2;
			int min_spins_up = (num_spins   ) /2;
			int num_unknown_spins = num_spins - num_known_spins;

			if (Q + num_unknown_spins < min_spins_up || Q > max_spins_up){
				// Q is inconsistent with an ice rule
				res &= ~(1<<updown);
				break; // no point checking anything else
			}
		}

	}
	return res;
}



void pyro_tree::print_state_tree() const {
	// print breadthwise
	std::queue<node_t*> next;
	next.push(root);

	while(next.size() > 0){
		node_t* curr = next.front();
		next.pop();
		_print_node(curr);
		if(curr->leaf[0]) next.push(curr->leaf[0]);
		if(curr->leaf[1]) next.push(curr->leaf[1]);
	}
}

void pyro_tree::_print_node(node_t* n) const
{
	printf("%16p |",static_cast<void*>(n));
	for (int i=0; i<n->spin_idx; i++){ printf(" "); }
	printf("0x%016llx%016llx --> ", n->state_thus_far.uint64[1],n->state_thus_far.uint64[0]);
	printf("0: %16p ", static_cast<void*>(n->leaf[0]));
	printf("1: %16p \n", static_cast<void*>(n->leaf[1]));
}


pyro_tree::~pyro_tree(){
	// delete the tree
	if (this->root == nullptr){
		return; // tree already deleted!
	}
	std::stack<node_t*> to_delete;
	to_delete.push(root);
	while (!to_delete.empty()){
		node_t* curr = to_delete.top();
		to_delete.pop();

		for (int i=0; i<2; i++){
			if(curr->leaf[i]){ to_delete.push(curr->leaf[i]); }
		}
		delete curr;
	}

}

void pyro_tree::build_state_tree(){
	this->root = new node_t;
	root->spin_idx=0;
	root->state_thus_far.uint128=0;

	/*
	// for aesthetics only
	this->tree_size.resize(this->spins.size());
	std::fill(tree_size.begin(), tree_size.end(), 0);
	this->tree_size[0] = 1;
	*/

	// start the recursion
	_build_state_tree(root);
}

void print_tree_state(const std::vector<unsigned> tree_size){
	printf("Tree state: ");
	for (auto t : tree_size){
		printf("|%x",t);
	}
	printf("\n");
}

void pyro_tree::_build_state_tree(node_t* node){
	if (node->spin_idx == spins.size() ){
		// end of the line, we have a working ice state (no children)
		states_2I2O.push_back(node->state_thus_far);
		return;
	}

	char options = possible_spin_states(node->state_thus_far, node->spin_idx);
#if VERBOSITY > 2
	printf("%p | index %d poss_states %d\n", static_cast<const void*>(node), node->spin_idx,options);
#endif
	for (int i=0; i<2; i++){
		auto& n = node->leaf[i];
		if (options & (1<<i)) {
			n = new node_t;
			n->spin_idx = node->spin_idx+1;
			n->state_thus_far = node->state_thus_far;
			if (i == 1){
				or_bit(n->state_thus_far, node->spin_idx);
			}
/*
			// for aesthetics only
			++tree_size[n->spin_idx];
			counter = (counter+1)%print_frequency;
			if (counter == 0){ print_tree_state(tree_size);	}
*/			
			// recurse
			_build_state_tree(n);
		}
	}
}


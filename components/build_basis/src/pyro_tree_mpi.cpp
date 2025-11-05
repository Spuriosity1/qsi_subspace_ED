
#include "pyro_tree_mpi.hpp"
#include "bittools.hpp"

MPI_Datatype create_vtree_node_type(){
    static MPI_Datatype vtree_node_type = MPI_DATATYPE_NULL;

    if (vtree_node_type == MPI_DATATYPE_NULL){

        vtree_node_t dummy;

        int block_lengths[3] = {16, 1, 1};
        MPI_Aint displacements[3];
        MPI_Datatype types[3] = {MPI_BYTE, MPI_UNSIGNED, MPI_UNSIGNED};

        MPI_Aint base;
        MPI_Get_address(&dummy, &base);
        MPI_Get_address(&dummy.state_thus_far, &displacements[0]);
        MPI_Get_address(&dummy.curr_spin, &displacements[1]);
        MPI_Get_address(&dummy.num_spinon_pairs, &displacements[2]);

        displacements[0] -= base;
        displacements[1] -= base;
        displacements[2] -= base;

        // Fix the size of the first block to cover the entire Uint128 object
        MPI_Datatype tmp;
        MPI_Type_create_struct(3, block_lengths, displacements, types, &tmp);

        // Resize to the real sizeof(vtree_node_t) so MPI doesn’t mis-align consecutive elements
        MPI_Type_create_resized(tmp, 0, sizeof(vtree_node_t), &vtree_node_type);
        MPI_Type_commit(&vtree_node_type);
        MPI_Type_free(&tmp);
    }

    return vtree_node_type;
}

template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::distribute_initial_work(std::queue<vtree_node_t>& starting_nodes){

    std::vector<std::vector<vtree_node_t>> others_job_stacks(world_size);

    // distribute the love in a round robin
    int r=0;
    while(!starting_nodes.empty()){
        others_job_stacks[r].push_back(starting_nodes.front());
        starting_nodes.pop();
        print_node(std::cout<<"(for rank "<<r<<") ",others_job_stacks[r].back());
        r = (r+1) % world_size;
    }

    // farm out to different MPI processes
    MPI_Datatype vtree_node_type = create_vtree_node_type();

    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size); 
    std::vector<vtree_node_t> sendbuf;

    int offset=0;
    for (int i=0; i<world_size; i++){
        sendcounts[i] = others_job_stacks[i].size();
        displs[i] = offset;
        offset += sendcounts[i];
        sendbuf.insert(sendbuf.end(), 
                others_job_stacks[i].begin(),
                others_job_stacks[i].end());
    }

    // scatter the work;
    int my_count;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, 
            &my_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<vtree_node_t> recvbuf(my_count);
    MPI_Scatterv(
            sendbuf.data(), sendcounts.data(), displs.data(), vtree_node_type,
            recvbuf.data(), my_count, vtree_node_type, 0, MPI_COMM_WORLD);


    std::cout << "[rank "<<my_rank<<"] got "<<my_count<<" nodes\n";
    // push received work to the local stack
    for (const auto& node : recvbuf) {
        print_node(std::cout<<"[init] ", node);
        my_job_stack.push(node);
    }
}

template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::receive_initial_work(){
    assert(my_rank != 0);

    MPI_Datatype vtree_node_type = create_vtree_node_type();
    // scatter the work;
    int my_count;
    MPI_Scatter(nullptr, 0, MPI_INT, 
            &my_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<vtree_node_t> recvbuf(my_count);
    MPI_Scatterv(nullptr, nullptr, nullptr, vtree_node_type,
            recvbuf.data(), my_count, vtree_node_type, 0, MPI_COMM_WORLD);
    
    std::cout << "[rank "<<my_rank<<"] got "<<my_count<<" nodes:";
    // push received work to the local stack
    for (const auto& node : recvbuf) {
        printHex(std::cout, node.state_thus_far) << node.curr_spin<<"\n";
        my_job_stack.push(node);
    }
}


template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::state_tree_init(){
    // Look for a checkpoint from an old run
    checkpoint.load_stack(my_job_stack);

    if (my_job_stack.empty()){
        // no checkpoint found: bootstrap needed
        std::cout<<"Distributing initial work\n";
        // part one: node 0 builds some initial states
        if (my_rank == 0){
            std::queue<vtree_node_t> starting_nodes;
            starting_nodes.push(vtree_node_t({0,0,0}));
            _build_state_bfs(starting_nodes, world_size*INITIAL_DEPTH_FACTOR);

            if (static_cast<int>(starting_nodes.size()) < world_size){
                std::cerr << "Too few starting nodes ("<<starting_nodes.size()<<
                    "). Try running with a msaller world size\n";
                MPI_Abort(MPI_COMM_WORLD, 5);
            }
            distribute_initial_work(starting_nodes);
        } else {
            // Other ranks receive their initial work
            receive_initial_work();
        }
    }
    
    std::cout<<"initial stack:";
    for (const auto& x : this->my_job_stack){
        printHex(std::cout, x.state_thus_far)<<" ";
    }
    std::cout<<"\n";
}

template<typename T>
requires std::derived_from<T, lat_container>
vtree_node_t mpi_par_searcher<T>::pop_hardest_job(){
    // the job with the lowest spin ID is the most time consuming
    unsigned lowest_spin_id = std::numeric_limits<unsigned>::max();
    int i=0;
    int min_idx =0;
    for (i=0; i<static_cast<int>(my_job_stack.size()); i++){
        if (my_job_stack[i].curr_spin < lowest_spin_id){
            lowest_spin_id = my_job_stack[i].curr_spin;
            min_idx = i;
        }
    }
    vtree_node_t tmp = my_job_stack[min_idx];
    my_job_stack.erase(std::next(my_job_stack.begin(), min_idx));
    return tmp;
}

//non-blocking check whether there are any nodes requesting work
template<typename T>
requires std::derived_from<T, lat_container>
bool mpi_par_searcher<T>::check_work_requests(){
    MPI_Status status;
    int flag;
    auto vtree_mpi_type = create_vtree_node_type();

    MPI_Iprobe(MPI_ANY_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
        int requester_rank;
        MPI_Recv(&requester_rank, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK_REQUEST, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // send work if available
        if (!my_job_stack.empty()){
            int available = WORK_AVAILABLE;
            MPI_Send(&available, 1, MPI_INT, requester_rank, TAG_WORK_RESPONSE, MPI_COMM_WORLD);
            vtree_node_t state_to_send = pop_hardest_job();
            MPI_Send(&state_to_send, 1, vtree_mpi_type, requester_rank, TAG_WORK_RESPONSE, MPI_COMM_WORLD);

        } else {
            // if unavailable, refuse
            int available = WORK_UNAVAILABLE;
            MPI_Send(&available, 1, MPI_INT, requester_rank, TAG_WORK_RESPONSE, MPI_COMM_WORLD);
        }
    }
    return flag;
}


template<typename T>
requires std::derived_from<T, lat_container>
bool mpi_par_searcher<T>::request_work_from(int target_rank){
    // send work request
    MPI_Send(&my_rank, 1, MPI_INT, target_rank, TAG_WORK_REQUEST, MPI_COMM_WORLD);
    // Wait for response
    int available;
    MPI_Recv(&available, 1, MPI_INT, target_rank, TAG_WORK_RESPONSE, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    auto vtree_mpi_type = create_vtree_node_type();

    if (available == WORK_AVAILABLE){
        // receive work item
        vtree_node_t node_obtained;
        MPI_Recv(&node_obtained, 1, vtree_mpi_type, target_rank, TAG_WORK_RESPONSE,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        my_job_stack.emplace_back(node_obtained);
        return true;
    }
    return false;
}

template<typename T>
requires std::derived_from<T, lat_container>
bool mpi_par_searcher<T>::request_work_from_shuffled(){
    if (world_size == 1) return false;

    // Build list of all other ranks
    std::vector<int> targets;
    targets.reserve(world_size - 1);
    for (int i = 0; i < my_rank; i++) {
        if (i != world_size) targets.push_back(i);
    }
    
    // Shuffle them
    std::shuffle(targets.begin(), targets.end(), rng);
    
    // Try each in sequence until we get work
    for (int target : targets) {
        if (request_work_from(target)) {
            return true;
        }
    }
    
    // All ranks were empty
    return false;
}

template<typename T>
requires std::derived_from<T, lat_container>
bool mpi_par_searcher<T>::check_termination_nonblocking(
        MPI_Request& term_req, bool& checking){
    if (!checking) {
        // new termination request
        if (my_job_stack.empty()) {
            int local_empty = 1;
            int* global_empty_ptr = new int;  // Will be cleaned up after wait
            MPI_Iallreduce(&local_empty, global_empty_ptr, 1, MPI_INT, MPI_LAND, 
                          MPI_COMM_WORLD, &term_req);
            checking = true;
            return false;
        }
        return false;
    } else {
         // Check if termination check is complete
        int flag;
        MPI_Test(&term_req, &flag, MPI_STATUS_IGNORE);

        if (flag) {
            // Allreduce completed
            checking = false;
            // If we got work while checking, cancel termination
            if (!my_job_stack.empty()) {
                return false;
            }
            // All ranks were idle and we're still idle
            return true;
        }
        return false;

    } 
}


template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::build_state_tree(){
    // Part 1: initialise
    state_tree_init();

    // Part 2: All nodes do their part with load balancing
    // main work loop
    size_t iter_count = 0;
    size_t local_processed =0;
    MPI_Request term_req = MPI_REQUEST_NULL;
    bool checking_termination = false;

    while (true) {
        // Process local work
        while (!my_job_stack.empty() && iter_count < CHECK_INTERVAL) {
            if (my_job_stack.top().curr_spin == 
                    static_cast<T*>(this)->lat.spins.size()) {
                shard.push(
                        permute(my_job_stack.top().state_thus_far, perm));
                my_job_stack.pop();
            } else {
               static_cast<T*>(this)->fork_state(my_job_stack);
            }
            iter_count++;
            local_processed++;

            // If we got work while checking termination, we know we're not done
            if (checking_termination) {
                checking_termination = false;
                if (term_req != MPI_REQUEST_NULL) {
                    MPI_Cancel(&term_req);
                    MPI_Request_free(&term_req);
                    term_req = MPI_REQUEST_NULL;
                }
            }
        }

        // check for requesters
        check_work_requests();
        iter_count=0;


        if (check_termination_nonblocking(term_req, checking_termination)){
            break;
        }

        // request work
        if (my_job_stack.empty() && !checking_termination){
            request_work_from_shuffled();
        }
        
        // checkpoint the stack
        checkpoint.save_stack(my_job_stack);
    }

    printf("[rank %d] completed processing %lu nodes\n", my_rank, local_processed);


    if (!my_job_stack.empty()){
        throw std::logic_error("[rank "+std::to_string(my_rank) +
                "] About to terminate without clearing the stack"+
                "-- something is terribly wrong!");
    }



}


template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::
_build_state_bfs(std::queue<vtree_node_t>& node_queue, 
		unsigned long max_stack_size){

	while (!node_queue.empty() && node_queue.size() < max_stack_size){

//        print_node(std::cout<<"[bfs] ", node_queue.front());
		if (node_queue.front().curr_spin == static_cast<T*>(this)->lat.spins.size()){
			shard.push(permute(node_queue.front().state_thus_far, perm));
			node_queue.pop();
		} else {
			static_cast<T*>(this)->fork_state(node_queue);
//            print_node(std::cout<<"[bfs post-fork] front ",node_queue.front());
//            print_node(std::cout<<"[bfs post-fork] back ",node_queue.back());
		}
	}
}

template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::
_build_state_dfs(lat_container::cust_stack& node_stack, 
		unsigned long max_queue_len){
	while (!node_stack.empty() && node_stack.size() < max_queue_len){
		if (node_stack.top().curr_spin == static_cast<T*>(this)->lat.spins.size()){
			shard.push(permute(node_stack.top().state_thus_far, perm));
			node_stack.pop();
		} else {
			static_cast<T*>(this)->fork_state(node_stack);
		}
	}
}

template class mpi_par_searcher<lat_container>;
template class mpi_par_searcher<lat_container_with_sector>;



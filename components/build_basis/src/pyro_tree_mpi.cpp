#include "pyro_tree_mpi.hpp"
#include "bittools.hpp"

MPI_Datatype create_vtree_node_type(){
    static MPI_Datatype vtree_node_type = MPI_DATATYPE_NULL;

    if (vtree_node_type == MPI_DATATYPE_NULL){

    int block_lengths[3] = {Uint128::i64_width, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_UINT64_T, MPI_UNSIGNED, MPI_UNSIGNED};
    
    vtree_node_t dummy;
    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.state_thus_far, &displacements[0]);
    MPI_Get_address(&dummy.curr_spin, &displacements[1]);
    MPI_Get_address(&dummy.num_spinon_pairs, &displacements[2]);
    
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);
    
    MPI_Type_create_struct(3, block_lengths, displacements, types, &vtree_node_type);
    MPI_Type_commit(&vtree_node_type);
    }

    return vtree_node_type;
}

template <typename StackT>
void rebalance_stacks(std::vector<StackT>& stacks) {

    std::vector<vtree_node_t> all_jobs;

    for (auto& stack : stacks) {
        printf("%4lu | ", stack.size());
        while (!stack.empty()) {
            all_jobs.push_back(stack.top());
            stack.pop();
        }
    }

    printf("\n");

    size_t i = 0;
    for (auto& job : all_jobs) {
        stacks[i++ % stacks.size()].push(job);
    }

}

void mpi_par_searcher::distribute_initial_work(std::queue<vtree_node_t>& starting_nodes){

    std::vector<std::vector<vtree_node_t>> others_job_stacks(world_size);

    // distribute the love in a round robin
    int r=0;
    while(!starting_nodes.empty()){
        others_job_stacks[r].push_back(starting_nodes.front());
        starting_nodes.pop();
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

    // push received work to the local stack
    for (const auto& node : recvbuf) {
        my_job_stack.push(node);
    }

}

bool mpi_par_searcher::check_for_work_requests() {
    int flag;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, WORK_REQUEST_TAG, MPI_COMM_WORLD, &flag, &status);
    
    if (flag) {
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, WORK_REQUEST_TAG, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        send_work_to_requester(status.MPI_SOURCE);
        return true;
    }
    return false;
}

void mpi_par_searcher::send_work_to_requester(int requester_rank) {
    MPI_Datatype vtree_node_type = create_vtree_node_type();
    
    // Split work: give away half of our stack
    int work_to_send = my_job_stack.size() / 2;
    std::vector<vtree_node_t> work_items;
    
    for (int i = 0; i < work_to_send && !my_job_stack.empty(); ++i) {
        work_items.push_back(my_job_stack.top());
        my_job_stack.pop();
    }
    
    // Send count first, then data
    int count = work_items.size();
    MPI_Send(&count, 1, MPI_INT, requester_rank, WORK_RESPONSE_TAG, MPI_COMM_WORLD);
    
    if (count > 0) {
        MPI_Send(work_items.data(), count, vtree_node_type, 
                 requester_rank, WORK_RESPONSE_TAG, MPI_COMM_WORLD);
    }
    
    MPI_Type_free(&vtree_node_type);
}

bool mpi_par_searcher::request_work_from_others() {
    MPI_Datatype vtree_node_type = create_vtree_node_type();
    
    // Try requesting work from each process in sequence
    for (int target = 0; target < world_size; ++target) {
        if (target == my_rank) continue;
        
        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
        
        int count;
        MPI_Recv(&count, 1, MPI_INT, target, WORK_RESPONSE_TAG, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (count > 0) {
            std::vector<vtree_node_t> received_work(count);
            MPI_Recv(received_work.data(), count, vtree_node_type, target, 
                     WORK_RESPONSE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (const auto& node : received_work) {
                my_job_stack.push(node);
            }
            
            MPI_Type_free(&vtree_node_type);
            return true;
        }
    }
    
    MPI_Type_free(&vtree_node_type);
    return false;
}

bool mpi_par_searcher::global_termination_check() {
    int local_done = my_job_stack.empty() ? 1 : 0;
    int global_done;
    MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return global_done == 1;
}


void mpi_par_searcher::build_state_tree(){
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
        std::queue<vtree_node_t> dummy;
        distribute_initial_work(dummy);
    }

    // Part 2: All nodes do their part with load balancing
    size_t local_processed = 0;
    bool work_available = true;

    while (work_available) {

        size_t batch_size = 0;
        while (!my_job_stack.empty() && batch_size < CHECKIN_INTERVAL){
            if (my_job_stack.top().curr_spin == lat.spins.size()){
                shard.push(permute(my_job_stack.top().state_thus_far, perm));
                my_job_stack.pop();
            } else {
                fork_state(my_job_stack);
            }
            batch_size++;
        }

        // check if anyone's a little heavy
        check_for_work_requests();

        // if out of work, seek more
        if (my_job_stack.empty()) {
            bool got_work = request_work_from_others();
            if (!got_work) {
                // are we done?
                work_available = !global_termination_check();
            }
        }
        
        // status update
        if (local_processed % (CHECKIN_INTERVAL * 10) == 0) {
            printf("[rank %d] procesed %lu nodes \n", my_rank, local_processed);
        }
    }

    printf("[rank %d] completed processing %lu nodes\n", my_rank, local_processed);

    if (!my_job_stack.empty()){
        throw std::logic_error("[rank "+std::to_string(my_rank) +
                "] About to terminate without clearing the stack"+
                "-- something is terribly wrong!");
    }



}


void mpi_par_searcher::
_build_state_bfs(std::queue<vtree_node_t>& node_stack, 
		unsigned long max_queue_len){
	while (!node_stack.empty() && node_stack.size() < max_queue_len){
		if (node_stack.front().curr_spin == lat.spins.size()){
			shard.push(permute(node_stack.front().state_thus_far, perm));
			node_stack.pop();
		} else {
			fork_state(node_stack);
		}
	}
}

void mpi_par_searcher::
_build_state_dfs(cust_stack& node_stack, 
		unsigned long max_queue_len){
	while (!node_stack.empty() && node_stack.size() < max_queue_len){
		if (node_stack.top().curr_spin == lat.spins.size()){
			shard.push(permute(node_stack.top().state_thus_far, perm));
			node_stack.pop();
		} else {
			fork_state(node_stack);
		}
	}
}



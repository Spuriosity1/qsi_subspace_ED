
#include "pyro_tree_mpi.hpp"
#include "bittools.hpp"
#include "mpi.h"

volatile sig_atomic_t GLOBAL_SHUTDOWN_REQUEST=0;


template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::distribute_initial_work(std::queue<vtree_node_t>& starting_nodes){
    std::cout<<"Distributing initial work (clean run)";

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
        // part one: node 0 builds some initial states
        if (my_rank == 0){
            std::cout<<"Distributing initial work..."<<std::endl;
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
    assert(my_job_stack.size() > 0);
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
bool mpi_par_searcher<T>::check_work_requests(bool allow_steal){
    MPI_Status status;
    int flag;
//    auto packet_mpi_type = create_packet_type();

    MPI_Iprobe(MPI_ANY_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
        int requester_rank;
        MPI_Recv(&requester_rank, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK_REQUEST, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        packet p_send;
        // send work if available
        if (!allow_steal || my_job_stack.size() < 3){
            // if unavailable, refuse
            p_send.available = WORK_UNAVAILABLE;
            p_send.state = vtree_node_t{0,0,0};

        } else {
            p_send.available = WORK_AVAILABLE;
            p_send.state = pop_hardest_job();
        }

        assert(p_send.available == WORK_AVAILABLE || p_send.available == WORK_UNAVAILABLE);
        printHex(
        std::cout <<my_rank<<"] sending "<<p_send.available<<" (=" << (p_send.available ? "AVAIL) " : "UNAVAIL) ") ,
                p_send.state.state_thus_far)<<" spin "<<p_send.state.curr_spin<<" to " << requester_rank<<std::endl;
        MPI_Ssend(&p_send, 1, create_packet_type(), requester_rank,
                TAG_WORK_RESPONSE, MPI_COMM_WORLD);
//        MPI_Ssend(&p_send, sizeof(packet), MPI_BYTE, requester_rank, TAG_WORK_RESPONSE, MPI_COMM_WORLD);

    }
    return flag;
}

// non-blocking request 
template<typename T>
requires std::derived_from<T, lat_container>
bool mpi_par_searcher<T>::request_work_from(int target_rank)
{
    using namespace std::this_thread; // sleep_for, sleep_until
    using namespace std::chrono; // nanoseconds, system_clock, seconds

    MPI_Request req_send, req_recv;

    MPI_Isend(&my_rank, 1, MPI_INT, target_rank, TAG_WORK_REQUEST, MPI_COMM_WORLD, &req_send);
//    db_print("S -> ") << target_rank << " tag "<<TAG_WORK_REQUEST<<std::endl;


    double dt;
    double t;
    const static double MAX_DELAY=0.1; // seconds

    int flag = 0;

    // await send success
    for (dt=0.001; !flag && dt < MAX_DELAY; dt*=2) {
        MPI_Test(&req_send, &flag, MPI_STATUS_IGNORE);
        t=MPI_Wtime();
        while(MPI_Wtime() < t+dt); // check_work_requests(false);
    }

    if(!flag){
        MPI_Cancel(&req_send);
        std::cout <<my_rank<<"] Cannot send to rank " <<target_rank<<std::endl;
        return false;
    }


    auto packet_mpi_type = create_packet_type();
    packet received;
    MPI_Irecv(&received, 1, packet_mpi_type, target_rank, TAG_WORK_RESPONSE, MPI_COMM_WORLD, &req_recv);

    // await recv success
    for (dt=0.001; !flag && dt < MAX_DELAY; dt*=2) {
        MPI_Test(&req_recv, &flag, MPI_STATUS_IGNORE);
        t=MPI_Wtime();
        while(MPI_Wtime() < t+dt);// check_work_requests(false);
    }

    if(!flag){
        MPI_Cancel(&req_recv);
        std::cout <<my_rank<<"] Cannot recv from rank " <<target_rank<<std::endl;
        return false;
    }



    printHex(
            std::cout <<my_rank<<"] got " <<received.available << " (=" << (received.available ? "AVAIL) " : "UNAVAIL) ") ,
            received.state.state_thus_far)<<" spin "<<received.state.curr_spin<<" from " << target_rank<<std::endl;

    assert(received.available == WORK_AVAILABLE || received.available == WORK_UNAVAILABLE);
    if (received.available == WORK_AVAILABLE) {
        my_job_stack.push_back(received.state);
        return true;
    }

    return false;
}

template<typename T>
requires std::derived_from<T, lat_container>
bool mpi_par_searcher<T>::request_work_from_shuffled(){
    if (world_size == 1) return false;
    

    std::cout<<my_rank<<"] requesting work... "<<std::endl;

    // Build list of all other ranks
    std::vector<int> targets;
    targets.reserve(world_size - 1);
    for (int i = 0; i < world_size; i++) {
        if (i != my_rank) targets.push_back(i);
    }
    
    // Shuffle them
    std::shuffle(targets.begin(), targets.end(), rng);
    
    // Try each in sequence until we get work
    for(int target : targets) {
//        std::cout << my_rank<<"] trying "<<target<<"... "<<std::flush;
//        check_work_requests(false);
        if(request_work_from(target)){
            std::cout << my_rank << "] pulled ";
            printHex(std::cout, my_job_stack[0].state_thus_far)<<" from " << target << std::endl;
            return true;
        }
    }


    std::cout << my_rank << "] all ranks empty "<<std::endl;
    
    // All ranks were empty
    return false;
}




template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::initiate_termination_check(MPI_Request* send_req){
    int dest_rank = (1 % world_size); // send to self if world_size == 1
    if (my_rank==0) {
        if (my_job_stack.empty()){
            int continue_shutdown=1;
        // send the token around and add
        // counts number of empty stacks
            MPI_Isend(&continue_shutdown, 1, MPI_INT, dest_rank, TAG_SHUTDOWN_RING, MPI_COMM_WORLD, send_req);
//            db_print("S -> ") << dest_rank << " tag "<<TAG_WORK_REQUEST<<std::endl;
        }
    }
}


template<typename T>
requires std::derived_from<T, lat_container>
bool mpi_par_searcher<T>::check_termination_requests(MPI_Request* send_req){
    // returns true if we should exit
    const int src_rank = (my_rank + world_size - 1) % world_size;
    const int dest_rank = (my_rank + 1) % world_size;

    int flag;
    continue_exit = 0;


    MPI_Iprobe(src_rank, TAG_SHUTDOWN_RING, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    if (flag){
        std::cout <<  src_rank << " -> " << my_rank << " -> "
            << dest_rank << "\n";

        MPI_Recv(&continue_exit, 1, MPI_INT, src_rank, TAG_SHUTDOWN_RING,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (!my_job_stack.empty()) {
            // cancel shutdown and continue

            std::cout << "rank " << my_rank << " cancelling... ";
            continue_exit = 0;
        } else {
            continue_exit++;
        }


        // with world_size =4, NUM_TERMINATE_LOOPS = 3
        //  rank | acc
        //  0    |       4    8  12   16 x
        //  1    |  1    5    9  13 x
        //  2    |  2    6   10  14 x
        //  3    |  3    7   11  15 x
        
        // condition to continue: either rank != 0, or 0 < continue_exit < world_size*(NUM_TERMINATE_LOOPS+1)
        if (my_rank != 0 || 
            (continue_exit > 0 && continue_exit < world_size * (NUM_TERMINATE_LOOPS + 1))
            ){
            MPI_Isend(&continue_exit, 1, MPI_INT, dest_rank, TAG_SHUTDOWN_RING, MPI_COMM_WORLD, send_req);
//            db_print("S -> ") << dest_rank << " tag "<<TAG_SHUTDOWN_RING<<std::endl;
        }

        if (continue_exit > world_size * NUM_TERMINATE_LOOPS){
            std::cout << my_rank<<"] shutdown complete.\n";
            *send_req = MPI_REQUEST_NULL;
            return true;
        }

        if (my_rank == 0){
            // request denied
            *send_req = MPI_REQUEST_NULL;
        }

    }

    
    return false;
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
    size_t num_checks =0;
    size_t idle_iterations=0;

    MPI_Request fin_send_req = MPI_REQUEST_NULL;

//    bool force_reject = false;

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
        }


        if (GLOBAL_SHUTDOWN_REQUEST) {
          break;
        }       

        // give an update
        if ( num_checks++ > PRINT_INTERVAL && !my_job_stack.empty()){
            num_checks=0;
            std::cout<<my_rank<<"] bottom job @ spin "<<my_job_stack[0].curr_spin<<std::endl;
        }

        // service ring-token requests
        if (check_termination_requests(&fin_send_req))
            break;


        // check for requesters
        iter_count=0;

        // if idle: ask for work
        if (my_job_stack.empty()) {
            bool got_work = request_work_from_shuffled();
            
            if (!got_work) {
                idle_iterations++;

                // Rank 0: After being idle for a bit, enter termination phase
                if (idle_iterations > 3) {
                    break;
                }

                std::this_thread::sleep_for(std::chrono::microseconds(1000));
            } else {
                idle_iterations = 0;
            }
        } else {
            idle_iterations = 0;
        }
        check_work_requests();
    }

    std::cout<<my_rank<<"] entering termination "<<std::endl;

    // termination phase
    while (!GLOBAL_SHUTDOWN_REQUEST){
        if (my_rank==0 && fin_send_req == MPI_REQUEST_NULL){
            std::cout << "MPI termination begin\n";
            initiate_termination_check(&fin_send_req);
        }
        // service ring-token requests
        if (check_termination_requests(&fin_send_req))
            break;
        // check for requesters
        check_work_requests();
    }

    if (GLOBAL_SHUTDOWN_REQUEST) {
        // graceful exit mid-processing

        printf("[rank %d] interrupted: processed %lu nodes\n", my_rank, local_processed);
        shard.flush(true);
        checkpoint.save_stack(my_job_stack);

        if (my_rank == 0) {
            int shutdown = 1;
            MPI_Bcast(&shutdown, 1, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            int dummy;
            MPI_Bcast(&dummy, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    } else if (!my_job_stack.empty()){
        throw std::logic_error("[rank "+std::to_string(my_rank) +
                "] About to terminate without clearing the stack"+
                "-- something is terribly wrong!");
    } else {
        printf("[rank %d] completed processing %lu nodes\n", my_rank, local_processed);
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


MPI_Datatype create_vtree_node_type() {
  static MPI_Datatype type = MPI_DATATYPE_NULL;
  static MPI_Datatype tmp;
  if (type != MPI_DATATYPE_NULL)
      return type;

  vtree_node_t dummy;

  int block_lengths[3] = {sizeof(Uint128), 1, 1};
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
  MPI_Type_create_struct(3, block_lengths, displacements, types, &tmp);

  // Resize to the real sizeof(vtree_node_t) so MPI doesn’t mis-align consecutive elements
  MPI_Type_create_resized(tmp, 0, sizeof(vtree_node_t), &type);
  MPI_Type_commit(&type);

  return type;
}

MPI_Datatype create_packet_type() {
  static MPI_Datatype type = MPI_DATATYPE_NULL;
  static MPI_Datatype tmp;

  if (type != MPI_DATATYPE_NULL)
    return type;

  packet dummy;

  int block_lengths[2] = {1, 1};
  MPI_Aint displacements[2];
  MPI_Datatype types[2] = { create_vtree_node_type(), MPI_INT32_T};

  MPI_Aint base;
  MPI_Get_address(&dummy, &base);
  MPI_Get_address(&dummy.state, &displacements[0]);
  MPI_Get_address(&dummy.available, &displacements[1]);

  displacements[0] -= base;
  displacements[1] -= base;

  // Fix the size of the first block to cover the entire Uint128 object
  MPI_Type_create_struct(2, block_lengths, displacements, types, &tmp);

  // Resize to the real sizeof(vtree_node_t) so MPI doesn’t mis-align consecutive elements
  MPI_Type_create_resized(tmp, 0, sizeof(packet), &type);
  MPI_Type_commit(&type);

  return type;
}


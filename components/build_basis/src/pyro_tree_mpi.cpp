
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
//    assert(min_idx == 0);
    vtree_node_t tmp = my_job_stack[min_idx];
    my_job_stack.erase(std::next(my_job_stack.begin(), min_idx));
    return tmp;
}


template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::request_work_from(int target_rank){
    db_print("Requesting work from rank ")<<target_rank<<std::endl;
    char dummy=0;
    MPI_Send(&dummy, 1, MPI_BYTE, target_rank, TAG_WORK_REQUEST, MPI_COMM_WORLD);
}


template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

static_assert(std::is_trivially_copyable_v<vtree_node_t>, 
              "Cannot safely MPI_Send this type");


template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::handle_send_request(int dest_rank){
    char dummy;
    MPI_Recv(&dummy, 1, MPI_BYTE, dest_rank, TAG_WORK_REQUEST,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vtree_node_t tmp;
    if (my_job_stack.size() > MIN_CHUNK_SIZE){
        tmp = pop_hardest_job();
        db_print("Sending valid work to rank ")<<dest_rank<<std::endl;
    } else {
        tmp.curr_spin = SPINID_RANK_EMPTY;
        // not actually needed, but definitely noticeable (this spin state should never occur)
        tmp.state_thus_far = {static_cast<uint64_t>(-1), static_cast<uint64_t>(-1)};
        db_print("my stack is empty -> ")<<dest_rank<<std::endl;
    }
    MPI_Send(&tmp, 1, create_vtree_node_type(), dest_rank, TAG_WORK_RESPONSE, MPI_COMM_WORLD);
}

// return value: successful receipt
template<typename T>
requires std::derived_from<T, lat_container>
bool mpi_par_searcher<T>::recv_stack_state(int src_rank){
    vtree_node_t tmp;
    MPI_Recv(&tmp, 1, create_vtree_node_type(), src_rank, TAG_WORK_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (tmp.curr_spin == SPINID_RANK_EMPTY){
        db_print(" request refused by rank ") << src_rank<<std::endl;
        return false;
    } else {
        db_print(" received state from rank ") << src_rank<<std::endl;
        my_job_stack.push(tmp);
        return true;
    }
}

// rreturns true if we should exit
template<typename T>
requires std::derived_from<T, lat_container>
bool mpi_par_searcher<T>::handle_shutdown_ring(bool& shutdown_continues){
    // early exit if world size is trivial
    if (world_size == 1) return true;

    // returns true if we should exit
    const int src_rank = (my_rank + world_size - 1) % world_size;
    const int dest_rank = (my_rank + 1) % world_size;

//    int flag;
    int continue_exit;
    MPI_Recv(&continue_exit, 1, MPI_INT, src_rank, TAG_SHUTDOWN_RING,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!my_job_stack.empty()) {
        // cancel shutdown and continue
        std::cout << "rank " << my_rank << " cancelling shutdown... "<<std::endl;
        continue_exit = 0;
        shutdown_continues = false;
    } else {
        continue_exit++;
    }

    // with world_size =4, NUM_TERMINATE_LOOPS = 3
    //  rank | continue_exit
    //  0    |       4    8  12   16 x
    //  1    |  1    5    9  13 x
    //  2    |  2    6   10  14 x
    //  3    |  3    7   11  15 x
    
    // condition to continue: either rank != 0, or 0 < continue_exit < world_size*(NUM_TERMINATE_LOOPS+1)
    bool terminate =false;
    if (continue_exit > world_size * NUM_TERMINATE_LOOPS){
        std::cout << my_rank<<"] shutdown complete.\n";
        std::cout <<  src_rank << " -> " << my_rank << " X " <<std::endl;
        terminate = true;
    }

    std::cout << continue_exit << " | " << src_rank << " -> " << my_rank << " -> " << dest_rank << std::endl;
    if (!((terminate || continue_exit==0 ) && my_rank == 0)){
        // stop forwarding here
        MPI_Send(&continue_exit, 1, MPI_INT, dest_rank, TAG_SHUTDOWN_RING, MPI_COMM_WORLD);
    }
    return terminate;
}




auto obtain_shuffled_targets(int world_size, int my_rank, std::mt19937& rng){
    std::vector<int> targets;
    targets.reserve(world_size - 1);
    for (int i = 0; i < world_size; i++) {
        if (i != my_rank) targets.push_back(i);
    }
    std::shuffle(targets.begin(), targets.end(), rng);
    return targets;
}

auto obtain_ring_targets(int world_size, int my_rank){
    std::vector<int> targets;
    targets.reserve(world_size - 1);
    for (int i = 1; i < world_size; i++) {
        targets.push_back( (my_rank + i) % world_size );
    }
    return targets;
}


template<typename T>
requires std::derived_from<T, lat_container>
void mpi_par_searcher<T>::build_state_tree(){
    // Part 1: initialise
    state_tree_init();

    // Part 2: All nodes do their part with load balancing
    // main work loop
    size_t local_processed =0;
    size_t num_checks =0;
    size_t iter_count;

    int flag;
    MPI_Status status;

    auto steal_targets = obtain_ring_targets(world_size, my_rank);
    int steal_idx = 0;

    int active_request = -1; // who we are asking for work from
    bool shutdown_continues = false;


    while (true) {
        // Process local work
        for (iter_count = 0;
                !my_job_stack.empty() && 
                iter_count < static_cast<size_t>(CHECK_INTERVAL);
             iter_count++)
        {
            if (my_job_stack.top().curr_spin ==
                    static_cast<T *>(this)->lat.spins.size()) {
                shard.push(permute(my_job_stack.top().state_thus_far, perm));
                my_job_stack.pop();
            } else {
                static_cast<T *>(this)->fork_state(my_job_stack);
            }
            iter_count++;
            local_processed++;
        }

        if ( GLOBAL_SHUTDOWN_REQUEST) {
            break;
        }


        // handle signals

        //    db_print("Probing for singals... ")<<std::endl;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        if (flag){
//            db_print("Handling signal: ")<<status.MPI_TAG<<std::endl;
            if (status.MPI_TAG == TAG_WORK_REQUEST) {
                handle_send_request(status.MPI_SOURCE);
                // sends a noticeably invalid payload if no work is available
            } else if (status.MPI_TAG == TAG_WORK_RESPONSE) {
                active_request = -1; // show that request complete
                if (recv_stack_state(status.MPI_SOURCE)){
                    steal_idx = 0; // reset stealing on success
                } 
            } else if (status.MPI_TAG == TAG_SHUTDOWN_RING) {
                if (handle_shutdown_ring(shutdown_continues)){
                    break;  // shutdown complete
                }  
            }
        }
        

        // steal work if idle 
        if (my_job_stack.empty()){
            if (world_size==1) break;
            if (!shutdown_continues && active_request == -1) {
                if (steal_idx < static_cast<int>(steal_targets.size()) ) {
//                    if ( active_request == -1){
                        request_work_from(steal_targets[steal_idx]);
                        active_request = steal_idx;
                        steal_idx++;
//                    }
                } else {
                    // have tried everyone -- probably time to exit
                    // initiate shutdown on rank 0
                    if (my_rank == 0 ){
                        const int dest_rank = 1;
                        const int continue_exit = 0;
                        MPI_Send(&continue_exit, 1, MPI_INT, dest_rank, TAG_SHUTDOWN_RING, MPI_COMM_WORLD);
                    }

                    shutdown_continues=true;
                }
            }
        }

        // give an update
        if ( num_checks++ > PRINT_INTERVAL && !my_job_stack.empty()){
            num_checks=0;
            std::cout<<my_rank<<"] bottom job @ spin "<<my_job_stack[0].curr_spin<<std::endl;
        }
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
void mpi_par_searcher<T>::build_state_tree_allgather(){
    // Part 1: initialise
    state_tree_init();

    // Part 2: All nodes do their part with load balancing
    // main work loop
    size_t local_processed =0;
    size_t num_checks =0;
    size_t iter_count;

    while (true) {
        // Process local work
        for (iter_count=0;
                !my_job_stack.empty() && iter_count < CHECK_INTERVAL;
                iter_count++) {
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

        int n_idle =0;
        // Rebalance job stacks across ranks

        // Gather stack sizes
        int my_size = my_job_stack.size();
        std::vector<int> all_sizes(world_size);
        MPI_Allgather(&my_size, 1, MPI_INT,
                all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Count idle ranks
        for (int size : all_sizes) {
            if (size == 0) n_idle++;
        }

        if (n_idle > 0) {
            if (n_idle == world_size) goto end_of_loop;

            // Find one donor and one receiver
            int donor = -1, receiver = -1;
            for (int i = 0; i < world_size; i++) {
                if (all_sizes[i] == 0 && receiver == -1) {
                    receiver = i;
                } else if (all_sizes[i] > 1 && donor == -1) {
                    donor = i;
                }
            }

            if (donor != -1 && receiver != -1) {

            // Transfer shallowest node from donor to receiver
            if (my_rank == donor) {
                MPI_Send(&my_job_stack.front(), 
                        sizeof(vtree_node_t), MPI_BYTE, receiver, 0, MPI_COMM_WORLD);
                my_job_stack.erase(my_job_stack.begin());

                std::cout<<my_rank<<"] donate -> "<<receiver<<std::endl;
            } else if (my_rank == receiver) {
                vtree_node_t received_node;
                MPI_Recv(&received_node, sizeof(vtree_node_t),
                        MPI_BYTE, donor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::cout<<my_rank<<"] recv <- "<<donor<<std::endl;
                my_job_stack.push_back(received_node);
            }
            }
            
        }

        if (GLOBAL_SHUTDOWN_REQUEST) {
          break;
        }       

        // give an update
        if ( num_checks++ > PRINT_INTERVAL && !my_job_stack.empty()){
            num_checks=0;
            std::cout<<my_rank<<"] bottom job @ spin "<<my_job_stack[0].curr_spin<<std::endl;
        }

    }

end_of_loop:

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
  if (type != MPI_DATATYPE_NULL)
    return type;

  vtree_node_t dummy;
  MPI_Aint base, disp[3];
  int blocklen[3] = {sizeof(Uint128), 1, 1};

  MPI_Get_address(&dummy, &base);
  MPI_Get_address(&dummy.state_thus_far, &disp[0]);
  MPI_Get_address(&dummy.curr_spin, &disp[1]);
  MPI_Get_address(&dummy.num_spinon_pairs, &disp[2]);

  for (int i = 0; i < 3; i++)
    disp[i] -= base;

  // For the 128-bit field, send 16 bytes as a contiguous block
  // but better to use MPI_Type_contiguous(16, MPI_BYTE)

  MPI_Datatype types[3] = {MPI_BYTE, MPI_UNSIGNED, MPI_UNSIGNED};

  MPI_Datatype tmp;
  MPI_Type_create_struct(3, blocklen, disp, types, &tmp);

  MPI_Type_create_resized(tmp, 0, sizeof(vtree_node_t), &type);
  MPI_Type_commit(&type);

//  MPI_Type_free(&tmp);

  return type;
}



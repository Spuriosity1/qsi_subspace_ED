#include "operator_mpi.hpp"
#include <mpi.h>
#include <cassert>
#include "timeit.hpp"

#ifndef NDEBUG
#define ASSERT_STATE_FOUND(error_msg, state, result) \
    do { \
        if (result == 0) { \
            std::cerr << "State not found on rank " << ctx.my_rank << ": "; \
            printHex(std::cerr, state) << "\n"; \
            throw std::logic_error("State not found in " error_msg); \
        } \
        if (local_idx >= ctx.local_block_size()) { \
            throw std::logic_error("Bad local_idx in " error_msg); \
        } \
    } while(0)

#else
#define ASSERT_STATE_FOUND(error_msg, state, result) result
#endif

#ifdef DEBUG
#define DEBUG_PRINT_VEC(msg, op_index, vector, ctx) \
            std::cout << msg<<" (op "<<op_index<<") [node "<<ctx.my_rank<< "]\n";\
            for (int r=0; r<ctx.world_size; r++){\
                if (r == ctx.my_rank) std::cout<<"*";\
                std::cout << "\tvector["<<r<<"] -> "<<curr_op_comm.send_states[r].size() <<"\n";\
            }
#else
#define DEBUG_PRINT_VEC(msg, op_index, vector, ctx)
#endif

// reads only the local basis into memory
inline std::vector<Uint128> read_basis_hdf5(
        MPIctx& ctx,
        const std::string& infile,
        const char* dset_name = "basis"){

	std::vector<Uint128> result;

	// HDF5 identifiers
	hid_t file_id = -1, dataset_id = -1, dataspace_id = -1;
	herr_t status;

    try {
        // open the file
        file_id = H5Fopen(infile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file_id < 0) throw HDF5Error(file_id, -1, -1, "read_basis: Failed to open file");
		
		// Open the dataset
		dataset_id = H5Dopen(file_id, dset_name, H5P_DEFAULT);
		if (dataset_id < 0) throw HDF5Error(file_id, -1, dataset_id, "read_basis: Failed to open dataset");
		
		// Get the dataspace to retrieve the dimensions
		dataspace_id = H5Dget_space(dataset_id);
		if (dataspace_id < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "read_basis: Failed to get dataspace");
		
		// Get the dimensions
		int ndims = H5Sget_simple_extent_ndims(dataspace_id);
		if (ndims != 2) throw HDF5Error(file_id, dataspace_id, dataset_id, "read_basis: Expected 2D data");
		
        static_assert(sizeof(hsize_t) == sizeof(int64_t), "hsize_t is too small to index the dataset correctly");
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);
		if (status < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "read_basis: Failed to get dimensions");

        hsize_t row_width= dims[1];
        hsize_t total_rows = dims[0];

        if (total_rows == 0){
            throw std::runtime_error("Basis is empty!");
        }

        // build the index partition
        ctx.build_idx_partition(dims[0]);

        // do some random access to figure out the terminal states
        {     
            // Each rank reads its boundary states directly from file
            auto read_state = [&](hsize_t idx) {
                Uint128 val{};
                hsize_t offset[2] = { idx, 0 };
                hsize_t count[2]  = { 1, row_width };
                herr_t status;
                status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, nullptr, count, nullptr);
                if (status < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "read_state: Failed to select hyperslab");

                hid_t memspace = H5Screate_simple(2, count, nullptr);
                status = H5Dread(dataset_id, H5T_NATIVE_UINT64, memspace, dataspace_id, H5P_DEFAULT, &val);

                if (status < 0) throw HDF5Error(file_id, dataspace_id, dataset_id, "read_state: Failed to read memory");

                H5Sclose(memspace);
                return val;
            };

            for (int r = 0; r < ctx.world_size; ++r) {              
                assert(ctx.idx_partition[r] < static_cast<int>(total_rows));
                ctx.state_partition[r] = read_state(ctx.idx_partition[r]);
            }

            // one past last: last_state + 1
            Uint128 last = read_state(total_rows - 1);
            ++last.uint128;
            ctx.state_partition[ctx.world_size] = last;
        }
    
        
        // Local chunk indices (by global index)
        const uint64_t local_start = static_cast<uint64_t>(ctx.idx_partition[ctx.my_rank]);
        const uint64_t local_end   = static_cast<uint64_t>(ctx.idx_partition[ctx.my_rank+1]);
        const uint64_t local_count = (local_end > local_start) ? (local_end - local_start) : 0;

        if (local_start + local_count > total_rows){
            std::cerr << "Error on Rank "<<ctx.my_rank<<":"
                <<"\ntotal_rows = "<<total_rows
                <<"\nlocal_start = "<<local_start
                <<"\nlocal_start + local_count = "<<local_start+local_count
                <<"\nidx_partition=[";
            for (const auto& I : ctx.idx_partition){
                std::cerr << I <<",";
            }
            std::cerr <<"]\n";
            throw std::logic_error("Bad slab spec: trying to read past the end!");
        }
		
        // read the slab in [local_start ... local_end)
        if (local_count > 0) {
		    // Allocate memory for the result
            result.resize(local_count);

            // Select hyperslab in file dataspace
            hsize_t file_offset[2] = { local_start, 0 };
            hsize_t file_count[2]  = { local_count, row_width };
            status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, file_offset, nullptr, file_count, nullptr);
            if (status < 0) throw std::runtime_error("read_basis_hdf5: Failed to select hyperslab");

            // Memory dataspace
            hid_t memspace = H5Screate_simple(2, file_count, nullptr);
            if (memspace < 0) throw std::runtime_error("read_basis_hdf5: Failed to create memspace");

            // Read as native uint64s into the memory of local_states
            status = H5Dread(dataset_id, H5T_NATIVE_UINT64, memspace, dataspace_id, H5P_DEFAULT, reinterpret_cast<void*>(result.data()));
            H5Sclose(memspace);
            if (status < 0) throw std::runtime_error("read_basis_hdf5: Failed to read local chunk");
        }

        // Print diagnostics
        assert(ctx.idx_partition.size() == ctx.state_partition.size());
        std::cout<<"Loaded basis chunk. Partition scheme:\n index\t state\n";
        for (size_t i=0; i<ctx.idx_partition.size(); i++){
            std::cout<<ctx.idx_partition[i]<<"\t";
            printHex(std::cout, ctx.state_partition[i])<<"\n";
        }
		
		// Clean up
		H5Sclose(dataspace_id);
		H5Dclose(dataset_id);
		H5Fclose(file_id);
	} catch (const HDF5Error& e){
		if (dataset_id >= 0) H5Dclose(dataset_id);
		if (dataspace_id >= 0) H5Sclose(dataspace_id);
		if (file_id >= 0) H5Fclose(file_id);
		throw;
	}

    return result;

}



MPIctx MPI_ZBasisBST::load_from_file(const fs::path& bfile, const std::string& dataset){
     // MPI setup
    MPIctx ctx;
//    int rank = ctx.world_rank;
//    int size = ctx.world_size;

    std::cerr << "Loading basis from file " << bfile <<"\n";
    if (bfile.stem().extension() == ".partitioned"){
        assert(bfile.extension() == ".h5");
        states = read_basis_hdf5(ctx, bfile, dataset.c_str());
    } else if (bfile.extension() == ".h5"){
        assert(dataset=="basis");
        states = read_basis_hdf5(ctx, bfile, "basis"); 
    } else {
        throw std::runtime_error(
                "Bad basis format: file must end with .csv or .h5");
    }

    std::cerr << "Done!" <<"\n";
    return ctx;
}

//void MPI_ZBasisBST::load_state(std::vector<double>& psi, const fs::path& eig_file){
//    
//}


template<RealOrCplx coeff_t, Basis B >
void MPILazyOpSum<coeff_t, B>::inplace_bucket_sort(std::vector<ZBasisBase::state_t>& states,
        std::vector<coeff_t>& c,
        std::vector<int>& bucket_sizes,
        std::vector<int>& bucket_starts
        ) const {
    size_t n = states.size();
    assert(c.size() == n);

    bucket_sizes.resize(ctx.world_size);
    bucket_starts.resize(ctx.world_size);
    std::fill(bucket_sizes.begin(), bucket_sizes.end(), 0);


     // Step 1: Count elements per bucket
//    std::vector<size_t> bucket_sizes(context.world_size, 0);
    for (const auto& s : states) {
        ++bucket_sizes[ctx.rank_of_state(s)];
    }

    // Step 2: Compute bucket start indices
//    std::vector<size_t> bucket_starts(context.world_size, 0);
    bucket_starts[0]=0;
    for (int i = 1; i < ctx.world_size; ++i) {
        bucket_starts[i] = bucket_starts[i - 1] + bucket_sizes[i - 1];
    }

     // Step 3: Rearrange
    std::vector<coeff_t> _scratch_coeff;
    std::vector<ZBasisBase::state_t> _scratch_states;
    _scratch_states.resize(n);
    _scratch_coeff.resize(n);
    std::vector<int> bucket_next = bucket_starts; // Next free slot in each bucket

     for (size_t i = 0; i < n; ++i) {
        int target_bucket = ctx.rank_of_state(states[i]);
        size_t target_index = bucket_next[target_bucket]++;
        _scratch_states[target_index] = states[i];
        _scratch_coeff[target_index] = c[i];
    }

     states = std::move(_scratch_states);
     c = std::move(_scratch_coeff);

}




template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const {
    for (const auto& term : ops.diagonal_terms) {
        const auto& c = term.first;   
        const auto& op = term.second;

        assert(op.is_diagonal());
       
        // there is no need to communicate, it's literally just this??
//        #pragma omp parallel for schedule(static)
        for (ZBasisBase::idx_t i = 0; i<ctx.local_block_size(); ++i){
            ZBasisBase::state_t psi = basis[i];
            coeff_t dy = c * x[i] * static_cast<double>(op.applyState(psi));
            // completely in place, no i collisions
            y[i] += dy;
        }       
    }
}


/* BROKEN on tcm-sc1, probably a problem with operator ID collisions
template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::evaluate_add_off_diag_sync(const coeff_t* x, coeff_t* y) const {
     // For each off-diagonal term do:
    // 1) produce per-destination vectors of (state,dy)
    // These are generally directed only to one or two nodes, so Alltoallv is inefficient.
    // 2) Exchange metadata. Synchronously communicate i) which nodes I will be receiving and ii) how much data to expect. Allocate send and receive buffers.
    // 3) Send and receive state-coeff pairs.
    // 4) on receive side, binary-search local basis block to find local index and add to y.
    //
    // Notes & assumptions:
    // - basis[i] for i in [0, block_size) returns the local state's sorted values.
    //
    //
    const int world_size = static_cast<int>(ctx.world_size);


    size_t debug_count =0;
    for (const auto& term : ops.off_diag_terms) {
#ifdef DEBUG
        std::cout << "[AOD] Operator "<<debug_count++<<"\n";
#endif

        const auto& c = term.first;
        const auto& op = term.second;

        const ZBasisBase::idx_t local_block = ctx.local_block_size();

        std::vector<std::vector<coeff_t>> send_dy(ctx.world_size); 
        std::vector<std::vector<ZBasisBase::state_t>> send_states(ctx.world_size);

        // receivers
        std::vector<coeff_t> recv_dy;
        std::vector<ZBasisBase::state_t> recv_states;


        BENCH_TIMEIT("[EAOD] local apply",
        // Apply to all local basis states and store non-vanishing entries
        for (ZBasisBase::idx_t il=0; il<local_block; ++il){
            ZBasisBase::state_t state = basis[il];
            auto sign = op.applyState(state); // "state" is new now
            if (sign == 0 || abs(x[il]) < APPLY_TOL) continue;

            auto target_rank = ctx.rank_of_state(state);
            send_dy[target_rank].push_back( c * x[il] * sign);
            send_states[target_rank].push_back(state);
        }
        )

        // collect non-self and non-empty ranks
        std::vector<int> send_targets;
        for (int i=0; i<ctx.world_size; i++){
            if ((i != ctx.my_rank) && (!send_states[i].empty())){
                send_targets.push_back(i);
            }
        }
        // Begin farming data out to others.
        BENCH_TIMEIT("[EAOD] Metadata exchange",
        // Flirtation phase: let other ranks know I will be bothering them
        int num_targets = send_targets.size();
        std::vector<int> all_num_targets(world_size);
        MPI_Allgather(&num_targets, 1, MPI_INT,
                      all_num_targets.data(), 1, MPI_INT,
                      MPI_COMM_WORLD);
        // all_num_targets is the globally shared number of targets on each node

        // displacements for the received target data
        std::vector<int> recv_displs(world_size + 1, 0);
        for (int r = 0; r < world_size; ++r) {
            recv_displs[r + 1] = recv_displs[r] + all_num_targets[r];
        }

        std::vector<int> all_targets(recv_displs[world_size]);
        MPI_Allgatherv(send_targets.data(), num_targets, MPI_INT,
                       all_targets.data(), all_num_targets.data(),
                       recv_displs.data(), MPI_INT,
                       MPI_COMM_WORLD);
        
        // Find who will send to me
        std::vector<int> recv_sources;
        for (int r = 0; r < world_size; ++r) {
            if (r == ctx.my_rank) continue;
            for (int i = recv_displs[r]; i < recv_displs[r + 1]; ++i) {
                if (all_targets[i] == ctx.my_rank) {
                    recv_sources.push_back(r);
                    break;
                }
            }
        }
        )

        BENCH_TIMEIT("[EAOD] p2p data sharing",
        // Post receives from all sources
        std::vector<MPI_Request> requests;
        std::vector<std::vector<ZBasisBase::state_t>> recv_states_bufs(recv_sources.size());
        std::vector<std::vector<coeff_t>> recv_dy_bufs(recv_sources.size());
        
        for (size_t i = 0; i < recv_sources.size(); ++i) {
            int source = recv_sources[i];
            int recv_count;
            
            // Receive size (blocking for simplicity)
            MPI_Recv(&recv_count, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Allocate buffers
            recv_states_bufs[i].resize(recv_count);
            recv_dy_bufs[i].resize(recv_count);
            
            // Post non-blocking receives
            requests.push_back(MPI_Request{});
            MPI_Irecv(recv_states_bufs[i].data(), recv_count, get_mpi_type<ZBasisBase::state_t>(),
                     source, 1, MPI_COMM_WORLD, &requests.back());
            
            requests.push_back(MPI_Request{});
            MPI_Irecv(recv_dy_bufs[i].data(), recv_count, get_mpi_type<coeff_t>(),
                     source, 2, MPI_COMM_WORLD, &requests.back());
        }
        
        // Send to all targets
        for (int target : send_targets) {
            int send_count = send_states[target].size();
            
            // Send size (blocking is fine)
            MPI_Send(&send_count, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
            
            // Non-blocking sends for data
            requests.push_back(MPI_Request{});
            MPI_Isend(send_states[target].data(), send_count, get_mpi_type<ZBasisBase::state_t>(),
                     target, 1, MPI_COMM_WORLD, &requests.back());
            
            requests.push_back(MPI_Request{});
            MPI_Isend(send_dy[target].data(), send_count, get_mpi_type<coeff_t>(),
                     target, 2, MPI_COMM_WORLD, &requests.back());
        }
        )

        // update the local stuff first
        {
            BENCH_TIMEIT("[EAOD] local updates",
            // process my own data while MPI is sending the big buffers
            const auto& local_states = send_states[ctx.my_rank];
            const auto& local_dy = send_dy[ctx.my_rank];

            for (int i = 0; i < local_states.size(); ++i) {
                ZBasisBase::idx_t local_idx;
                ASSERT_STATE_FOUND("self", local_states[i],
                basis.search(local_states[i], local_idx)
                );
                y[local_idx] += local_dy[i];
            }
            )
        }

        BENCH_TIMEIT("[EAOD] idling for p2p sharing",
        // Wait for all communication to complete
        if (!requests.empty()) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
        }
        )

         // Process received updates
        BENCH_TIMEIT("[EAOD] received updates",
        for (size_t i = 0; i < recv_sources.size(); ++i) {
            for (size_t j = 0; j < recv_states_bufs[i].size(); ++j) {
                ZBasisBase::idx_t local_idx;
                ASSERT_STATE_FOUND("remote", recv_states_bufs[i][j], 
                basis.search( recv_states_bufs[i][j], local_idx)
                );
                y[local_idx] += recv_dy_bufs[i][j];
            }
        }
        )
    
    } // end external for loop
}
*/

template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y) const {
    // State for pipelined communication
    struct OperatorCommState {
        std::vector<MPI_Request> requests;

        std::vector<std::vector<coeff_t>> send_dy;
        std::vector<std::vector<ZBasisBase::state_t>> send_states;

        std::vector<std::vector<ZBasisBase::state_t>> recv_states_bufs;
        std::vector<std::vector<coeff_t>> recv_dy_bufs;
        std::vector<int> recv_sources;
        MPI_Request count_exchange_req;
        std::vector<int> recvcounts;
        bool count_exchange_done = false;
    };

    Timer loc_apply_timer("[local apply]", ctx.my_rank);
    Timer loc_up_timer("[local update]", ctx.my_rank);
    Timer rem_up_timer("[remote update]", ctx.my_rank);
    Timer countx_wait_timer("[count exchange wait]", ctx.my_rank);
    Timer countx_wait_timer_2("[count exchange wait 2]", ctx.my_rank);
    Timer remx_wait_timer("[remote exchange wait]", ctx.my_rank);

    std::vector<const Timer*> timers{&loc_apply_timer, &loc_up_timer, &rem_up_timer, &countx_wait_timer, &remx_wait_timer};

    std::vector<std::chrono::duration<double, std::milli>> loc_apply_times;

    OperatorCommState prev_op_comm;
    bool has_prev_op = false;

    int op_index = 0;
    for ( const auto& [c, op] : ops.off_diag_terms ){

        OperatorCommState curr_op_comm;
        curr_op_comm.send_states.resize(ctx.world_size);
        curr_op_comm.send_dy.resize(ctx.world_size);

         // Organize sends by destination rank
        BENCH_TIMER_TIMEIT(loc_apply_timer,

        for (ZBasisBase::idx_t il = 0; il < ctx.local_block_size(); ++il) {
            ZBasisBase::state_t state = basis[il];
            auto sign = op.applyState(state);
            if (sign == 0) continue;
            
            auto target_rank = ctx.rank_of_state(state);
            curr_op_comm.send_dy[target_rank].push_back(c * x[il] * sign);
            curr_op_comm.send_states[target_rank].push_back(state);
        }
        )


        // Tell all other nodes how many entries I will send
        // begin non-blocking metadata exchange for CURRENT operator
        std::vector<int> sendcounts(ctx.world_size, 0);
        {

            for (int r = 0; r < ctx.world_size; ++r) {
                sendcounts[r] = curr_op_comm.send_states[r].size();
            }

            DEBUG_PRINT_VEC("<< send ", op_index, sendcounts, ctx)

            curr_op_comm.recvcounts.resize(ctx.world_size);
            MPI_Ialltoall(sendcounts.data(), 1, MPI_INT,
                         curr_op_comm.recvcounts.data(), 1, MPI_INT,
                         MPI_COMM_WORLD, &curr_op_comm.count_exchange_req);
        }
        

        // Process local updates immediately
        BENCH_TIMER_TIMEIT(loc_up_timer,
            for (size_t i = 0; i < curr_op_comm.send_states[ctx.my_rank].size(); ++i) {
                ZBasisBase::idx_t local_idx;
                ASSERT_STATE_FOUND("self", curr_op_comm.send_states[ctx.my_rank][i],
                        basis.search(curr_op_comm.send_states[ctx.my_rank][i], local_idx)
                        );
                y[local_idx] += curr_op_comm.send_dy[ctx.my_rank][i];
            }
        )


        // === PROCESS PREVIOUS OPERATOR'S RECEIVES ===
        if (has_prev_op) {
            BENCH_TIMER_TIMEIT(countx_wait_timer,
            // Wait for previous operator's count exchange if not done
            // this if statement is probably always true, it's just for safety
            if (!prev_op_comm.count_exchange_done) { 
                MPI_Wait(&prev_op_comm.count_exchange_req, MPI_STATUS_IGNORE);
                prev_op_comm.count_exchange_done = true;
            }
            )

            DEBUG_PRINT_VEC(">> recv ", op_index-1, prev_op_comm.recvcounts, ctx)
            
            // Now we know who will send to us for previous operator
            // recvcounts is now populated and readable
            for (int source = 0; source < ctx.world_size; ++source) {
                if (source == ctx.my_rank || prev_op_comm.recvcounts[source] == 0) continue;
                
                // allocate space to store the received data
                prev_op_comm.recv_sources.push_back(source);
                prev_op_comm.recv_states_bufs.emplace_back(prev_op_comm.recvcounts[source]);
                prev_op_comm.recv_dy_bufs.emplace_back(prev_op_comm.recvcounts[source]);
                
                size_t idx = prev_op_comm.recv_states_bufs.size() - 1;
                // we just added a new row to the buffer, this is its index
                prev_op_comm.requests.push_back(MPI_Request{});
                MPI_Irecv(prev_op_comm.recv_states_bufs[idx].data(), 
                         prev_op_comm.recvcounts[source], get_mpi_type<ZBasisBase::state_t>(),
                         source, 10*(op_index-1) +1, MPI_COMM_WORLD, &prev_op_comm.requests.back());
                
                prev_op_comm.requests.push_back(MPI_Request{});
                MPI_Irecv(prev_op_comm.recv_dy_bufs[idx].data(),
                         prev_op_comm.recvcounts[source], get_mpi_type<coeff_t>(),
                         source, 10*(op_index-1) +2, MPI_COMM_WORLD, &prev_op_comm.requests.back());
            }
            
            
            BENCH_TIMER_TIMEIT(remx_wait_timer,
            // Wait for previous operator's communication to complete
            if (!prev_op_comm.requests.empty()) {
                MPI_Waitall(prev_op_comm.requests.size(), prev_op_comm.requests.data(), 
                           MPI_STATUSES_IGNORE);
            }
            )
            
            BENCH_TIMER_TIMEIT(rem_up_timer,
            // Process previous operator's received updates
            for (size_t i = 0; i < prev_op_comm.recv_sources.size(); ++i) {
                for (size_t j = 0; j < prev_op_comm.recv_states_bufs[i].size(); ++j) {
                    ZBasisBase::idx_t local_idx;
                    ASSERT_STATE_FOUND("remote", prev_op_comm.recv_states_bufs[i][j], 
                            basis.search( prev_op_comm.recv_states_bufs[i][j], local_idx)
                    );

                    y[local_idx] += prev_op_comm.recv_dy_bufs[i][j];
                }
            }
            )
        }

        // === DATA SENDS FOR CURRENT OPERATOR ===
        BENCH_TIMER_TIMEIT(countx_wait_timer_2,
        // Wait for current operator's count exchange to complete
        MPI_Wait(&curr_op_comm.count_exchange_req, MPI_STATUS_IGNORE);
        curr_op_comm.count_exchange_done = true;
        )

        // Begin sending to all non-empty, non-self targets
        for (int target_rank=0; target_rank<ctx.world_size; target_rank++){
            if (target_rank == ctx.my_rank || 
                    curr_op_comm.send_dy[target_rank].size() == 0) continue;

            curr_op_comm.requests.push_back(MPI_Request{});
            MPI_Isend(
                    curr_op_comm.send_states[target_rank].data(), curr_op_comm.send_states[target_rank].size(), get_mpi_type<ZBasisBase::state_t>(),
                    target_rank, 10*op_index + 1, MPI_COMM_WORLD,
                    &curr_op_comm.requests.back());

            curr_op_comm.requests.push_back(MPI_Request{});
            MPI_Isend(
                    curr_op_comm.send_dy[target_rank].data(), curr_op_comm.send_dy[target_rank].size(), get_mpi_type<coeff_t>(),
                    target_rank, 10*op_index + 2, MPI_COMM_WORLD,
                    &curr_op_comm.requests.back());

        }
        

        // get ready for next iteration
        prev_op_comm = std::move(curr_op_comm);
        has_prev_op = true;
        op_index++;

    } // end operator loop


    // === PROCESS FINAL OPERATOR'S RECEIVES ===
    if (has_prev_op) {
        BENCH_TIMER_TIMEIT(countx_wait_timer,
        // Wait for previous operator's count exchange if not done
        // this if statement is probably always true, it's just for safety
        if (!prev_op_comm.count_exchange_done) { 
            MPI_Wait(&prev_op_comm.count_exchange_req, MPI_STATUS_IGNORE);
            prev_op_comm.count_exchange_done = true;
        }
        )

        DEBUG_PRINT_VEC(">> recv final ", op_index-1, prev_op_comm.recvcounts, ctx)
        
        // Now we know who will send to us for previous operator
        // recvcounts is now populated and readable
        for (int source = 0; source < ctx.world_size; ++source) {
            if (source == ctx.my_rank || prev_op_comm.recvcounts[source] == 0) continue;
            
            // allocate space to store the received data
            prev_op_comm.recv_sources.push_back(source);
            prev_op_comm.recv_states_bufs.emplace_back(prev_op_comm.recvcounts[source]);
            prev_op_comm.recv_dy_bufs.emplace_back(prev_op_comm.recvcounts[source]);
            
            size_t idx = prev_op_comm.recv_states_bufs.size() - 1;
            // we just added a new row to the buffer, this is its index
            prev_op_comm.requests.push_back(MPI_Request{});
            MPI_Irecv(prev_op_comm.recv_states_bufs[idx].data(), 
                     prev_op_comm.recvcounts[source], get_mpi_type<ZBasisBase::state_t>(),
                     source, 10*(op_index-1) + 1, MPI_COMM_WORLD, &prev_op_comm.requests.back());
            
            prev_op_comm.requests.push_back(MPI_Request{});
            MPI_Irecv(prev_op_comm.recv_dy_bufs[idx].data(),
                     prev_op_comm.recvcounts[source], get_mpi_type<coeff_t>(),
                     source, 10*(op_index-1) + 2, MPI_COMM_WORLD, &prev_op_comm.requests.back());
        }
        
        
        BENCH_TIMER_TIMEIT(remx_wait_timer,
        // Wait for previous operator's communication to complete
        if (!prev_op_comm.requests.empty()) {
            MPI_Waitall(prev_op_comm.requests.size(), prev_op_comm.requests.data(), 
                       MPI_STATUSES_IGNORE);
        }
        )
        
        BENCH_TIMER_TIMEIT(rem_up_timer,
        // Process previous operator's received updates
        for (size_t i = 0; i < prev_op_comm.recv_sources.size(); ++i) {
            for (size_t j = 0; j < prev_op_comm.recv_states_bufs[i].size(); ++j) {
                ZBasisBase::idx_t local_idx;
                ASSERT_STATE_FOUND("remote", prev_op_comm.recv_states_bufs[i][j], 
                        basis.search( prev_op_comm.recv_states_bufs[i][j], local_idx)
                );

                y[local_idx] += prev_op_comm.recv_dy_bufs[i][j];
            }
        }
        )
    }


// print diagnostics
#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
        for (auto t : timers){
            t->print_summary();
        }
#endif


}



// explicit template instantiations: generate symbols to link with
template struct MPILazyOpSum<double, MPI_ZBasisBST>;
//template struct MPILazyOpSum<double, ZBasisInterp>;

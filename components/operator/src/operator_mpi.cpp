#include "operator_mpi.hpp"
#include <mpi.h>
#include <cassert>
#include "timeit.hpp"
#include <numeric>

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


        // do some random access to figure out the terminal states
        {     
            // Each rank reads its boundary states directly from file
            auto read_state = [&](uint64_t idx) {
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

            // build the index partition
            ctx.partition_indices_equal(total_rows);
            ctx.partition_basis(total_rows, read_state);
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

        {
            // populate mpi_context's buffer 
            // TODO move this logic inside the operator class
            std::unordered_set<uint64_t> hashes_seen_here;
            for (hsize_t j=0; j<local_count; j++){
                hashes_seen_here.insert(ctx.hash_state(result[j]));
            }

            std::vector<uint64_t> hashes_seen_here_v(hashes_seen_here.begin(),
                    hashes_seen_here.end());

            // MPI exchante number of unique hashes
            std::vector<int> hash_counts(ctx.world_size);
            int my_hash_count = hashes_seen_here.size();

            MPI_Allgather(&my_hash_count, 1, get_mpi_type<int>(),
                    hash_counts.data(), 1, get_mpi_type<int>(), MPI_COMM_WORLD);

            auto g_total_unique_hashes = std::accumulate(
                    hash_counts.begin(), hash_counts.end(),0);

            std::vector<int> hash_displs(ctx.world_size);
            hash_displs[0] = 0;
            for (int i=1; i<ctx.world_size; i++){
                hash_displs[i] = hash_displs[i-1] + hash_counts[i-1];
            }

            std::vector<uint64_t> g_hashes(g_total_unique_hashes);
            
            MPI_Allgatherv(
                    hashes_seen_here_v.data(), hashes_seen_here.size(), get_mpi_type<uint64_t>(),
                    g_hashes.data(), hash_counts.data(), hash_displs.data(), get_mpi_type<uint64_t>(), MPI_COMM_WORLD);

            for (int r=0; r<ctx.world_size; r++){
                ctx.insert_hashes(g_hashes, hash_displs[r], hash_counts[r], r);
            }


        }

        // Print diagnostics
        if(ctx.my_rank == 0){
            std::cout<<"[Main] Loaded basis chunk.\n"<<ctx;
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



template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::evaluate_add_off_diag_batched(const coeff_t* x, coeff_t* y) {

    assert(send_dy.size() != 0);
    assert(send_state.size() == send_dy.size());

    Timer initial_apply_timer("[initial apply]", ctx.my_rank);
    Timer sort_vectors_timer("[sort]", ctx.my_rank);
    Timer loc_apply_timer("[local apply]", ctx.my_rank);
    Timer remx_wait_timer("[waiting for data]", ctx.my_rank);
    Timer rem_apply_timer("[remote apply]", ctx.my_rank);

    std::vector<const Timer*> timers{&initial_apply_timer, &sort_vectors_timer,
        &loc_apply_timer, &remx_wait_timer, &rem_apply_timer};

    
    std::cout <<"[rank "<<ctx.my_rank<<"] Displacements: ";
    for (auto d : send_displs) std::cout << d <<", ";
    std::cout<<std::endl;

    std::cout <<"[rank "<<ctx.my_rank<<"] Sizes: ";
    for (auto d : send_counts) std::cout << d <<", ";
    std::cout<<std::endl;

    // current positions in the send arrays
    std::vector<int> send_cursors = send_displs;                 
    std::vector<int> send_counts_no_self = send_counts;
    std::vector<int> recv_counts_no_self = recv_counts;
    send_counts_no_self[ctx.my_rank] = 0; // handle this separately
    recv_counts_no_self[ctx.my_rank] = 0; // handle this separately

    // apply to all local basis vectors, il = local state index
    BENCH_TIMER_TIMEIT(initial_apply_timer,
    for (ZBasisBase::idx_t il = 0; il < ctx.local_block_size(); ++il) {
        for ( const auto& [c, op] : ops.off_diag_terms ){
            ZBasisBase::state_t state = basis[il];
            auto sign = op.applyState(state);
            if (sign == 0) continue;
            
            auto target_rank = ctx.rank_of_state(state);
            int pos = send_cursors[target_rank]++;
            send_state[pos] = state;
            send_dy[pos] = c*x[il]*sign;
        }
    }
    );

    // Sort 
    BENCH_TIMER_TIMEIT(sort_vectors_timer,

        // Sort both arrays by state to improve cache locality during basis.search()
        size_t total_count = send_state.size(); // or send_cursors.back() if not using full buffer
        std::vector<size_t> perm(total_count);
        std::iota(perm.begin(), perm.end(), 0);

        std::sort(perm.begin(), perm.end(),
            [&](size_t i, size_t j) {
            return send_state[i] < send_state[j];
            });

        // Apply permutation to both arrays
        std::vector<bool> done(total_count, false);
        for (size_t i = 0; i < total_count; ++i) {
        if (done[i]) continue;

        done[i] = true;
        size_t prev_j = i;
        size_t j = perm[i];

        while (i != j) {
            std::swap(send_state[prev_j], send_state[j]);
            std::swap(send_dy[prev_j], send_dy[j]);
            done[j] = true;
            prev_j = j;
            j = perm[j];
        }
        }
        );

    for (int r=0; r<ctx.world_size; r++){
        assert(send_cursors[r] == send_displs[r] + send_counts[r]);
    }

    MPI_Request req_state, req_coeff;

    MPI_Ialltoallv(
        send_state.data(), send_counts_no_self.data(), send_displs.data(),
        get_mpi_type<ZBasisBase::state_t>(),
        recv_state.data(), recv_counts_no_self.data(), recv_displs.data(),
        get_mpi_type<ZBasisBase::state_t>(),
        MPI_COMM_WORLD, &req_state
    );

    MPI_Ialltoallv(
        send_dy.data(), send_counts_no_self.data(), send_displs.data(),
        get_mpi_type<coeff_t>(),
        recv_dy.data(), recv_counts_no_self.data(), recv_displs.data(),
        get_mpi_type<coeff_t>(),
        MPI_COMM_WORLD, &req_coeff
    );

    assert(send_counts[ctx.my_rank] == recv_counts[ctx.my_rank]);
    const int loc_send_offset = send_displs[ctx.my_rank];

    BENCH_TIMER_TIMEIT(loc_apply_timer,
    for (int i=loc_send_offset; 
            i<loc_send_offset+send_counts[ctx.my_rank]; i++){
        ZBasisBase::idx_t local_idx;
        ASSERT_STATE_FOUND("local",
            send_state[i],
            basis.search(send_state[i], local_idx)
            );
        y[local_idx] += send_dy[i];
    }
    );

    // synchronise
    BENCH_TIMER_TIMEIT(remx_wait_timer,
    MPI_Wait(&req_state, MPI_STATUS_IGNORE);
    MPI_Wait(&req_coeff, MPI_STATUS_IGNORE);
    );

    BENCH_TIMER_TIMEIT(rem_apply_timer,
    // Applying rank-local updates
    for (int r=0; r<ctx.world_size; r++){
    const int rem_displs = recv_displs[r];
        for (int i = rem_displs; 
                i < rem_displs + recv_counts_no_self[r]; ++i) {
            ZBasisBase::idx_t local_idx;
            ASSERT_STATE_FOUND("remote",
                recv_state[i],
                basis.search(recv_state[i], local_idx)
            );
            y[local_idx] += recv_dy[i];
        }
    }
    );
    

#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
    for (auto t : timers){
        t->print_summary();
    }
#endif

}


template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::allocate_temporaries() {
    // runs through the current local basis applying op to everything.
    // We cound how many we want to send to each rank, then exchange synchronously.
    // We can then resize recv_states_bufs appropriately.
    send_counts.resize(ctx.world_size);
    send_displs.resize(ctx.world_size);

    recv_counts.resize(ctx.world_size);
    recv_displs.resize(ctx.world_size);

    std::fill(send_counts.begin(), send_counts.end(), 0);
    std::fill(recv_counts.begin(), recv_counts.end(), 0);
    std::fill(send_displs.begin(), send_displs.end(), 0);
    std::fill(recv_displs.begin(), recv_displs.end(), 0);

    // TODO: consider a once-over compression step
    for (ZBasisBase::idx_t il = 0; il < ctx.local_block_size(); ++il) {
        for (auto& [c, op] : ops.off_diag_terms ) {
            ZBasisBase::state_t state = basis[il];
            auto sign = op.applyState(state);
            if (sign == 0) continue;
            
            auto target_rank = ctx.rank_of_state(state);
            send_counts[target_rank]++;
        }
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int r=1; r<ctx.world_size; r++){
        send_displs[r] = send_counts[r-1] + send_displs[r-1];
        recv_displs[r] = recv_counts[r-1] + recv_displs[r-1];
    }

    const int total_send =
        std::accumulate(send_counts.begin(), send_counts.end(), 0);

    const int total_recv =
        std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    send_state.resize(total_send);
    send_dy.resize(total_send);

    recv_state.resize(total_recv);
    recv_dy.resize(total_recv);


#ifdef DEBUG
    std::cout <<"[alloc "<<ctx.my_rank<<"] Send Sizes: ";
    for (auto d : send_counts) std::cout << d <<", ";
    std::cout<<"\n\ttotal:"<<total_send<<std::endl;

    std::cout <<"[alloc "<<ctx.my_rank<<"] Send Displacements: ";
    for (auto d : send_displs) std::cout << d <<", ";
    std::cout<<std::endl;
#endif


}



// explicit template instantiations: generate symbols to link with
template struct MPILazyOpSum<double, MPI_ZBasisBST>;
//template struct MPILazyOpSum<double, ZBasisInterp>;

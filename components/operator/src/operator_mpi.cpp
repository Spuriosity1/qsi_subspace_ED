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
            ctx.log << msg<<" (op "<<op_index<<") [node "<<ctx.my_rank<< "]\n";\
            for (int r=0; r<ctx.world_size; r++){\
                if (r == ctx.my_rank) ctx.log<<"*";\
                ctx.log << "\tvector["<<r<<"] -> "<<curr_op_comm.send_states[r].size() <<"\n";\
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


        // calculate the index partition in an efficient manner
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

//            ctx.estimate_optimal_mask(total_rows, read_state, 4);
            ctx.partition_indices_equal(total_rows);
            ctx.populate_state_terminals(total_rows, read_state);
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
        ctx.log<<"Loaded basis chunk.\n"<<ctx;
        if (ctx.my_rank == 0){
            std::cout<<"[Main] "<<ctx;
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



MPIctx ZBasisBST_MPI::load_from_file(const fs::path& bfile, const std::string& dataset){
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

//void ZBasisBST_MPI::load_state(std::vector<double>& psi, const fs::path& eig_file){
//    
//}


template<RealOrCplx coeff_t, Basis B >
void MPILazyOpSumBase<coeff_t, B>::inplace_bucket_sort(std::vector<ZBasisBase::state_t>& states,
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
void MPILazyOpSumBase<coeff_t, B>::evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const {
    for (const auto& term : ops.diagonal_terms) {
        const auto& c = term.first;   
        const auto& op = term.second;

        assert(op.is_diagonal());
       
        // there is no need to communicate, it's literally just this??
//        #pragma omp parallel for schedule(static)
        for (ZBasisBase::idx_t i = 0; i<ctx.local_block_size(); ++i){
            ZBasisBase::state_t psi = basis[i];
            coeff_t dy = c * x[i] * static_cast<double>(op.applyState(psi));
            assert(psi == basis[i]);
            // completely in place, no i collisions
            y[i] += dy;
        }       
    }
}


template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSumPipe<coeff_t, B>::evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y) const {
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

    auto& ctx = this->ctx;

    Timer loc_apply_timer("[local apply]", ctx.log);
    Timer loc_up_timer("[local update]", ctx.log);
    Timer rem_up_timer("[remote update]", ctx.log);
    Timer countx_wait_timer("[count exchange wait]", ctx.log);
    Timer countx_wait_timer_2("[count exchange wait 2]", ctx.log);
    Timer remx_wait_timer("[remote exchange wait]", ctx.log);

    std::vector<const Timer*> timers{&loc_apply_timer, &loc_up_timer, &rem_up_timer, &countx_wait_timer, &remx_wait_timer};

    std::vector<std::chrono::duration<double, std::milli>> loc_apply_times;

    OperatorCommState prev_op_comm;
    bool has_prev_op = false;

    int op_index = 0;
    for ( const auto& [c, op] : this->ops.off_diag_terms ){

        OperatorCommState curr_op_comm;
        curr_op_comm.send_states.resize(ctx.world_size);
        curr_op_comm.send_dy.resize(ctx.world_size);

         // Organize sends by destination rank
        BENCH_TIMER_TIMEIT(loc_apply_timer,

        for (ZBasisBase::idx_t il = 0; il < ctx.local_block_size(); ++il) {
            ZBasisBase::state_t state = this->basis[il];
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
                        this->basis.search(curr_op_comm.send_states[ctx.my_rank][i], local_idx)
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
                            this->basis.search( prev_op_comm.recv_states_bufs[i][j], local_idx)
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
                        this->basis.search( prev_op_comm.recv_states_bufs[i][j], local_idx)
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
void MPILazyOpSumPipePrealloc<coeff_t, B>::allocate_temporaries(){
    
    auto& ctx = this->ctx;
    ctx.log<<"Allocating temporaries..."<<std::endl;

    log_rss(ctx.log, "allocate_temporaries() entry");
    ctx.log << "[rank " << ctx.my_rank << "] local_block_size=" 
            << ctx.local_block_size() 
            << "  n_off_diag_ops=" << this->ops.off_diag_terms.size() << "\n";
    

     // Build communication pattern cache
    comm_cache.sendcounts_per_op.resize(this->ops.off_diag_terms.size());
    comm_cache.recvcounts_per_op.resize(this->ops.off_diag_terms.size());

    size_t max_sends=0;
    size_t max_recvs=0;

    for (size_t op_idx=0; op_idx<this->ops.off_diag_terms.size(); op_idx++ ){
        const auto& [c, op] = this->ops.off_diag_terms[op_idx];

        auto& sendcounts = comm_cache.sendcounts_per_op[op_idx];
        auto& recvcounts = comm_cache.recvcounts_per_op[op_idx];

        sendcounts.resize(ctx.world_size, 0);
        recvcounts.resize(ctx.world_size, 0);
        
        // Count how many states each rank will receive from us
        for (ZBasisBase::idx_t il = 0; il < ctx.local_block_size(); ++il) {
            ZBasisBase::state_t state = this->basis[il];
            auto sign = op.applyState(state);
            if (sign == 0) continue;
            
            auto target_rank = ctx.rank_of_state(state);
            sendcounts[target_rank]++;
        }
        
        printvec(ctx.log << "(op "<<op_idx<<")<< send pattern "<< op_idx, sendcounts)<<"\n";
        
        // Exchange counts to learn recvcounts
        MPI_Alltoall(sendcounts.data(), 1, get_mpi_type<size_t>(),
                    recvcounts.data(), 1, get_mpi_type<size_t>(),
                    MPI_COMM_WORLD);
        
        printvec(ctx.log << "(op "<<op_idx<<")>> recv pattern "<< op_idx, recvcounts)<<"\n";

        // short-circuit: we do not receive from ourselves
        recvcounts[ctx.my_rank] = 0;

        max_sends = std::max(max_sends, std::accumulate(sendcounts.begin(), sendcounts.end(),
                    static_cast<size_t>(0)));

        max_recvs = std::max(max_recvs, std::accumulate(recvcounts.begin(), recvcounts.end(),
                    static_cast<size_t>(0)));
        
    }

        
    // ---- Critical: compute projected allocation before touching memory ----
    // coeff_t + state_t, two buffers (send+recv), two OperatorCommState objects
    constexpr size_t bytes_per_comm = sizeof(coeff_t) + sizeof(ZBasisBase::state_t);
    size_t projected_bytes = 2 * (max_sends + max_recvs) * bytes_per_comm;
    
    ctx.log << "max_sends=" << max_sends 
            << " max_recvs=" << max_recvs << "\n"
            << "projected comm buffer alloc: " 
            << projected_bytes / (1024.0*1024.0) << " MB"
            << " (" << projected_bytes / (1024.0*1024.0*1024.0) << " GB)\n";
    ctx.log.flush();
    
    log_rss(ctx.log, "before reserve()");

    for (int i=0; i<2; i++){
        comm_buffers[i].reserve(max_sends, max_recvs);
    }


    comm_cache.is_initialized = true;

    log_rss(ctx.log, "allocate_temporaries exit");
    ctx.log.flush();

}


template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSumPipePrealloc<coeff_t, B>::evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y) {
    // guard against invalid calls
    // cheap check
    if (!comm_cache.is_initialized){
        throw std::runtime_error("Must call allocate_temporaries() first!");
    }

    auto& ctx = this->ctx;


    Timer loc_apply_timer("[local apply]", ctx.log);
    Timer loc_up_timer("[local update]", ctx.log);
    Timer rem_up_timer("[remote update]", ctx.log);
    Timer remx_wait_timer("[remote exchange wait]", ctx.log);

    std::vector<const Timer *> timers{&loc_apply_timer, &loc_up_timer,
                                      &rem_up_timer,
                                      &remx_wait_timer};

    int prev_opbuf_id=0;
    int curr_opbuf_id=1;

    bool has_prev_op = false;

    for ( size_t op_index=0; op_index<this->ops.off_diag_terms.size(); op_index++ ){
        const auto& [c, op] = this->ops.off_diag_terms[op_index];


        auto& prev_op_buf = comm_buffers[prev_opbuf_id];
        auto& curr_op_buf = comm_buffers[curr_opbuf_id];

        // Clear current buffer for reuse (but keep allocated capacity)
        curr_op_buf.clear_for_reuse();

        const auto& sendcounts = comm_cache.sendcounts_per_op[op_index];
        const auto& recvcounts = comm_cache.recvcounts_per_op[op_index];
        // allocate space for the sends and receives
        curr_op_buf.reserve_send_resize_recv(sendcounts, recvcounts);

        // Organize sends by destination rank
        BENCH_TIMER_TIMEIT(loc_apply_timer,
            for (ZBasisBase::idx_t il = 0; il < ctx.local_block_size(); ++il) {
                ZBasisBase::state_t state = this->basis[il];
                auto sign = op.applyState(state);
                if (sign == 0) continue;
                
                auto target_rank = ctx.rank_of_state(state);

                curr_op_buf.sendbuf_push_back(target_rank, c*x[il]*sign, state);
            }
        )



        // Immediately post receives for the current operator
        for (int source = 0; source < ctx.world_size; ++source) {
            if (source == ctx.my_rank || curr_op_buf.get_recv_count(source) == 0) continue;

            const auto& [recv_coeff_buf, recv_state_buf] = curr_op_buf.get_recv_buffers(source);
            
            curr_op_buf.requests.push_back(MPI_Request{});
            MPI_Irecv(recv_coeff_buf,
                     recvcounts[source], get_mpi_type<coeff_t>(),
                     source, 10*op_index + 2, MPI_COMM_WORLD, &curr_op_buf.requests.back());
            
            curr_op_buf.requests.push_back(MPI_Request{});
            MPI_Irecv(recv_state_buf, 
                     recvcounts[source], get_mpi_type<ZBasisBase::state_t>(),
                     source, 10*op_index + 1, MPI_COMM_WORLD, &curr_op_buf.requests.back());
            
        }

        // Process local updates while receives are cooking
        BENCH_TIMER_TIMEIT(loc_up_timer,

            const auto& [loc_send_coeff_buf, loc_send_state_buf] = curr_op_buf.get_send_buffers(ctx.my_rank);
            for (size_t r = 0; r < sendcounts[ctx.my_rank]; ++r) {
                ZBasisBase::idx_t local_idx;
                ASSERT_STATE_FOUND("self", loc_send_state_buf[r],
                        this->basis.search(loc_send_state_buf[r], local_idx)
                        );
                y[local_idx] += loc_send_coeff_buf[r];
            }
        )


        // === PROCESS PREVIOUS OPERATOR'S RECEIVES ===
        if (has_prev_op) {
            BENCH_TIMER_TIMEIT(remx_wait_timer,
            // Wait for prev communications to arrive
            if (!prev_op_buf.requests.empty()) {
                MPI_Waitall(prev_op_buf.requests.size(),
                        prev_op_buf.requests.data(),
                        MPI_STATUSES_IGNORE);
                )
            }

            // Apply the states received from remote to the PREV operator
            BENCH_TIMER_TIMEIT(rem_up_timer, 
            for(int r=0; r<ctx.world_size; r++){
                if (r == ctx.my_rank || prev_op_buf.get_recv_count(r) == 0) continue;

                const auto& [recv_dy_buf, recv_state_buf] = prev_op_buf.get_recv_buffers(r);
                for (size_t j=0; j<prev_op_buf.get_recv_count(r); ++j){
                    ZBasisBase::idx_t local_idx;
                    ASSERT_STATE_FOUND("remote", 
                            recv_state_buf[j], 
                            this->basis.search(recv_state_buf[j], local_idx)
                            );

                    y[local_idx] += recv_dy_buf[j];
                }
            })
        }
        // === DATA SENDS FOR CURRENT OPERATOR ===

        // Begin sending to all non-empty, non-self targets
        for (int target_rank=0; target_rank<ctx.world_size; target_rank++){
            auto n_send = curr_op_buf.get_send_count(target_rank);
            if (target_rank == ctx.my_rank || n_send == 0) continue;

            const auto& [send_dy_buf, send_state_buf] = curr_op_buf.get_send_buffers(target_rank);

            curr_op_buf.requests.push_back(MPI_Request{});
            MPI_Isend(
                    send_dy_buf, n_send, get_mpi_type<coeff_t>(),
                    target_rank, 10*op_index + 2, MPI_COMM_WORLD,
                    &curr_op_buf.requests.back());

            curr_op_buf.requests.push_back(MPI_Request{});
            MPI_Isend(
                    send_state_buf, n_send, get_mpi_type<ZBasisBase::state_t>(),
                    target_rank, 10*op_index + 1, MPI_COMM_WORLD,
                    &curr_op_buf.requests.back());

        }
        
        // get ready for next iteration
        std::swap(prev_opbuf_id, curr_opbuf_id);
        has_prev_op = true;
                    
    } // end operator loop

    auto& prev_op_buf = comm_buffers[prev_opbuf_id];
    // === PROCESS FINAL OPERATOR'S RECEIVES ===
    if (has_prev_op) {
        BENCH_TIMER_TIMEIT(remx_wait_timer,
        // Wait for prev communications to arrive
        if (!prev_op_buf.requests.empty()) {
            MPI_Waitall(prev_op_buf.requests.size(),
                    prev_op_buf.requests.data(),
                    MPI_STATUSES_IGNORE);
            )
        }

        // Apply the states received from remote processing of PREV operator
        BENCH_TIMER_TIMEIT(rem_up_timer, 
        for(int r=0; r<ctx.world_size; r++){
            if (r == ctx.my_rank || prev_op_buf.get_recv_count(r) == 0) continue;

            const auto& [recv_dy_buff, recv_state_buf] = prev_op_buf.get_recv_buffers(r);
            for (size_t j=0; j<prev_op_buf.get_recv_count(r); ++j){
                ZBasisBase::idx_t local_idx;
                ASSERT_STATE_FOUND("remote", 
                        recv_state_buf[j], 
                        this->basis.search(recv_state_buf[j], local_idx)
                        );

                y[local_idx] += recv_dy_buff[j];
            }
        })

    }

    // print diagnostics
#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
    for (auto t : timers) {
        t->print_summary();
    }
#endif
}



template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSumBatched<coeff_t, B>::evaluate_add_off_diag_batched(
        const coeff_t* x, coeff_t* y) {

    assert(send_dy.size() != 0);
    assert(send_state.size() == send_dy.size());

    auto& ctx = this->ctx;

    Timer initial_apply_timer("[initial apply]", ctx.log);
//    Timer sort_vectors_timer("[sort]", ctx.log);
    Timer loc_apply_timer("[local apply]", ctx.log);
    Timer remx_wait_timer("[waiting for data]", ctx.log);
    Timer rem_apply_timer("[remote apply]", ctx.log);

    std::vector<const Timer*> timers{&initial_apply_timer,
        &loc_apply_timer, &remx_wait_timer, &rem_apply_timer};


    // current positions in the send arrays
    std::vector<MPI_Count> send_cursors = send_displs;                 
    std::vector<MPI_Count> send_counts_no_self = send_counts;
    std::vector<MPI_Count> recv_counts_no_self = recv_counts;
    send_counts_no_self[ctx.my_rank] = 0; // handle this separately
    recv_counts_no_self[ctx.my_rank] = 0; // handle this separately
 

//    // thread local storage
//    const int nthreads = omp_get_max_threads();
//    std::vector<std::vector<ZBasisBase::state_t>> tls_send_state(nthreads);
//    std::vector<std::vector<coeff_t>> tls_send_dy(nthreads);

    auto N = send_state.size();
    send_state.resize(N+1); // need space for one past the end
    send_dy.resize(N+1);
    // apply to all local basis vectors, il = local state index
    BENCH_TIMER_TIMEIT(initial_apply_timer,

    for ( const auto& [c, op] : this->ops.off_diag_terms ){
        for (ZBasisBase::idx_t il = 0; il < ctx.local_block_size(); ++il) {
            ZBasisBase::state_t og_state = this->basis[il];
            auto state = og_state;
            auto sign = op.applyState(state);

            assert(sign == 0 || sign == 1 || sign == -1);
           // if (sign == 0) continue;
            
            auto target_rank = ctx.rank_of_state(state);
            auto dy = c*x[il]*sign;

//            if (target_rank == ctx.my_rank){
//                // immediately apply
//                ZBasisBase::idx_t local_idx;
//                ASSERT_STATE_FOUND("local",
//                    state,
//                    basis.search(state, local_idx)
//                    );
//                y[local_idx] += dy;
//            }
    
            MPI_Count& pos = send_cursors[target_rank];
            send_state[pos] = state;
            send_dy[pos] = dy;

            pos += (sign !=0); // overwrite if not needed
            assert(pos < static_cast<MPI_Count>(send_state.size()));
            assert(pos < static_cast<MPI_Count>(send_dy.size()));
         }
    }
    );

    send_state.resize(N); // need space for one past the end
    send_dy.resize(N);


    std::vector<MPI_Request> send_reqs;
    std::vector<MPI_Request> recv_reqs;


    const int STATE_REQ = 0x10000;
    const int COEFF_REQ = 0x20000;

    for (int r=0; r<ctx.world_size; r++){
        // make sure we did this correctly -- negligible cost
        assert(send_cursors[r] == send_displs[r] + send_counts[r]);

        // these don't need sending or receiving
        if (r==ctx.my_rank) continue; 

        if (recv_counts[r] > 0){
            MPI_Request req_state, req_dy;
            MPI_Irecv(recv_state.data() + recv_displs[r], recv_counts[r], 
                    get_mpi_type<ZBasisBST::state_t>(), r, STATE_REQ, 
                    MPI_COMM_WORLD, &req_state); 
            MPI_Irecv(recv_dy.data() + recv_displs[r], recv_counts[r], 
                    get_mpi_type<coeff_t>(), r, COEFF_REQ, 
                    MPI_COMM_WORLD, &req_dy); 

            recv_reqs.emplace_back(req_state);
            recv_reqs.emplace_back(req_dy);
        }

        if (send_counts[r] > 0){
            MPI_Request req_state, req_dy;
            MPI_Isend(send_state.data() + send_displs[r], send_counts[r],
                    get_mpi_type<ZBasisBST::state_t>(), r, STATE_REQ, 
                    MPI_COMM_WORLD, &req_state); 
            MPI_Isend(send_dy.data() + send_displs[r], send_counts[r],
                    get_mpi_type<coeff_t>(), r, COEFF_REQ, 
                    MPI_COMM_WORLD, &req_dy);
            send_reqs.emplace_back(req_state);
            send_reqs.emplace_back(req_dy);
        }
        
    }


    assert(send_counts[ctx.my_rank] == recv_counts[ctx.my_rank]);
    const auto loc_send_offset = send_displs[ctx.my_rank];

    BENCH_TIMER_TIMEIT(loc_apply_timer,
    for (int i=loc_send_offset; 
            i<loc_send_offset+send_counts[ctx.my_rank]; i++){
        ZBasisBase::idx_t local_idx;
        ASSERT_STATE_FOUND("local",
            send_state[i],
            this->basis.search(send_state[i], local_idx)
            );
        y[local_idx] += send_dy[i];
    }
    );

    #ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
        int num_send_neighbors = 0, num_recv_neighbors = 0;
        for (int r = 0; r < ctx.world_size; r++) {
            if (r != ctx.my_rank && send_counts[r] > 0) num_send_neighbors++;
            if (r != ctx.my_rank && recv_counts[r] > 0) num_recv_neighbors++;
        }
        
        ctx.log << "[ rank "<<ctx.my_rank<<" ] " << num_send_neighbors << " send neighbors, "
                  << num_recv_neighbors << " recv neighbors" << std::endl;
        
    #endif

    // synchronise
    BENCH_TIMER_TIMEIT(remx_wait_timer,
    MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
    );
    

    BENCH_TIMER_TIMEIT(rem_apply_timer,
    // Applying rank-local updates we received remotely
    for (int r=0; r<ctx.world_size; r++){
        const auto rem_displs = recv_displs[r];
        for (int i = rem_displs; 
                i < rem_displs + recv_counts_no_self[r]; ++i) {
            ZBasisBase::idx_t local_idx;
            ASSERT_STATE_FOUND("remote",
                recv_state[i],
                this->basis.search(recv_state[i], local_idx)
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
BasisTransferWisdom MPILazyOpSumBase<coeff_t, basis_t>::find_optimal_basis_load() {

    std::vector<int> all_hardness(ctx.world_size);
    int my_hardness=0;

    // estimate the work of my rank
    for (ZBasisBase::idx_t il = 0; il < ctx.local_block_size(); ++il) {
        for (auto& [c, op] : ops.off_diag_terms ) {
            ZBasisBase::state_t state = basis[il];
            auto sign = op.applyState(state);
            if (sign == 0) continue;
            my_hardness++;
//            auto target_rank = ctx.rank_of_state(state);
//            naive_send_counts[target_rank]++;
        }
    }
    MPI_Allgather(&my_hardness, 1, MPI_INT, all_hardness.data(), 1, MPI_INT, MPI_COMM_WORLD);

    if (ctx.my_rank == 0)
        printvec(std::cout<<"[Main] Global hardness: ", all_hardness)<<std::endl;


    // rebalance the load
    const auto partition = ctx.get_rebalance_plan(all_hardness);
    const auto& my_send_partition = partition[ctx.my_rank];

    // my_send_partition is a list of pairs [ (r0, n0), (r1, n1), (r2, n2) ]
    // notes: 
    // i) the ranks are strictly sequential and increasing
    // ii) the sum of the n's must add up to the local size
#ifndef NDEBUG
    {
        int r_prev=-1;
        int acc=0;
        for (auto& [r, n] : my_send_partition){
            assert(r > r_prev);
            r_prev = r;
            acc += n;
        }
        assert(acc == ctx.local_block_size());
    }
#endif

    // each rank knows its local send_counts, must be told the remote send sizes
    // we know that we need to send the basis states
    // [0 ... partition_local_idx[0]) -> rank my_send_partition[0].first
    // [0 ... partition_local_idx[1]) -> rank my_send_partition[1].first
    BasisTransferWisdom btw;

    // for global exchange
    std::vector<MPI_Count> all_b_send_counts(ctx.world_size, 0); // number of elements in send_counts of each rank
    std::vector<MPI_Count> all_b_recv_counts(ctx.world_size, 0); // number of elements to be received by each rank        

    ctx.log<<"<basis rebalance>\n";

    for (size_t j=0; j<my_send_partition.size(); j++){
        auto& [target_r, count] = my_send_partition[j];

        all_b_send_counts[target_r] = count;
        btw.send_counts.push_back(count); // the number of BASIS indices to send
        btw.send_ranks.push_back(target_r);
        ctx.log << "("<<count<<"records ) -> "<<target_r<<"\n";
    }


    MPI_Alltoall(all_b_send_counts.data(), 1, get_mpi_type<MPI_Count>(), all_b_recv_counts.data(), 1, get_mpi_type<MPI_Count>(), MPI_COMM_WORLD);
    for (int r=0; r<ctx.world_size; r++){
        auto n = all_b_recv_counts[r];
        if ( n != 0){
            btw.recv_ranks.push_back(r);
            btw.recv_counts.push_back(n);
        }
    }

    // figure out what the terminal indices ought to be
    btw.idx_partition.clear();
    btw.idx_partition.resize(ctx.world_size+1, 0);

    MPI_Count my_new_dimension = std::accumulate(btw.recv_counts.begin(), btw.recv_counts.end(), 0);
    std::vector<MPI_Count> all_dimensions(ctx.world_size);
    MPI_Allgather(&my_new_dimension, 1, get_mpi_type<MPI_Count>(), all_dimensions.data(), 1, get_mpi_type<MPI_Count>(), MPI_COMM_WORLD);

    btw.idx_partition[0] = 0;
    for(int r=0; r<ctx.world_size; r++){
        btw.idx_partition[r+1] = btw.idx_partition[r] + all_dimensions[r];
    }

    ctx.log<<"Sugested basis rebalance:";
    printvec(ctx.log<<"\n[b_send_counts] ", btw.send_counts);
    printvec(ctx.log<<"\n[b_send_ranks] ", btw.send_ranks);
    printvec(ctx.log<<"\n[b_recv_counts] ", btw.recv_counts);
    printvec(ctx.log<<"\n[b_recv_ranks] ", btw.recv_ranks);

    ctx.log<<"\n</basis rebalance>"<<std::endl;

    return btw;
}




template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSumBatched<coeff_t, basis_t>::allocate_temporaries() {
    // runs through the current local basis applying op to everything.
    // We cound how many we want to send to each rank, then exchange synchronously.
    // We can then resize recv_states_bufs appropriately.
    send_counts.resize(this->ctx.world_size);
    send_displs.resize(this->ctx.world_size);

    recv_counts.resize(this->ctx.world_size);
    recv_displs.resize(this->ctx.world_size);

    std::fill(send_counts.begin(), send_counts.end(), 0);
    std::fill(recv_counts.begin(), recv_counts.end(), 0);
    std::fill(send_displs.begin(), send_displs.end(), 0);
    std::fill(recv_displs.begin(), recv_displs.end(), 0);


    // set up the real send_counts
    // TODO this should be computable just from known information
    for (ZBasisBase::idx_t il = 0; il < this->ctx.local_block_size(); ++il) {
        for (auto& [c, op] : this->ops.off_diag_terms ) {
            ZBasisBase::state_t state = this->basis[il];
            auto sign = op.applyState(state);
            if (sign == 0) continue;
            
            auto target_rank = this->ctx.rank_of_state(state);
            send_counts[target_rank]++;
        }
    }


    // allocate auxiliary arrays

    MPI_Alltoall(send_counts.data(), 1, get_mpi_type<MPI_Count>(),
            recv_counts.data(), 1, get_mpi_type<MPI_Count>(), MPI_COMM_WORLD);

    const int total_send =
        std::accumulate(send_counts.begin(), send_counts.end(), 0);

    const int total_recv =
        std::accumulate(recv_counts.begin(), recv_counts.end(), 0);


    // one past the end for all (spacer needed)
    for (int r=1; r<this->ctx.world_size; r++){
        send_displs[r] = send_counts[r-1] + send_displs[r-1] + 1;
        recv_displs[r] = recv_counts[r-1] + recv_displs[r-1];
    }
    send_state.resize(total_send + this->ctx.world_size);
    send_dy.resize(total_send + this->ctx.world_size);

    recv_state.resize(total_recv);
    recv_dy.resize(total_recv);

    // logging
    this->ctx.log <<"[alloc] Send Sizes: ";
    for (auto d : send_counts) this->ctx.log << d <<", ";
    this->ctx.log<<"\n\ttotal:"<<total_send<<std::endl;

    this->ctx.log <<"[alloc] Send Displacements: ";
    for (auto d : send_displs) this->ctx.log << d <<", ";
    this->ctx.log<<std::endl;

    // check that things fit into int
    for (int r = 0; r < this->ctx.world_size; r++) {
        if (send_counts[r] > INT_MAX || recv_counts[r] > INT_MAX) {
            throw std::runtime_error("MPI count overflow: message size exceeds INT_MAX");
        }
    }

}



// explicit template instantiations: generate symbols to link with
template struct MPILazyOpSumBase<double, ZBasisBST_MPI>;
template struct MPILazyOpSumBatched<double, ZBasisBST_MPI>;
template struct MPILazyOpSumPipe<double, ZBasisBST_MPI>;
template struct MPILazyOpSumPipePrealloc<double, ZBasisBST_MPI>;

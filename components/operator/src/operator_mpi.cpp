#include "operator_mpi.hpp"
#include <mpi.h>
#include <cassert>
#include <fstream>
#include "timeit.hpp"
#include <numeric>
#include <omp.h>
#include <stdexcept>

const char* to_string(ApplyStrategy s) {
    switch (s) {
        case ApplyStrategy::Serial:      return "serial";
        case ApplyStrategy::Omp:         return "omp";
        case ApplyStrategy::OmpOwner:    return "omp-owner";
        case ApplyStrategy::OmpPrefetch: return "omp-prefetch";
    }
    return "serial";
}

ApplyStrategy parse_strategy(const std::string& s) {
    if (s == "serial")       return ApplyStrategy::Serial;
    if (s == "omp")          return ApplyStrategy::Omp;
    if (s == "omp-owner")    return ApplyStrategy::OmpOwner;
    if (s == "omp-prefetch") return ApplyStrategy::OmpPrefetch;
    throw std::runtime_error("Unknown apply strategy '" + s +
            "' (expected: serial | omp | omp-owner | omp-prefetch)");
}

#ifndef NDEBUG
#define ASSERT_STATE_FOUND(error_msg, state, result) \
    do { \
        if (result == 0) { \
            std::cerr << "State not found on rank " << ctx.my_rank << ": "; \
            printHex(std::cerr, state) << "\n"; \
            throw std::logic_error("State not found in " error_msg); \
        } \
    } while(0)

#else
#define ASSERT_STATE_FOUND(error_msg, state, result) result
#endif

#ifdef DEBUG
#define DEBUG_PRINT_VEC(msg, op_index, vector, ctx) \
            ctx.log(logging::DEBUG) << msg<<" (op "<<op_index<<") [node "<<ctx.my_rank<< "]\n";\
            for (int r=0; r<ctx.world_size; r++){\
                if (r == ctx.my_rank) ctx.log(logging::DEBUG)<<"*";\
                ctx.log(logging::DEBUG) << "\tvector["<<r<<"] -> "<<curr_op_comm.send_states[r].size() <<"\n";\
            }
#else
#define DEBUG_PRINT_VEC(msg, op_index, vector, ctx)
#endif

// reads only the local basis into memory
inline std::vector<Uint128> read_basis_hdf5_MPI(
        const std::string& infile,
        const char* dset_name = "basis"
        ){

	std::vector<Uint128> result;

    int world_size, my_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

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
        
        // Local chunk indices (by global index)
        uint64_t chunk = total_rows / world_size;
        int64_t rem   = total_rows % world_size;

        uint64_t local_count = chunk + (my_rank < rem ? 1 : 0);
    
        uint64_t my_offset = my_rank * chunk + std::min<uint64_t>(my_rank, rem);
		
        // read the slab in [local_start ... local_end)
        if (local_count > 0) {
		    // Allocate memory for the result
            result.resize(local_count);

            // Select hyperslab in file dataspace
            hsize_t file_offset[2] = { my_offset, 0 };
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
        logging::log(logging::DEBUG)<<"[r"<<my_rank<<"] Loaded basis chunk.\n";
		
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




template<typename B>
void ZBasisMPI<B>::load_raw(const fs::path& bfile, const std::string& dataset){
    logging::log(logging::INFO) << "Loading basis from file " << bfile <<"\n";

    if (bfile.stem().extension() == ".partitioned"){
        assert(bfile.extension() == ".h5");
        this->states = read_basis_hdf5_MPI(bfile, dataset.c_str());
    } else if (bfile.extension() == ".h5"){
        assert(dataset=="basis");
        this->states = read_basis_hdf5_MPI(bfile, "basis");
    } else {
        throw std::runtime_error(
                "Bad basis format: file must end with .csv or .h5");
    }
}

template<typename B>
void ZBasisMPI<B>::redistribute(){
    MPIHashContext ctx;
    ctx.log(logging::DEBUG)<<"[r"<<ctx.my_rank<<"] Transfer states to correct ranks...\n";
    tfer_states_to_correct_ranks(ctx);
    ctx.log(logging::DEBUG) << "Done!\n";
}

template<typename B>
void ZBasisMPI<B>::load_from_file(const fs::path& bfile, const std::string& dataset){
    load_raw(bfile, dataset);
    redistribute();
}


// Redistribute states to the correct ranks via hash-based partitioning,
// then sort the local partition and rebuild any search-acceleration structures.
template<typename B>
void ZBasisMPI<B>::tfer_states_to_correct_ranks(MPIHashContext& ctx){
    constexpr size_t S = sizeof(ZBasisBase::state_t);
    auto log_mem = [&](const char* phase) {
        size_t rss = rss_bytes();
        ctx.log(logging::DEBUG) << "[tfer r" << ctx.my_rank << " " << phase << "]"
                  << "  n=" << this->size()
                  << "  cap=" << this->states.capacity()
                  << "  states_MiB=" << this->states.capacity() * S / (1<<20)
                  << "  rss_MiB=" << rss / (1<<20)
                  << "\n" << std::flush;
    };

    std::vector<ZBasisBase::state_t> recv_states;

    std::vector<int> send_counts(ctx.world_size, 0);
    std::vector<int> recv_counts(ctx.world_size);
    std::vector<int> send_displs(ctx.world_size, 0);
    std::vector<int> recv_displs(ctx.world_size, 0);

    log_mem("entry");

    for (const auto& psi : this->states)
        send_counts[ctx.rank_of_state(psi)]++;

    MPI_Request r1;
    MPI_Ialltoall(send_counts.data(), 1, get_mpi_type<int>(),
            recv_counts.data(), 1, get_mpi_type<int>(), MPI_COMM_WORLD, &r1);

    for (int r = 1; r < ctx.world_size; r++)
        send_displs[r] = send_displs[r-1] + send_counts[r-1];

    // Bucket-sort this->states in-place by destination rank.
    // This lets us use it directly as the MPI send buffer, avoiding a
    // separate send_states allocation and keeping peak at 2× basis size
    // (this->states + recv_states) rather than 3×.
    {
        std::vector<ZBasisBase::state_t> sorted(this->size());
        std::vector<int> counters(send_displs);
        for (int il = 0; il < this->size(); il++){
            auto rank = ctx.rank_of_state(this->states[il]);
            sorted[counters[rank]] = this->states[il];
            counters[rank]++;
        }
        std::swap(sorted, this->states);
        // sorted (old unsorted states) freed here
    }
    log_mem("post-sort");  // old capacity freed; this->states is trimmed size

    MPI_Wait(&r1, MPI_STATUS_IGNORE);
    recv_states.resize(std::accumulate(recv_counts.begin(), recv_counts.end(), 0ull));
    for (int r = 1; r < ctx.world_size; r++)
        recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];

    ctx.log(logging::DEBUG) << "[tfer r" << ctx.my_rank << " pre-alltoallv]"
              << "  send_MiB=" << this->states.size() * S / (1<<20)
              << "  recv_MiB=" << recv_states.size() * S / (1<<20)
              << "  rss_MiB=" << rss_bytes() / (1<<20)
              << "\n" << std::flush;

    MPI_Alltoallv(this->states.data(), send_counts.data(), send_displs.data(), get_mpi_type<ZBasisBase::state_t>(),
            recv_states.data(), recv_counts.data(), recv_displs.data(), get_mpi_type<ZBasisBase::state_t>(), MPI_COMM_WORLD);

    // Release the send data before taking ownership of recv (keeps peak at 1×).
    { std::vector<ZBasisBase::state_t> tmp; std::swap(tmp, this->states); }

    std::swap(recv_states, this->states);
    log_mem("post-alltoallv");  // send buffer freed, recv now in this->states

    finalize_local_partition(ctx);
    log_mem("post-finalize");  // sorted, bounds built, dims populated
}

// Sort the local partition, rebuild search-acceleration structures, and
// populate global_dim / dim_of_rank metadata. Shared by redistribute() and
// ingest_shard_streaming().
template<typename B>
void ZBasisMPI<B>::finalize_local_partition(MPIHashContext& ctx){
    std::sort(this->states.begin(), this->states.end());

    // Rebuild search-acceleration structures (bounds, sentinels, …) for
    // whichever LocalBasis is being used.
    this->on_states_changed();

    ZBasisBase::idx_t my_size = this->size();
    _all_rank_dims.resize(ctx.world_size);
    MPI_Allgather(&my_size, 1, get_mpi_type<ZBasisBase::idx_t>(),
            _all_rank_dims.data(), 1, get_mpi_type<ZBasisBase::idx_t>(), MPI_COMM_WORLD);
    _global_dim = std::accumulate(_all_rank_dims.begin(), _all_rank_dims.end(),
            static_cast<ZBasisBase::idx_t>(0));
}

// Streaming hash-redistribution straight off a rank-local binary shard.
// See the header for the memory rationale. Each round every rank reads one
// block, filters it, hash-buckets it, and participates in a collective count
// exchange + Alltoallv; owned states are appended to this->states. Rounds
// continue until no rank read any records in a round (so a rank whose shard
// drains early keeps taking part in empty rounds). The Alltoallv is collective,
// so the loop trip count is identical on every rank by construction.
template<typename B>
void ZBasisMPI<B>::ingest_shard_streaming(const fs::path& shard_path,
        size_t block_records, const StateFilter& filter){
    using state_t = ZBasisBase::state_t;
    MPIHashContext ctx;
    if (block_records == 0) block_records = (1u << 20);

    // ShardWriter always creates the file (even for a rank that found nothing),
    // but be defensive: a missing shard is treated as an empty one.
    std::ifstream in(shard_path, std::ios::binary);

    std::vector<state_t> block;
    std::vector<state_t> send_buf;
    std::vector<int> send_counts(ctx.world_size), recv_counts(ctx.world_size);
    std::vector<int> send_displs(ctx.world_size), recv_displs(ctx.world_size);

    this->states.clear();

    while (true) {
        // Read up to block_records states; got == 0 means this shard is drained.
        block.resize(block_records);
        size_t got = 0;
        if (in) {
            in.read(reinterpret_cast<char*>(block.data()),
                    block_records * sizeof(state_t));
            got = static_cast<size_t>(in.gcount()) / sizeof(state_t);
        }
        block.resize(got);

        if (filter && !block.empty()) filter(block);

        // Hash-bucket the (possibly now-shorter) block by destination rank.
        std::fill(send_counts.begin(), send_counts.end(), 0);
        for (const auto& psi : block)
            send_counts[ctx.rank_of_state(psi)]++;

        send_displs[0] = 0;
        for (int r = 1; r < ctx.world_size; r++)
            send_displs[r] = send_displs[r-1] + send_counts[r-1];

        send_buf.resize(block.size());
        {
            std::vector<int> counters(send_displs);
            for (const auto& psi : block)
                send_buf[counters[ctx.rank_of_state(psi)]++] = psi;
        }

        MPI_Alltoall(send_counts.data(), 1, get_mpi_type<int>(),
                recv_counts.data(), 1, get_mpi_type<int>(), MPI_COMM_WORLD);

        recv_displs[0] = 0;
        for (int r = 1; r < ctx.world_size; r++)
            recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
        size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), 0ull);

        // Append received states directly onto this->states.
        size_t old = this->states.size();
        this->states.resize(old + recv_total);
        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                get_mpi_type<state_t>(),
                this->states.data() + old, recv_counts.data(), recv_displs.data(),
                get_mpi_type<state_t>(), MPI_COMM_WORLD);

        // Stop once no rank read anything this round.
        int local_active = block.empty() ? 0 : 1;
        int any_active = 0;
        MPI_Allreduce(&local_active, &any_active, 1, get_mpi_type<int>(),
                MPI_MAX, MPI_COMM_WORLD);
        if (!any_active) break;
    }

    finalize_local_partition(ctx);
}

template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const {
    for (const auto& term : ops.diagonal_terms) {
        const auto& c = term.first;   
        const auto& op = term.second;

        assert(op.is_diagonal());
       
        // there is no need to communicate, it's literally just this??
//        #pragma omp parallel for schedule(static)
        for (ZBasisBase::idx_t i = 0; i<basis.dim(); ++i){
            ZBasisBase::state_t psi = basis[i];
            coeff_t dy = c * x[i] * static_cast<double>(op.applyState(psi));
            assert(psi == basis[i]);
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

        std::vector<std::vector<coeff_t>> recv_dy_bufs;
        std::vector<std::vector<ZBasisBase::state_t>> recv_states_bufs;

        MPI_Request count_exchange_req;
        std::vector<int> recvcounts;

        bool count_exchange_done = false;

        void resize(int world_size){
            send_dy.resize(world_size);
            send_states.resize(world_size);

            recv_dy_bufs.resize(world_size);
            recv_states_bufs.resize(world_size);

            recvcounts.resize(world_size);
        }

        void reset_for_new_op(){
            count_exchange_done=false;
            requests.clear();
            for (auto& v : send_dy)     v.clear();
            for (auto& v : send_states) v.clear();
            for (auto& v : recv_dy_bufs)     v.clear();
            for (auto& v : recv_states_bufs) v.clear();
        }
    };

    Timer loc_apply_timer("[local apply]", ctx.my_rank);
    Timer loc_up_timer("[local update]", ctx.my_rank);
    Timer rem_up_timer("[remote update]", ctx.my_rank);
    Timer countx_wait_timer("[count exchange wait]", ctx.my_rank);
    Timer countx_wait_timer_2("[count exchange wait 2]", ctx.my_rank);
    Timer remx_wait_timer("[remote exchange wait]", ctx.my_rank);

    std::vector<const Timer*> timers{&loc_apply_timer, &loc_up_timer, &rem_up_timer,
        &countx_wait_timer, &countx_wait_timer_2, &remx_wait_timer};

    OperatorCommState prev_op_comm;
    OperatorCommState curr_op_comm;

    // Pre-size the send buffers once; reset_for_new_op() will only clear them
    prev_op_comm.resize(ctx.world_size);
    curr_op_comm.resize(ctx.world_size);

    bool has_prev_op = false;

    // Post receives for operator prev_index's data into comm, wait for all of
    // its communication to finish, then apply the received updates to y.
    // Called once per iteration for the previous operator, and once after the
    // loop for the final operator.
    auto process_receives = [&](OperatorCommState& comm, int prev_index) {
        BENCH_TIMER_TIMEIT(countx_wait_timer,
        // Wait for the count exchange if not done
        // (should already be complete; this is just for safety)
        if (!comm.count_exchange_done) {
            MPI_Wait(&comm.count_exchange_req, MPI_STATUS_IGNORE);
            comm.count_exchange_done = true;
        }
        )

        DEBUG_PRINT_VEC(">> recv ", prev_index, comm.recvcounts, ctx)

        // Every rank sends to every rank: recv buf per source is just recvcounts[source]
        for (int source = 0; source < ctx.world_size; ++source) {
            if (source == ctx.my_rank) continue;
            int cnt = comm.recvcounts[source];
            comm.recv_states_bufs[source].resize(cnt);
            comm.recv_dy_bufs[source].resize(cnt);
            if (cnt == 0) continue;

            comm.requests.push_back(MPI_Request{});
            MPI_Irecv(comm.recv_states_bufs[source].data(),
                     cnt, get_mpi_type<ZBasisBase::state_t>(),
                     source, 10*prev_index + 1, MPI_COMM_WORLD, &comm.requests.back());

            comm.requests.push_back(MPI_Request{});
            MPI_Irecv(comm.recv_dy_bufs[source].data(),
                     cnt, get_mpi_type<coeff_t>(),
                     source, 10*prev_index + 2, MPI_COMM_WORLD, &comm.requests.back());
        }

        BENCH_TIMER_TIMEIT(remx_wait_timer,
        if (!comm.requests.empty()) {
            MPI_Waitall(comm.requests.size(), comm.requests.data(),
                       MPI_STATUSES_IGNORE);
        }
        )

        BENCH_TIMER_TIMEIT(rem_up_timer,
        for (int source = 0; source < ctx.world_size; ++source) {
            if (source == ctx.my_rank) continue;
            for (size_t j = 0; j < comm.recv_states_bufs[source].size(); ++j) {
                ZBasisBase::idx_t local_idx;
                ASSERT_STATE_FOUND("remote", comm.recv_states_bufs[source][j],
                        basis.search(comm.recv_states_bufs[source][j], local_idx));
                y[local_idx] += comm.recv_dy_bufs[source][j];
            }
        }
        )
    };

    int op_index = 0;
    for ( const auto& [c, op] : ops.off_diag_terms ){
        curr_op_comm.reset_for_new_op();

         // Organize sends by destination rank
        BENCH_TIMER_TIMEIT(loc_apply_timer,

        for (ZBasisBase::idx_t il = 0; il < basis.dim(); ++il) {
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
        


        BENCH_TIMER_TIMEIT(loc_up_timer,
        for (size_t i = 0; i < curr_op_comm.send_states[ctx.my_rank].size(); ++i) {
            ZBasisBase::idx_t local_idx;

            ASSERT_STATE_FOUND("self",
                    curr_op_comm.send_states[ctx.my_rank][i],
                    basis.search(curr_op_comm.send_states[ctx.my_rank][i], local_idx)
                    );

            y[local_idx] += curr_op_comm.send_dy[ctx.my_rank][i];
        }
        )


        // === PROCESS PREVIOUS OPERATOR'S RECEIVES ===
        if (has_prev_op) {
            process_receives(prev_op_comm, op_index - 1);
        }

        // === DATA SENDS FOR CURRENT OPERATOR ===
        BENCH_TIMER_TIMEIT(countx_wait_timer_2,
        // Wait for current operator's count exchange to complete
        MPI_Wait(&curr_op_comm.count_exchange_req, MPI_STATUS_IGNORE);
        curr_op_comm.count_exchange_done = true;
        )

        // Begin sending to all nonempty, non-self targets
        for (int target_rank=0; target_rank<ctx.world_size; target_rank++){
            if (target_rank == ctx.my_rank || 
                    curr_op_comm.send_states[target_rank].empty()) continue;

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
        std::swap(curr_op_comm, prev_op_comm);
        has_prev_op = true;
        op_index++;

    } // end operator loop


    // === PROCESS FINAL OPERATOR'S RECEIVES ===
    if (has_prev_op) {
        process_receives(prev_op_comm, op_index - 1);
    }


// print diagnostics
#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
        for (auto t : timers){
            t->print_summary(ctx.log(logging::DEBUG));
        }
#endif


}


// 4-pass LSB radix sort of (state, coeff) pairs by 128-bit state key.
// Pass order: bits 0-31, 32-63, 64-95, 96-127.
// states[] and coeffs[] are permuted identically.
template <RealOrCplx coeff_t>
static void radix_sort_pairs(
        ZBasisBase::state_t* states, coeff_t* coeffs, int64_t N)
{
    if (N <= 1) return;
    using state_t = ZBasisBase::state_t;

    std::vector<state_t> tmp_states(N);
    std::vector<coeff_t> tmp_coeffs(N);

    // Use 16-bit radix: 8 passes, histogram size = 65536.
    // (4 passes of 32-bit would need a 4 GB histogram — too large.)
    constexpr int RADIX_BITS = 16;
    constexpr int RADIX_SIZE = 1 << RADIX_BITS; // 65536
    constexpr int RADIX_MASK = RADIX_SIZE - 1;
    constexpr int N_PASSES   = 128 / RADIX_BITS; // 8 passes

    std::vector<int64_t> hist(RADIX_SIZE);

    state_t* src_s  = states;
    coeff_t* src_c  = coeffs;
    state_t* dst_s  = tmp_states.data();
    coeff_t* dst_c  = tmp_coeffs.data();

    for (int pass = 0; pass < N_PASSES; pass++) {
        // Which 16-bit chunk of the 128-bit key?
        // pass 0: bits 0-15 (uint64[0] low), pass 1: bits 16-31, ...
        int word = (pass * RADIX_BITS) / 64;      // 0 or 1
        int shift = (pass * RADIX_BITS) % 64;     // 0,16,32,48

        // Build histogram
        std::fill(hist.begin(), hist.end(), 0);
        for (int64_t i = 0; i < N; i++) {
            uint32_t key = (uint32_t)((src_s[i].uint64[word] >> shift) & RADIX_MASK);
            hist[key]++;
        }
        // Prefix sum -> starting positions
        int64_t sum = 0;
        for (int b = 0; b < RADIX_SIZE; b++) {
            int64_t cnt = hist[b];
            hist[b] = sum;
            sum += cnt;
        }
        // Scatter
        for (int64_t i = 0; i < N; i++) {
            uint32_t key = (uint32_t)((src_s[i].uint64[word] >> shift) & RADIX_MASK);
            int64_t pos = hist[key]++;
            dst_s[pos] = src_s[i];
            dst_c[pos] = src_c[i];
        }
        std::swap(src_s, dst_s);
        std::swap(src_c, dst_c);
    }

    // After N_PASSES (8, even) swaps src_s == states (original buffers).
    // Data is already in the right place — no copy needed when N_PASSES is even.
    static_assert(N_PASSES % 2 == 0, "N_PASSES must be even so result stays in original buffers");
}

// Walk sorted recv pairs + sorted basis simultaneously and accumulate y.
template <RealOrCplx coeff_t, Basis B>
static void linear_merge_update(
        const ZBasisBase::state_t* recv_states, const coeff_t* recv_dy,
        int64_t recv_total,
        const B& basis, coeff_t* y)
{
    int64_t ri = 0;
    ZBasisBase::idx_t bi = 0;
    int64_t bdim = basis.dim();
    while (ri < recv_total && bi < bdim) {
        if (recv_states[ri] == basis[bi]) {
            y[bi] += recv_dy[ri];
            ri++;
        } else if (recv_states[ri] < basis[bi]) {
            // state not in local basis (hash collision artefact); skip
            ri++;
        } else {
            bi++;
        }
    }
}

template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::evaluate_add_off_diag_batched(const coeff_t* x, coeff_t* y) {
    assert(!send_counts.empty() && "allocate_temporaries() must be called first");

    using state_t = ZBasisBase::state_t;
    int N        = ctx.world_size;
    int my_rank  = ctx.my_rank;
    int64_t dim  = (int64_t)basis.dim();
    int64_t bs   = (batch_size <= 0) ? dim : std::min((int64_t)batch_size, dim);

    Timer loc_apply_timer("[batched local apply]",    my_rank);
    Timer self_update_timer("[batched self update]",  my_rank);
    Timer alltoallv_timer("[batched mpi alltoallv]",  my_rank);
    Timer radix_sort_timer("[batched radix sort]",    my_rank);
    Timer remote_up_timer("[batched remote update]",  my_rank);

    std::vector<const Timer*> timers{&loc_apply_timer, &self_update_timer,
        &alltoallv_timer, &radix_sort_timer, &remote_up_timer};

    // Per-batch MPI count arrays (int for standard MPI_Alltoallv API)
    std::vector<int> batch_sc(N), batch_rc(N);

    // Recv buffer for one batch (reused each round).
    // Worst case = recv_counts total (sized in allocate_temporaries).
    // Threshold: use sort+merge when recv data > 2× local basis size
    // This means we prefer sort+merge when the batch is large enough.
    const int64_t sort_merge_threshold = 2LL * dim * (int64_t)sizeof(state_t);

    for (int64_t bidx = 0; bidx < n_batches; bidx++) {
        // Ranks whose dim is exhausted still run the round with empty ranges
        // so the collectives below stay matched across all ranks.
        int64_t state_start = std::min(bidx * bs, dim);
        int64_t state_end   = std::min(state_start + bs, dim);

        // --- LOCAL APPLY PASS ---
        // Reset send cursors to the start of each rank's slot in the flat buffer.
        std::vector<MPI_Count> cursors(send_displs.begin(), send_displs.end());

        BENCH_TIMER_TIMEIT(loc_apply_timer,
        for (int64_t il = state_start; il < state_end; ++il) {
            for (const auto& [c, op] : ops.off_diag_terms) {
                state_t state = basis[il];
                auto sign = op.applyState(state);
                if (sign == 0) continue;

                int r = (int)ctx.rank_of_state(state);
                coeff_t dy = c * x[il] * (coeff_t)sign;

                send_state[cursors[r]] = state;
                send_dy[cursors[r]]    = dy;
                cursors[r]++;
            }
        }
        ) // BENCH_TIMER_TIMEIT

        // Compute actual per-rank send counts for this batch
        for (int r = 0; r < N; r++)
            batch_sc[r] = (int)(cursors[r] - send_displs[r]);

        // --- SELF-UPDATE ---
        BENCH_TIMER_TIMEIT(self_update_timer,
        {
            int cnt = batch_sc[my_rank];
            MPI_Count base = send_displs[my_rank];
            for (int i = 0; i < cnt; i++) {
                ZBasisBase::idx_t local_idx;
                ASSERT_STATE_FOUND("batched self",
                    send_state[base + i],
                    basis.search(send_state[base + i], local_idx));
                y[local_idx] += send_dy[base + i];
            }
        }
        ) // BENCH_TIMER_TIMEIT

        // --- ONE COMMUNICATION ROUND ---
        // Exchange counts
        MPI_Alltoall(batch_sc.data(), 1, MPI_INT, batch_rc.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Compute send/recv displacements for this batch.
        // Send side: each rank's slot starts at send_displs[r] in the flat buffer.
        // Recv side: pack contiguously (excluding self) so update loop has no gaps.
        std::vector<int> alltoallv_sd(N), alltoallv_rd(N);
        for (int r = 0; r < N; r++)
            alltoallv_sd[r] = (int)send_displs[r];

        // Packed recv displacements — skip my_rank slot so buffer has no gap.
        alltoallv_rd[0] = 0;
        for (int r = 1; r < N; r++)
            alltoallv_rd[r] = alltoallv_rd[r-1] + (r-1 != my_rank ? batch_rc[r-1] : 0);
        int64_t batch_recv_total = 0;
        for (int r = 0; r < N; r++)
            if (r != my_rank) batch_recv_total += batch_rc[r];

        // Zero self counts so Alltoallv skips the self slot entirely.
        batch_sc[my_rank] = 0;
        batch_rc[my_rank] = 0;

        BENCH_TIMER_TIMEIT(alltoallv_timer,
        MPI_Alltoallv(
            send_state.data(), batch_sc.data(), alltoallv_sd.data(), get_mpi_type<state_t>(),
            recv_state.data(), batch_rc.data(), alltoallv_rd.data(), get_mpi_type<state_t>(),
            MPI_COMM_WORLD);
        MPI_Alltoallv(
            send_dy.data(), batch_sc.data(), alltoallv_sd.data(), get_mpi_type<coeff_t>(),
            recv_dy.data(), batch_rc.data(), alltoallv_rd.data(), get_mpi_type<coeff_t>(),
            MPI_COMM_WORLD);
        ) // BENCH_TIMER_TIMEIT

        // --- REMOTE UPDATE ---
        bool use_sort_merge = (batch_recv_total * (int64_t)(sizeof(state_t) + sizeof(coeff_t))
                               > sort_merge_threshold);

        if (use_sort_merge && batch_recv_total > 0) {
            BENCH_TIMER_TIMEIT(radix_sort_timer,
            radix_sort_pairs(recv_state.data(), recv_dy.data(), batch_recv_total);
            ) // BENCH_TIMER_TIMEIT

            BENCH_TIMER_TIMEIT(remote_up_timer,
            linear_merge_update(recv_state.data(), recv_dy.data(),
                                batch_recv_total, basis, y);
            ) // BENCH_TIMER_TIMEIT
        } else {
            BENCH_TIMER_TIMEIT(remote_up_timer,
            for (int64_t i = 0; i < batch_recv_total; i++) {
                ZBasisBase::idx_t local_idx;
                ASSERT_STATE_FOUND("batched remote",
                    recv_state[i],
                    basis.search(recv_state[i], local_idx));
                y[local_idx] += recv_dy[i];
            }
            ) // BENCH_TIMER_TIMEIT
        }
    } // end batch loop

#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
    for (auto t : timers)
        t->print_summary(ctx.log(logging::DEBUG));
#endif
}




// ---------------------------------------------------------------------------
// Threaded batched apply (ApplyStrategy::Omp / OmpOwner / OmpPrefetch)
//
// Shares the batch loop and MPI collectives with the serial batched path, but
// threads the three rank-local phases: local-apply scatter, self-update and
// remote-update. The receive-side search is the diagnosed bottleneck (chains of
// dependent DRAM misses), so it is what the strategies attack: OmpPrefetch
// recovers memory-level parallelism, OmpOwner removes the atomics.
// ---------------------------------------------------------------------------

// Interleaved, software-prefetched exact binary search over the sorted state
// array. Keeps GROUP probes in flight so several dependent cache misses overlap
// instead of serialising. Every query is assumed present (received states are
// always basis states); each out[i] is the index with S[out[i]] == q[i].
static inline void interleaved_search(
        const Uint128* __restrict S, int64_t dim,
        const Uint128* __restrict q, ZBasisBase::idx_t* __restrict out, int64_t n, 
        int GROUP=8)
{
    constexpr int MAX_GROUP=16;
    // Clamp here so oversized/invalid GROUP can never overrun lo[]/hi[].
    if (GROUP < 1) GROUP = 1;
    if (GROUP > MAX_GROUP) GROUP = MAX_GROUP;
    int64_t lo[MAX_GROUP], hi[MAX_GROUP];
    for (int64_t base = 0; base < n; base += GROUP) {
        int g = (int)std::min<int64_t>(GROUP, n - base);
        for (int j = 0; j < g; ++j) { lo[j] = 0; hi[j] = dim; }
        bool active = true;
        while (active) {
            active = false;
            for (int j = 0; j < g; ++j) {
                if (hi[j] - lo[j] > 1) {
                    active = true;
                    int64_t mid = (lo[j] + hi[j]) >> 1;
                    // S[mid] <= q  <=>  !(q < S[mid])
                    if (!(q[base + j] < S[mid])) lo[j] = mid; else hi[j] = mid;
                    __builtin_prefetch(&S[(lo[j] + hi[j]) >> 1], 0, 1);
                }
            }
        }
        for (int j = 0; j < g; ++j) out[base + j] = lo[j];
    }
}

template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::apply_records(
        const ZBasisBST::state_t* states, const coeff_t* dy, int64_t n,
        coeff_t* y, bool owner, bool prefetch)
{
    if (n == 0) return;
    ap_idx.resize(n);

    int APPLY_PREFETCH_GROUP=0;
    if (const char* s=std::getenv("APPLY_PREFETCH_GROUP")){
        APPLY_PREFETCH_GROUP=std::atoi(s);
    }
    if (APPLY_PREFETCH_GROUP == 0) APPLY_PREFETCH_GROUP=8; 
    // fall back to default if invalid specification

    // --- SEARCH PASS: resolve every record's local index (independent) ---
    if (prefetch) {
        const Uint128* S = basis.data();
        int64_t dim = (int64_t)basis.dim();
        #pragma omp parallel
        {
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int64_t lo = (int64_t)tid * n / nt;
            int64_t hi = (int64_t)(tid + 1) * n / nt;
            interleaved_search(S, dim, states + lo, ap_idx.data() + lo, hi - lo, 
                    APPLY_PREFETCH_GROUP);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            ZBasisBase::idx_t idx = 0;
            ASSERT_STATE_FOUND("threaded apply", states[i],
                    basis.search(states[i], idx));
            ap_idx[i] = idx;
        }
    }

    // --- UPDATE PASS: y[idx] += dy ---
    if (!owner) {
        // Duplicate target indices can occur within one batch, so guard with an
        // atomic. Independent searches already gave the memory-level parallelism.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            #pragma omp atomic
            y[ap_idx[i]] += dy[i];
        }
        return;
    }

    // Owner-computes: bucket records by owning thread = idx * T / dim, then each
    // thread applies its own contiguous bucket to a disjoint slice of y — no
    // atomics, and each thread only touches memory it (first-)owns.
    const int64_t dim = std::max<int64_t>(1, (int64_t)basis.dim());
    int T = omp_get_max_threads();
    auto bucket_of = [&](int64_t idx) {
        int b = (int)((idx * (int64_t)T) / dim);
        return b < T ? b : T - 1;
    };

    ap_thist.assign((size_t)T * T, 0);          // [tid*T + bucket]
    std::vector<int64_t> boffs(T + 1, 0);       // global bucket offsets
    // Parallel histogram: each thread counts its own contiguous record range.
    #pragma omp parallel
    {
        int nt  = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t lo = (int64_t)tid * n / nt;
        int64_t hi = (int64_t)(tid + 1) * n / nt;
        int64_t* h = ap_thist.data() + (size_t)tid * T;
        for (int64_t i = lo; i < hi; ++i) h[bucket_of(ap_idx[i])]++;
    }
    // Global bucket sizes -> offsets.
    for (int b = 0; b < T; ++b) {
        int64_t s = 0;
        for (int t = 0; t < T; ++t) s += ap_thist[(size_t)t * T + b];
        boffs[b + 1] = boffs[b] + s;
    }
    // Per-(thread,bucket) write cursors: bucket base + sum of earlier threads.
    // Reuse ap_thist to hold each thread's start cursor for each bucket.
    {
        std::vector<int64_t> run(T, 0);
        for (int b = 0; b < T; ++b) run[b] = boffs[b];
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < T; ++b) {
                int64_t c = ap_thist[(size_t)t * T + b];
                ap_thist[(size_t)t * T + b] = run[b];
                run[b] += c;
            }
    }
    ap_bidx.resize(n);
    ap_bdy.resize(n);
    // Scatter into bucket order using the same contiguous range partition.
    #pragma omp parallel
    {
        int nt  = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t lo = (int64_t)tid * n / nt;
        int64_t hi = (int64_t)(tid + 1) * n / nt;
        int64_t* cur = ap_thist.data() + (size_t)tid * T;
        for (int64_t i = lo; i < hi; ++i) {
            int b = bucket_of(ap_idx[i]);
            int64_t p = cur[b]++;
            ap_bidx[p] = ap_idx[i];
            ap_bdy[p]  = dy[i];
        }
    }
    // Apply: thread t owns bucket t; targets are disjoint across buckets.
    #pragma omp parallel
    {
        int nt  = omp_get_num_threads();
        int tid = omp_get_thread_num();
        for (int b = tid; b < T; b += nt)
            for (int64_t i = boffs[b]; i < boffs[b + 1]; ++i)
                y[ap_bidx[i]] += ap_bdy[i];
    }
}

template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::evaluate_add_off_diag_batched_omp(
        const coeff_t* x, coeff_t* y)
{
    assert(!send_counts.empty() && "allocate_temporaries() must be called first");
    using state_t = ZBasisBase::state_t;
    const int N       = ctx.world_size;
    const int my_rank = ctx.my_rank;
    const int64_t dim = (int64_t)basis.dim();
    const int64_t bs  = (batch_size <= 0) ? dim : std::min((int64_t)batch_size, dim);
    const bool owner    = (strategy == ApplyStrategy::OmpOwner);
    const bool prefetch = (strategy == ApplyStrategy::OmpPrefetch);

    std::vector<int> batch_sc(N), batch_rc(N);
    std::vector<int> alltoallv_sd(N), alltoallv_rd(N);

    for (int64_t bidx = 0; bidx < n_batches; ++bidx) {
        const int64_t s0 = std::min(bidx * bs, dim);
        const int64_t s1 = std::min(s0 + bs, dim);

        // --- LOCAL APPLY (threaded scatter into per-rank contiguous slots) ---
        // Two passes over [s0,s1): count per (thread,rank) so each thread owns a
        // disjoint sub-slot of every rank's send region, then scatter. The two
        // passes use the same static tid partition so sub-slot offsets line up.
        int T = omp_get_max_threads();
        ap_tsend.assign((size_t)T * N, 0);
        #pragma omp parallel
        {
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int64_t lo = s0 + (int64_t)tid * (s1 - s0) / nt;
            int64_t hi = s0 + (int64_t)(tid + 1) * (s1 - s0) / nt;
            int64_t* tc = ap_tsend.data() + (size_t)tid * N;
            for (int64_t il = lo; il < hi; ++il)
                for (const auto& [c, op] : ops.off_diag_terms) {
                    (void)c;
                    state_t state = basis[il];
                    if (op.applyState(state) == 0) continue;
                    tc[ctx.rank_of_state(state)]++;
                }
        }
        // Per-(thread,rank) base cursors within each rank's slot; per-rank totals.
        for (int r = 0; r < N; ++r) {
            int64_t run = send_displs[r];
            for (int t = 0; t < T; ++t) {
                int64_t c = ap_tsend[(size_t)t * N + r];
                ap_tsend[(size_t)t * N + r] = run;
                run += c;
            }
            batch_sc[r] = (int)(run - send_displs[r]);
        }
        #pragma omp parallel
        {
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int64_t lo = s0 + (int64_t)tid * (s1 - s0) / nt;
            int64_t hi = s0 + (int64_t)(tid + 1) * (s1 - s0) / nt;
            int64_t* cur = ap_tsend.data() + (size_t)tid * N;
            for (int64_t il = lo; il < hi; ++il)
                for (const auto& [c, op] : ops.off_diag_terms) {
                    state_t state = basis[il];
                    auto sign = op.applyState(state);
                    if (sign == 0) continue;
                    int r = (int)ctx.rank_of_state(state);
                    int64_t p = cur[r]++;
                    send_state[p] = state;
                    send_dy[p]    = c * x[il] * (coeff_t)sign;
                }
        }

        // --- SELF-UPDATE (threaded search + apply) ---
        apply_records(send_state.data() + send_displs[my_rank],
                      send_dy.data() + send_displs[my_rank],
                      batch_sc[my_rank], y, owner, prefetch);

        // --- ONE COMMUNICATION ROUND (master thread only) ---
        MPI_Alltoall(batch_sc.data(), 1, MPI_INT, batch_rc.data(), 1, MPI_INT,
                     MPI_COMM_WORLD);
        for (int r = 0; r < N; ++r) alltoallv_sd[r] = (int)send_displs[r];
        alltoallv_rd[0] = 0;
        for (int r = 1; r < N; ++r)
            alltoallv_rd[r] = alltoallv_rd[r-1] + (r-1 != my_rank ? batch_rc[r-1] : 0);
        int64_t batch_recv_total = 0;
        for (int r = 0; r < N; ++r)
            if (r != my_rank) batch_recv_total += batch_rc[r];
        batch_sc[my_rank] = 0;
        batch_rc[my_rank] = 0;

        MPI_Alltoallv(
            send_state.data(), batch_sc.data(), alltoallv_sd.data(), get_mpi_type<state_t>(),
            recv_state.data(), batch_rc.data(), alltoallv_rd.data(), get_mpi_type<state_t>(),
            MPI_COMM_WORLD);
        MPI_Alltoallv(
            send_dy.data(), batch_sc.data(), alltoallv_sd.data(), get_mpi_type<coeff_t>(),
            recv_dy.data(), batch_rc.data(), alltoallv_rd.data(), get_mpi_type<coeff_t>(),
            MPI_COMM_WORLD);

        // --- REMOTE UPDATE (threaded search + apply) ---
        apply_records(recv_state.data(), recv_dy.data(), batch_recv_total,
                      y, owner, prefetch);
    }
}

template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::allocate_temporaries(int B) {
    using state_t = ZBasisBase::state_t;
    int N = ctx.world_size;
    int64_t local_dim = (int64_t)basis.dim();
    batch_size = (B <= 0) ? (int)local_dim : std::min(B, (int)local_dim);

    // Every rank must join every collective round, but local dims (and thus
    // batch counts) differ: take the global max as the shared round count.
    int64_t nb_local = (batch_size > 0)
        ? (local_dim + batch_size - 1) / batch_size : 0;
    MPI_Allreduce(&nb_local, &n_batches, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

    send_counts.assign(N, 0);
    recv_counts.resize(N);
    send_displs.resize(N);
    recv_displs.resize(N);

    // Dry-run: find the max per-rank send count for any window of batch_size states.
    // Buffers only need to hold one batch worth of data, so we size to the worst-case
    // window rather than the total across all batches.
    // applyState(state) modifies state in-place to the flipped state.
    std::vector<MPI_Count> window_counts(N, 0);
    for (int64_t il = 0; il < local_dim; ++il) {
        for (const auto& [c, op] : ops.off_diag_terms) {
            state_t state = basis[il];
            auto sign = op.applyState(state);
            if (sign == 0) continue;
            window_counts[ctx.rank_of_state(state)]++;
        }
        // At the end of each window, record max and reset for next window.
        if ((il + 1) % batch_size == 0 || il + 1 == local_dim) {
            for (int r = 0; r < N; r++) {
                send_counts[r] = std::max(send_counts[r], window_counts[r]);
                window_counts[r] = 0;
            }
        }
    }

    // Exchange max-per-batch send counts so each rank knows the most it will
    // ever receive from any other rank in a single communication round.
    std::vector<int> sc_int(N), rc_int(N);
    for (int r = 0; r < N; r++) sc_int[r] = (int)send_counts[r];
    MPI_Alltoall(sc_int.data(), 1, MPI_INT, rc_int.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int r = 0; r < N; r++) recv_counts[r] = rc_int[r];

    // Prefix sums for flat buffer offsets
    send_displs[0] = recv_displs[0] = 0;
    for (int r = 1; r < N; r++) {
        send_displs[r] = send_displs[r-1] + send_counts[r-1];
        recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
    }
    MPI_Count total_send = (N > 0) ? send_displs[N-1] + send_counts[N-1] : 0;
    MPI_Count total_recv = (N > 0) ? recv_displs[N-1] + recv_counts[N-1] : 0;

    // Overflow guard
    for (int r = 0; r < N; r++) {
        if (send_counts[r] > INT_MAX || recv_counts[r] > INT_MAX)
            throw std::runtime_error("allocate_temporaries: MPI count overflow (exceeds INT_MAX)");
    }

    send_state.resize(total_send);
    send_dy.resize(total_send);
    recv_state.resize(total_recv);
    recv_dy.resize(total_recv);

    constexpr size_t record_bytes = sizeof(state_t) + sizeof(coeff_t);
    ctx.log(logging::DEBUG) << "[alloc r" << ctx.my_rank << "]"
            << " total_send=" << total_send
            << " (" << total_send * record_bytes / (1 << 20) << " MiB)"
            << " total_recv=" << total_recv
            << " (" << total_recv * record_bytes / (1 << 20) << " MiB)"
            << " batch_size=" << batch_size << "/" << local_dim << " states\n";
}


// ---------------------------------------------------------------------------
// Static apply plan
//
// The basis, the Hamiltonian and the batch boundaries are invariant across
// Lanczos iterations, so the entire communication/update graph of the batched
// apply can be computed once:
//   - per-(batch, rank) send/recv counts are frozen (no count exchange),
//   - the local index of every incoming record is resolved by a single
//     setup-time state exchange + search and stored as uint32 (records arrive
//     in a deterministic order because the send side scatters in (state, op)
//     order and Alltoallv preserves per-pair ordering),
//   - likewise for self-update records.
// Steady state then ships only dy (8 B/record instead of 24) in one Alltoallv
// and applies y[idx[i]] += dy[i] — independent stores with full memory-level
// parallelism instead of dependent-miss binary-search chains.
// ---------------------------------------------------------------------------

template <RealOrCplx coeff_t, Basis B>
size_t MPILazyOpSum<coeff_t, B>::plan_bytes() const {
    return plan_sc.capacity() * sizeof(int)
         + plan_rc.capacity() * sizeof(int)
         + plan_tgt.capacity() * sizeof(uint32_t)
         + plan_self.capacity() * sizeof(uint32_t)
         + plan_tgt_offs.capacity() * sizeof(int64_t)
         + plan_self_offs.capacity() * sizeof(int64_t);
}

template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::release_state_buffers() {
    { std::vector<ZBasisBase::state_t> tmp; std::swap(tmp, send_state); }
    { std::vector<ZBasisBase::state_t> tmp; std::swap(tmp, recv_state); }
}

template <RealOrCplx coeff_t, Basis basis_t>
void MPILazyOpSum<coeff_t, basis_t>::build_plan(int B, size_t mem_cap_bytes) {
    using state_t = ZBasisBase::state_t;
    const int N = ctx.world_size;
    const int my_rank = ctx.my_rank;

    if (send_counts.empty())
        allocate_temporaries(B);

    const int64_t dim = (int64_t)basis.dim();
    if ((uint64_t)dim > UINT32_MAX)
        throw std::runtime_error("build_plan: local dim exceeds uint32 index range");

    plan_bs = std::max<int64_t>((int64_t)batch_size, 1);
    const int64_t NB = n_batches;

    // --- Pass 1: exact per-(batch, destination) send counts ---------------
    // Rank-major layout so a single Alltoall (NB ints per rank pair) turns
    // send counts into recv counts.
    std::vector<int> sc_rm((size_t)N * NB, 0);
    for (int64_t il = 0; il < dim; ++il) {
        const int64_t b = il / plan_bs;
        for (const auto& [c, op] : ops.off_diag_terms) {
            (void)c;
            state_t state = basis[il];
            if (op.applyState(state) == 0) continue;
            sc_rm[(size_t)ctx.rank_of_state(state) * NB + b]++;
        }
    }
    std::vector<int> rc_rm((size_t)N * NB);
    MPI_Alltoall(sc_rm.data(), (int)NB, MPI_INT,
                 rc_rm.data(), (int)NB, MPI_INT, MPI_COMM_WORLD);

    // Batch-major copies for the steady-state loop; recv self slot zeroed
    // (self records never cross MPI).
    plan_sc.resize((size_t)NB * N);
    plan_rc.resize((size_t)NB * N);
    plan_tgt_offs.assign(NB + 1, 0);
    plan_self_offs.assign(NB + 1, 0);
    for (int64_t b = 0; b < NB; b++) {
        int64_t recv_total = 0;
        for (int r = 0; r < N; r++) {
            plan_sc[b*N + r] = sc_rm[(size_t)r * NB + b];
            int rc = (r == my_rank) ? 0 : rc_rm[(size_t)r * NB + b];
            plan_rc[b*N + r] = rc;
            recv_total += rc;
        }
        plan_tgt_offs[b+1]  = plan_tgt_offs[b] + recv_total;
        plan_self_offs[b+1] = plan_self_offs[b] + plan_sc[b*N + my_rank];
    }
    const int64_t tgt_total  = plan_tgt_offs[NB];
    const int64_t self_total = plan_self_offs[NB];

    // --- Memory-cap safety valve (collective decision) --------------------
    size_t projected = (size_t)(tgt_total + self_total) * sizeof(uint32_t)
                     + (size_t)2 * NB * N * sizeof(int)
                     + (size_t)2 * (NB + 1) * sizeof(int64_t);
    size_t projected_max = projected;
    MPI_Allreduce(&projected, &projected_max, 1, get_mpi_type<size_t>(),
                  MPI_MAX, MPI_COMM_WORLD);
    if (mem_cap_bytes > 0 && projected_max > mem_cap_bytes) {
        logging::log(logging::INFO) << "[plan] refused: worst rank needs "
                      << projected_max / (1<<20) << " MiB > cap "
                      << mem_cap_bytes / (1<<20) << " MiB; "
                      << "falling back to search path\n";
        plan_sc.clear(); plan_sc.shrink_to_fit();
        plan_rc.clear(); plan_rc.shrink_to_fit();
        plan_tgt_offs.clear(); plan_self_offs.clear();
        return;
    }
    plan_tgt.resize(tgt_total);
    plan_self.resize(self_total);

    // --- Pass 2: exchange states once and resolve every record's index ----
    std::vector<int> batch_sc(N), batch_rc(N), sd(N), rd(N);
    for (int r = 0; r < N; r++) sd[r] = (int)send_displs[r];

    for (int64_t b = 0; b < NB; b++) {
        const int64_t s0 = std::min(b * plan_bs, dim);
        const int64_t s1 = std::min(s0 + plan_bs, dim);

        std::vector<MPI_Count> cursors(send_displs.begin(), send_displs.end());
        for (int64_t il = s0; il < s1; ++il) {
            for (const auto& [c, op] : ops.off_diag_terms) {
                (void)c;
                state_t state = basis[il];
                if (op.applyState(state) == 0) continue;
                send_state[cursors[ctx.rank_of_state(state)]++] = state;
            }
        }

        // Self-update indices, in send (= generation) order.
        {
            const int cnt = plan_sc[b*N + my_rank];
            const MPI_Count base = send_displs[my_rank];
            uint32_t* out = plan_self.data() + plan_self_offs[b];
            for (int i = 0; i < cnt; i++) {
                ZBasisBase::idx_t idx;
                if (!basis.search(send_state[base + i], idx))
                    throw std::logic_error("build_plan: self state not in local basis");
                out[i] = (uint32_t)idx;
            }
        }

        for (int r = 0; r < N; r++) {
            batch_sc[r] = (r == my_rank) ? 0 : plan_sc[b*N + r];
            batch_rc[r] = plan_rc[b*N + r];
        }
        rd[0] = 0;
        for (int r = 1; r < N; r++) rd[r] = rd[r-1] + batch_rc[r-1];

        MPI_Alltoallv(
            send_state.data(), batch_sc.data(), sd.data(), get_mpi_type<state_t>(),
            recv_state.data(), batch_rc.data(), rd.data(), get_mpi_type<state_t>(),
            MPI_COMM_WORLD);

        // Remote-update indices, in packed recv-buffer (= arrival) order.
        const int64_t recv_total = plan_tgt_offs[b+1] - plan_tgt_offs[b];
        uint32_t* out = plan_tgt.data() + plan_tgt_offs[b];
        for (int64_t i = 0; i < recv_total; i++) {
            ZBasisBase::idx_t idx;
            if (!basis.search(recv_state[i], idx))
                throw std::logic_error("build_plan: received state not in local basis");
            out[i] = (uint32_t)idx;
        }
    }

    plan_built = true;
    ctx.log(logging::DEBUG) << "[plan r" << my_rank << "]"
            << " batches=" << NB
            << " self_records=" << self_total
            << " recv_records=" << tgt_total
            << " plan_MiB=" << plan_bytes() / (1<<20) << "\n";
}

template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::evaluate_add_off_diag_planned(const coeff_t* x, coeff_t* y) {
    assert(plan_built && "build_plan() must be called first");
    const int N = ctx.world_size;
    const int my_rank = ctx.my_rank;
    const int64_t dim = (int64_t)basis.dim();

    Timer loc_apply_timer("[planned local apply]",   my_rank);
    Timer self_update_timer("[planned self update]", my_rank);
    Timer alltoallv_timer("[planned mpi alltoallv]", my_rank);
    Timer remote_up_timer("[planned remote update]", my_rank);
    std::vector<const Timer*> timers{&loc_apply_timer, &self_update_timer,
        &alltoallv_timer, &remote_up_timer};

    std::vector<int> batch_sc(N), batch_rc(N), sd(N), rd(N);
    for (int r = 0; r < N; r++) sd[r] = (int)send_displs[r];
    std::vector<MPI_Count> cursors(N);

    for (int64_t b = 0; b < n_batches; b++) {
        const int64_t s0 = std::min(b * plan_bs, dim);
        const int64_t s1 = std::min(s0 + plan_bs, dim);
        const int* sc = &plan_sc[b * N];

        // --- LOCAL APPLY: recompute applyState + hash, ship only dy ---
        std::copy(send_displs.begin(), send_displs.end(), cursors.begin());
        BENCH_TIMER_TIMEIT(loc_apply_timer,
        for (int64_t il = s0; il < s1; ++il) {
            for (const auto& [c, op] : ops.off_diag_terms) {
                ZBasisBase::state_t state = basis[il];
                auto sign = op.applyState(state);
                if (sign == 0) continue;
                send_dy[cursors[ctx.rank_of_state(state)]++] = c * x[il] * (coeff_t)sign;
            }
        }
        ) // BENCH_TIMER_TIMEIT
#ifndef NDEBUG
        for (int r = 0; r < N; r++)
            assert(cursors[r] - send_displs[r] == sc[r] &&
                   "plan out of date: send counts changed since build_plan()");
#endif

        // --- SELF-UPDATE via precomputed indices ---
        BENCH_TIMER_TIMEIT(self_update_timer,
        {
            const int cnt = sc[my_rank];
            const coeff_t* dy = send_dy.data() + send_displs[my_rank];
            const uint32_t* idx = plan_self.data() + plan_self_offs[b];
            for (int i = 0; i < cnt; i++)
                y[idx[i]] += dy[i];
        }
        ) // BENCH_TIMER_TIMEIT

        // --- SINGLE dy EXCHANGE with frozen counts ---
        for (int r = 0; r < N; r++) {
            batch_sc[r] = (r == my_rank) ? 0 : sc[r];
            batch_rc[r] = plan_rc[b * N + r];
        }
        rd[0] = 0;
        for (int r = 1; r < N; r++) rd[r] = rd[r-1] + batch_rc[r-1];

        BENCH_TIMER_TIMEIT(alltoallv_timer,
        MPI_Alltoallv(
            send_dy.data(), batch_sc.data(), sd.data(), get_mpi_type<coeff_t>(),
            recv_dy.data(), batch_rc.data(), rd.data(), get_mpi_type<coeff_t>(),
            MPI_COMM_WORLD);
        ) // BENCH_TIMER_TIMEIT

        // --- REMOTE UPDATE: streaming indexed scatter, no search ---
        BENCH_TIMER_TIMEIT(remote_up_timer,
        {
            const int64_t recv_total = plan_tgt_offs[b+1] - plan_tgt_offs[b];
            const uint32_t* idx = plan_tgt.data() + plan_tgt_offs[b];
            for (int64_t i = 0; i < recv_total; i++)
                y[idx[i]] += recv_dy[i];
        }
        ) // BENCH_TIMER_TIMEIT
    }

#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
    for (auto t : timers)
        t->print_summary(ctx.log(logging::DEBUG));
#endif
}


// explicit template instantiations: generate symbols to link with
template struct ZBasisMPI<ZBasisBST>;
template struct ZBasisMPI<ZBasisInterp>;
template struct ZBasisMPI<ZBasisBSTFast>;

template struct MPILazyOpSum<double, ZBasisBST_HashMPI>;
template struct MPILazyOpSum<double, ZBasisInterp_HashMPI>;
template struct MPILazyOpSum<double, ZBasisBSTFast_HashMPI>;

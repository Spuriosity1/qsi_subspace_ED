#include "operator_mpi.hpp"
#include <mpi.h>
#include <cassert>
#include <fstream>
#include "timeit.hpp"
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <cstdlib>

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


    // State for pipelined communication. Coeff and state travel as two parallel
    // native transfers: coeff as coeff_t, state as uint64 (a Uint128 is two
    // uint64, so state counts are doubled and the pointer is cast). Reassembled
    // bit-identically at the destination.
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
            for (auto& v : send_dy)          v.clear();
            for (auto& v : send_states)      v.clear();
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

    // Scratch for batched interleaved search (reused across sources/operators;
    // pipeline is single-threaded so one buffer is safe).
    std::vector<ZBasisBase::idx_t> idxbuf;

    // Scatter prefetch distance: how many records ahead to prefetch the random
    // y[idx] write target. Read once; tune via APPLY_SCATTER_PD (0 disables).
    static const int PD = [] {
        const char* e = std::getenv("APPLY_SCATTER_PD");
        int d = e ? std::atoi(e) : 16;
        return d < 0 ? 0 : d;
    }();

    // Resolve a block of `m` states in one prefetched interleaved search, then
    // scatter y[idx] += dy with the write target prefetched PD records ahead so
    // the random y-write miss overlaps too.
    auto apply_block = [&](const ZBasisBase::state_t* st, const coeff_t* dyv,
                           int64_t m, const char* what) {
        (void)what;
        if (m == 0) return;
        idxbuf.resize(m);
        basis.search_batch(st, m, idxbuf.data());
        for (int64_t j = 0; j < m; ++j) {
            if (j + PD < m && idxbuf[j + PD] >= 0)
                __builtin_prefetch(&y[idxbuf[j + PD]], 1, 0);
            const ZBasisBase::idx_t p = idxbuf[j];
#ifndef NDEBUG
            if (p < 0) {
                std::cerr << "State not found (" << what << ") on rank "
                          << ctx.my_rank << ": ";
                printHex(std::cerr, st[j]) << "\n";
                throw std::logic_error("State not found in batched apply");
            }
#endif
            if (p >= 0) y[p] += dyv[j];
        }
    };

    // Post receives for operator prev_index's data into comm, wait for all of
    // its communication to finish, then apply the received updates to y.
    auto process_receives = [&](OperatorCommState& comm, int prev_index) {
        BENCH_TIMER_TIMEIT(countx_wait_timer,
        if (!comm.count_exchange_done) {
            MPI_Wait(&comm.count_exchange_req, MPI_STATUS_IGNORE);
            comm.count_exchange_done = true;
        }
        )

        DEBUG_PRINT_VEC(">> recv ", prev_index, comm.recvcounts, ctx)

        for (int source = 0; source < ctx.world_size; ++source) {
            if (source == ctx.my_rank) continue;
            int cnt = comm.recvcounts[source];
            comm.recv_states_bufs[source].resize(cnt);
            comm.recv_dy_bufs[source].resize(cnt);
            if (cnt == 0) continue;

            comm.requests.push_back(MPI_Request{});
            MPI_Irecv(reinterpret_cast<uint64_t*>(comm.recv_states_bufs[source].data()),
                     2*cnt, MPI_UINT64_T,
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
            apply_block(comm.recv_states_bufs[source].data(),
                        comm.recv_dy_bufs[source].data(),
                        (int64_t)comm.recv_states_bufs[source].size(), "remote");
        }
        )
    };

    int op_index = 0;
    for ( const auto& [c, op] : ops.off_diag_terms ){
        curr_op_comm.reset_for_new_op();

         // Organise sends by destination rank
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
        apply_block(curr_op_comm.send_states[ctx.my_rank].data(),
                    curr_op_comm.send_dy[ctx.my_rank].data(),
                    (int64_t)curr_op_comm.send_states[ctx.my_rank].size(), "self");
        )

        // === PROCESS PREVIOUS OPERATOR'S RECEIVES ===
        if (has_prev_op) {
            process_receives(prev_op_comm, op_index - 1);
        }

        // === DATA SENDS FOR CURRENT OPERATOR ===
        BENCH_TIMER_TIMEIT(countx_wait_timer_2,
        MPI_Wait(&curr_op_comm.count_exchange_req, MPI_STATUS_IGNORE);
        curr_op_comm.count_exchange_done = true;
        )

        // Begin sending to all nonempty, non-self targets (state as uint64, dy native).
        for (int target_rank=0; target_rank<ctx.world_size; target_rank++){
            if (target_rank == ctx.my_rank ||
                    curr_op_comm.send_states[target_rank].empty()) continue;

            curr_op_comm.requests.push_back(MPI_Request{});
            MPI_Isend(
                    reinterpret_cast<uint64_t*>(curr_op_comm.send_states[target_rank].data()),
                    2*curr_op_comm.send_states[target_rank].size(), MPI_UINT64_T,
                    target_rank, 10*op_index + 1, MPI_COMM_WORLD,
                    &curr_op_comm.requests.back());

            curr_op_comm.requests.push_back(MPI_Request{});
            MPI_Isend(
                    curr_op_comm.send_dy[target_rank].data(),
                    curr_op_comm.send_dy[target_rank].size(), get_mpi_type<coeff_t>(),
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


// Precompute, per off-diagonal operator, how many records this rank sends to
// (and receives from) every rank, and the packed displacements. Static across
// applies: the basis partition and Hamiltonian never change. Also sizes the
// ping-pong record buffers to the largest per-operator total.
template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::build_apply_metadata() const {
    using idx_t = ZBasisBase::idx_t;
    const int N = ctx.world_size;
    const size_t n_ops = ops.off_diag_terms.size();

    op_comm_metadata.resize(n_ops);
    int64_t max_send = 0, max_recv = 0;

    for (size_t oi = 0; oi < n_ops; ++oi) {
        const auto& op = ops.off_diag_terms[oi].second;
        auto& md = op_comm_metadata[oi];
        md.send_sizes.assign(N, 0);
        md.send_displs.assign(N, 0);
        md.recv_sizes.assign(N, 0);
        md.recv_displs.assign(N, 0);

        // applyState mutates the state into its target; the destination rank is
        // the rank owning that target state.
        for (idx_t il = 0; il < basis.dim(); ++il) {
            ZBasisBase::state_t state = basis[il];
            if (op.applyState(state) == 0) continue;
            md.send_sizes[ctx.rank_of_state(state)]++;
        }

        MPI_Alltoall(md.send_sizes.data(), 1, get_mpi_type<int>(),
                     md.recv_sizes.data(), 1, get_mpi_type<int>(), MPI_COMM_WORLD);

        for (int r = 1; r < N; ++r) {
            md.send_displs[r] = md.send_displs[r-1] + md.send_sizes[r-1];
            md.recv_displs[r] = md.recv_displs[r-1] + md.recv_sizes[r-1];
        }
        md.send_total = md.send_displs[N-1] + md.send_sizes[N-1];
        md.recv_total = md.recv_displs[N-1] + md.recv_sizes[N-1];
        max_send = std::max(max_send, md.send_total);
        max_recv = std::max(max_recv, md.recv_total);

        // Doubled (uint64) counts/displs for the native state transfer: a
        // Uint128 state is shipped as two uint64, so counts and offsets double.
        md.send_sizes_u64.resize(N);  md.send_displs_u64.resize(N);
        md.recv_sizes_u64.resize(N);  md.recv_displs_u64.resize(N);
        for (int r = 0; r < N; ++r) {
            md.send_sizes_u64[r]  = 2 * md.send_sizes[r];
            md.send_displs_u64[r] = 2 * md.send_displs[r];
            md.recv_sizes_u64[r]  = 2 * md.recv_sizes[r];
            md.recv_displs_u64[r] = 2 * md.recv_displs[r];
        }
    }

    for (int s = 0; s < 2; ++s) {
        send_dy_ring[s].resize(max_send);
        send_state_ring[s].resize(max_send);
        recv_dy_ring[s].resize(max_recv);
        recv_state_ring[s].resize(max_recv);
    }
}


// ---- Shared prealloc building blocks (threaded) --------------------------

// Counting-sort operator oi's (coeff, target-state) records into the parallel
// send_dy_ring[slot]/send_state_ring[slot], grouped by destination rank per the precomputed
// displacements. Threaded: the send buffer must stay grouped by rank and the
// per-rank write cursor is shared, so a plain parallel-for won't do. Each
// thread owns a contiguous chunk of states; pass 1 counts how many records it
// contributes to each rank, an exclusive prefix over threads gives each thread
// a disjoint write slot within every rank's block, and pass 2 scatters into
// those slots. applyState is recomputed in pass 2 rather than stashing an
// O(dim) per-state scratch (prohibitive at the target scale).
template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::fill_sends(int slot, size_t oi,
                                          const coeff_t* x, Timer& timer) const {
    using idx_t = ZBasisBase::idx_t;
    BENCH_TIMER_TIMEIT(timer,
    // Plain locals, not a structured binding: Clang cannot capture a
    // structured binding into an OpenMP region.
    const coeff_t c  = ops.off_diag_terms[oi].first;
    const auto&   op = ops.off_diag_terms[oi].second;
    const auto& md = op_comm_metadata[oi];
    auto& dbuf = send_dy_ring[slot];
    auto& tbuf = send_state_ring[slot];
    const int N = ctx.world_size;
    const idx_t dim = basis.dim();
    const int T = omp_get_max_threads();

    std::vector<int> hist((size_t)T * N, 0);   // [t*N + r], tiny (T*N)
    _Pragma("omp parallel num_threads(T)")
    {
        const int t = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        const idx_t chunk = (dim + nth - 1) / nth;
        const idx_t s0 = std::min<idx_t>((idx_t)t * chunk, dim);
        const idx_t s1 = std::min<idx_t>(s0 + chunk, dim);

        int* h = &hist[(size_t)t * N];
        for (idx_t il = s0; il < s1; ++il) {
            ZBasisBase::state_t s = basis[il];
            if (op.applyState(s) == 0) continue;
            h[ctx.rank_of_state(s)]++;
        }
        _Pragma("omp barrier")

        // This thread's write head per rank: rank's base displacement plus
        // everything earlier threads place into that rank.
        std::vector<int> cur(N);
        for (int r = 0; r < N; ++r) {
            int base = md.send_displs[r];
            for (int tt = 0; tt < t; ++tt) base += hist[(size_t)tt * N + r];
            cur[r] = base;
        }
        for (idx_t il = s0; il < s1; ++il) {
            ZBasisBase::state_t st = basis[il];
            auto sign = op.applyState(st);  // mutates state -> target
            if (sign == 0) continue;
            const int pos = cur[ctx.rank_of_state(st)]++;
            dbuf[pos] = c * x[il] * sign;
            tbuf[pos] = st;
        }
    }
    )
}

// Search each received target into the local basis and accumulate. A single
// off-diagonal operator is injective on its support (a fixed bit-flip), so the
// records for one operator are images O(s) of distinct basis states and resolve
// to DISTINCT local indices: no two iterations write the same y[idx] => the
// accumulation is race-free with no atomics. basis.search is read-only /
// re-entrant. (_Pragma, not #pragma: we are inside a macro.)
template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::apply_recvs(int slot, size_t oi,
                                           coeff_t* y, Timer& timer) const {
    using idx_t = ZBasisBase::idx_t;
    BENCH_TIMER_TIMEIT(timer,
    const auto& rstate = recv_state_ring[slot];
    const auto& rdy    = recv_dy_ring[slot];
    const int64_t n = op_comm_metadata[oi].recv_total;
    _Pragma("omp parallel for schedule(static)")
    for (int64_t j = 0; j < n; ++j) {
        idx_t idx;
        ASSERT_STATE_FOUND("remote", rstate[j],
                basis.search(rstate[j], idx));
        y[idx] += rdy[j];
    }
    )
}


// ---- Variant A: collective transport (Alltoallv per operator) ------------
template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::
evaluate_add_off_diag_pipeline_prealloc(const coeff_t* x, coeff_t* y) const
{
    Timer prealloc_timer("[preallocate]", ctx.my_rank);
    Timer fill_sends_timer("[fill_sends]", ctx.my_rank);
    Timer post_timer("[post]", ctx.my_rank);
    Timer wait_timer("[wait]", ctx.my_rank);
    Timer apply_recvs_timer("[apply_recvs]", ctx.my_rank);
    std::vector<const Timer*> timers {&prealloc_timer, &fill_sends_timer, &post_timer, &wait_timer, &apply_recvs_timer};

    BENCH_TIMER_TIMEIT(prealloc_timer,
    if (op_comm_metadata.empty()) build_apply_metadata();
    )

    const size_t n_ops = ops.off_diag_terms.size();
    if (n_ops == 0) return;

    // Two native collectives per operator: coeff as coeff_t, state as uint64
    // (a Uint128 is two uint64; counts/displs doubled). req[slot][0]=dy,
    // req[slot][1]=state.
    auto post = [&](int slot, size_t oi, MPI_Request req[2]) {
        BENCH_TIMER_TIMEIT(post_timer,
        const auto& md = op_comm_metadata[oi];
        MPI_Ialltoallv(
            send_dy_ring[slot].data(), md.send_sizes.data(), md.send_displs.data(), get_mpi_type<coeff_t>(),
            recv_dy_ring[slot].data(), md.recv_sizes.data(), md.recv_displs.data(), get_mpi_type<coeff_t>(),
            MPI_COMM_WORLD, &req[0]);
        MPI_Ialltoallv(
            reinterpret_cast<uint64_t*>(send_state_ring[slot].data()),
            md.send_sizes_u64.data(), md.send_displs_u64.data(), MPI_UINT64_T,
            reinterpret_cast<uint64_t*>(recv_state_ring[slot].data()),
            md.recv_sizes_u64.data(), md.recv_displs_u64.data(), MPI_UINT64_T,
            MPI_COMM_WORLD, &req[1]);
        )
    };

    // Software-pipelined over operators: operator i's exchange overlaps
    // operator (i-1)'s search and operator (i+1)'s fill. Slot i-2 is guaranteed
    // complete (waited at iteration i-1) before it is reused at iteration i.
    MPI_Request req[2][2];
    fill_sends(0, 0, x, fill_sends_timer);
    post(0, 0, req[0]);
    for (size_t i = 1; i < n_ops; ++i) {
        const int cur = i & 1, prev = (i - 1) & 1;
        fill_sends(cur, i, x, fill_sends_timer);
        post(cur, i, req[cur]);
        BENCH_TIMER_TIMEIT(wait_timer,
        MPI_Waitall(2, req[prev], MPI_STATUSES_IGNORE);
        )
        apply_recvs(prev, i - 1, y, apply_recvs_timer);
    }
    const int last = (n_ops - 1) & 1;
    BENCH_TIMER_TIMEIT(wait_timer,
    MPI_Waitall(2, req[last], MPI_STATUSES_IGNORE);
    )
    apply_recvs(last, n_ops - 1, y, apply_recvs_timer);

#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
        for (auto t : timers){
            t->print_summary(ctx.log(logging::DEBUG));
        }
#endif
}


// ---- Variant B: point-to-point transport (Isend/Irecv per peer/op) -------
// Identical plan, fill and search to variant A; only the transport differs:
// per non-self peer, one Irecv+Isend for the coeff (coeff_t) and one for the
// state (uint64), with the self block copied locally. The motivation is that
// point-to-point over shared memory / eager protocols makes progress without
// re-entering MPI, so the transfer can overlap the fill and search instead of
// stalling at a collective wait. tag = 2*slot (+1 for state) disambiguates the
// (at most two) operators in flight under the ping-pong and the two messages.
template <RealOrCplx coeff_t, Basis B>
void MPILazyOpSum<coeff_t, B>::
evaluate_add_off_diag_prealloc_p2p(const coeff_t* x, coeff_t* y) const
{
    const int N  = ctx.world_size;
    const int me = ctx.my_rank;

    Timer prealloc_timer("[preallocate]", me);
    Timer fill_sends_timer("[fill_sends]", me);
    Timer post_timer("[post]", me);
    Timer wait_timer("[wait]", me);
    Timer apply_recvs_timer("[apply_recvs]", me);
    std::vector<const Timer*> timers {&prealloc_timer, &fill_sends_timer, &post_timer, &wait_timer, &apply_recvs_timer};

    BENCH_TIMER_TIMEIT(prealloc_timer,
    if (op_comm_metadata.empty()) build_apply_metadata();
    )

    const size_t n_ops = ops.off_diag_terms.size();
    if (n_ops == 0) return;

    std::array<std::vector<MPI_Request>, 2> reqs;
    // Parallel to reqs[slot]: the source rank of each recv request, or -1 for a
    // send request. Used to route Waitany completions to per-peer processing.
    std::array<std::vector<int>, 2> req_peer;

    auto post_p2p = [&](int slot, size_t oi) {
        BENCH_TIMER_TIMEIT(post_timer,
        const auto& md = op_comm_metadata[oi];
        auto* sdy = send_dy_ring[slot].data();
        auto* rdy = recv_dy_ring[slot].data();
        auto* sst = reinterpret_cast<uint64_t*>(send_state_ring[slot].data());
        auto* rst = reinterpret_cast<uint64_t*>(recv_state_ring[slot].data());
        const int tag_dy = 2 * slot, tag_st = 2 * slot + 1;
        auto& rq = reqs[slot];
        auto& rp = req_peer[slot];
        rq.clear();
        rp.clear();
        for (int r = 0; r < N; ++r) {
            if (r == me || md.recv_sizes[r] == 0) continue;
            rq.emplace_back();
            MPI_Irecv(rdy + md.recv_displs[r], md.recv_sizes[r], get_mpi_type<coeff_t>(),
                      r, tag_dy, MPI_COMM_WORLD, &rq.back());
            rp.push_back(r);
            rq.emplace_back();
            MPI_Irecv(rst + md.recv_displs_u64[r], md.recv_sizes_u64[r], MPI_UINT64_T,
                      r, tag_st, MPI_COMM_WORLD, &rq.back());
            rp.push_back(r);
        }
        for (int r = 0; r < N; ++r) {
            if (r == me || md.send_sizes[r] == 0) continue;
            rq.emplace_back();
            MPI_Isend(sdy + md.send_displs[r], md.send_sizes[r], get_mpi_type<coeff_t>(),
                      r, tag_dy, MPI_COMM_WORLD, &rq.back());
            rp.push_back(-1);
            rq.emplace_back();
            MPI_Isend(sst + md.send_displs_u64[r], md.send_sizes_u64[r], MPI_UINT64_T,
                      r, tag_st, MPI_COMM_WORLD, &rq.back());
            rp.push_back(-1);
        }
        // Self block: no message, copy send self-slot -> recv self-slot
        // (send_sizes[me] == recv_sizes[me] by construction).
        for (int k = 0; k < md.send_sizes[me]; ++k) {
            recv_dy_ring[slot][md.recv_displs[me] + k]    = send_dy_ring[slot][md.send_displs[me] + k];
            recv_state_ring[slot][md.recv_displs[me] + k] = send_state_ring[slot][md.send_displs[me] + k];
        }
        )
    };

    // Search+accumulate one peer's contiguous recv slice. Within an operator all
    // target indices are distinct (injectivity), and each peer lands in a
    // disjoint buffer region, so the slice is threadable and never collides with
    // another slice.
    auto process_slice = [&](int slot, size_t oi, int peer) {
        BENCH_TIMER_TIMEIT(apply_recvs_timer,
        const auto& md = op_comm_metadata[oi];
        const int base = md.recv_displs[peer];
        const int cnt  = md.recv_sizes[peer];
        const auto& rstate = recv_state_ring[slot];
        const auto& rdy    = recv_dy_ring[slot];
        _Pragma("omp parallel for schedule(static)")
        for (int k = 0; k < cnt; ++k) {
            ZBasisBase::idx_t idx;
            ASSERT_STATE_FOUND("remote", rstate[base + k],
                    basis.search(rstate[base + k], idx));
            y[idx] += rdy[base + k];
        }
        )
    };

    // Drain a completed operator's receives with Waitany: process each peer as
    // soon as BOTH its (dy, state) recvs land, overlapping the search with the
    // still-incoming transfers. Sends drain through the same loop (peer -1,
    // skipped) so all requests are complete before the buffers are reused.
    auto drain = [&](int slot, size_t oi) {
        process_slice(slot, oi, me);           // self block: ready immediately
        auto& rq = reqs[slot];
        auto& rp = req_peer[slot];
        std::vector<int> ready(N, 0);          // recvs completed per peer (0..2)
        while (true) {
            int k = MPI_UNDEFINED;
            BENCH_TIMER_TIMEIT(wait_timer,
            MPI_Waitany((int)rq.size(), rq.data(), &k, MPI_STATUS_IGNORE);
            )
            if (k == MPI_UNDEFINED) break;
            const int p = rp[k];
            if (p < 0) continue;               // a send completed
            if (++ready[p] == 2) process_slice(slot, oi, p);
        }
    };

    fill_sends(0, 0, x, fill_sends_timer);
    post_p2p(0, 0);
    for (size_t i = 1; i < n_ops; ++i) {
        const int cur = i & 1, prev = (i - 1) & 1;
        fill_sends(cur, i, x, fill_sends_timer);
        post_p2p(cur, i);
        drain(prev, i - 1);
    }
    drain((n_ops - 1) & 1, n_ops - 1);

#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
        for (auto t : timers){
            t->print_summary(ctx.log(logging::DEBUG));
        }
#endif
}




// explicit template instantiations: generate symbols to link with
template struct ZBasisMPI<ZBasisBST>;
template struct ZBasisMPI<ZBasisInterp>;
template struct ZBasisMPI<ZBasisBSTFast>;

template struct MPILazyOpSum<double, ZBasisBST_HashMPI>;
template struct MPILazyOpSum<double, ZBasisInterp_HashMPI>;
template struct MPILazyOpSum<double, ZBasisBSTFast_HashMPI>;

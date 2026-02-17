#pragma once
#include "operator.hpp"
#include <cassert>
#include <mpi.h>
//#include <bit>
#include "mpi_context.hpp"

typedef SparseMPIContext<ZBasisBST::idx_t> MPIctx;


struct BasisTransferWisdom {
    std::vector<int> send_counts, send_ranks, recv_counts, recv_ranks;
    std::vector<ZBasisBase::idx_t> idx_partition;
//    std::vector<ZBasisBase::state_t> state_partition;
};

// TODO this is a mess, MPIctx should clearly be a member of the MPI basis types

struct ZBasisBST_MPI : public ZBasisBST 
{
     MPIctx load_from_file(const fs::path& bfile, const std::string& dataset="basis");

     template<typename coeff_t>
     void remove_null_states(const SymbolicOpSum<coeff_t>& osm, MPIctx& ctx){
         remove_annihilated_states(osm, states);
         // state terminals have not changed (still work for binary search purposes)
         // BUT idx terminals have
         std::vector<size_t> all_state_counts(ctx.world_size);
         size_t my_size = states.size();

         MPI_Allgather(&my_size, 1, get_mpi_type<size_t>(),
                 all_state_counts.data(), 1, get_mpi_type<size_t>(), MPI_COMM_WORLD);
         // rebuild the index partition
         std::fill(ctx.idx_partition.begin(), ctx.idx_partition.end(), 0);
         for (int r=0; r<ctx.world_size; r++){
             ctx.idx_partition[r+1] = all_state_counts[r] + ctx.idx_partition[r];
         }
     }




     void exchange_local_states(
             const BasisTransferWisdom& btw,
             MPIctx& ctx
             ){
         auto& send_counts = btw.send_counts;
         auto& recv_counts = btw.recv_counts;
         auto& send_ranks = btw.send_ranks;
         auto& recv_ranks = btw.recv_ranks;
         // sanity checks
         {
             assert(send_counts.size() == send_ranks.size());
             assert(recv_counts.size() == recv_ranks.size());
             for (size_t i=1; i<send_ranks.size(); i++){assert(send_ranks[i] == send_ranks[i-1]+1);}
             for (size_t i=1; i<recv_ranks.size(); i++){assert(recv_ranks[i] == recv_ranks[i-1]+1);}
             auto total_sends = std::accumulate(send_counts.begin(), send_counts.end(),0ull);
             if(states.size() != total_sends){
                 ctx.log<<"[Error] state.size() = "<<states.size()<<", total_sends="<<total_sends<<std::endl;
                 throw std::logic_error("states size is bad");
             }
         }


         std::vector<MPI_Count> send_displs(send_counts.size(), 0);
         std::vector<MPI_Count> recv_displs(recv_counts.size(), 0);

         send_displs[0]=0;
         recv_displs[0]=0;
         for (size_t j=1; j<send_counts.size(); j++){
             send_displs[j] = send_displs[j-1] + send_counts[j-1];
         }
         for (size_t j=1; j<recv_counts.size(); j++){
             recv_displs[j] = recv_displs[j-1] + recv_counts[j-1];
         }

         printvec(ctx.log << "send_displs ", send_displs)<<std::endl;
         printvec(ctx.log << "send_counts ", send_counts)<<std::endl;
         printvec(ctx.log << "recv_displs ", recv_displs)<<std::endl;
         printvec(ctx.log << "recv_counts ", recv_counts)<<std::endl;

         // allocate a temporary buffer for the sent data
         std::vector<state_t> states_tmp(states);
         states.resize(std::accumulate(recv_counts.begin(), recv_counts.end(), 0));

         const int BASIS_STATEX=0x50;
         std::vector<MPI_Request> reqs;

         for (size_t i=0; i<recv_counts.size(); i++){
             MPI_Request req;
             MPI_Irecv(states.data() + recv_displs[i], recv_counts[i], get_mpi_type<state_t>(),
                     recv_ranks[i], BASIS_STATEX, MPI_COMM_WORLD, &req);
             reqs.push_back(std::move(req));
         }
         for (size_t i=0; i<send_counts.size(); i++){
             MPI_Request req;
             MPI_Isend(states_tmp.data()+send_displs[i], send_counts[i], get_mpi_type<state_t>(), 
                     send_ranks[i],BASIS_STATEX, MPI_COMM_WORLD, &req); 
             reqs.push_back(std::move(req));
         }
         MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

         // states now initialised with (hopefully) correct set of vectors
         // update the terminals of ctx
         ctx.idx_partition = btw.idx_partition;
         ctx.state_partition.resize(ctx.idx_partition.size());

         MPI_Allgather(states.data(), 1, get_mpi_type<state_t>(),
                 ctx.state_partition.data(), 1, get_mpi_type<state_t>(), MPI_COMM_WORLD);

#ifndef NDEBUG
        // states strictly inceasing?
        for (int64_t il=1; il<static_cast<int64_t>(states.size()); il++){
            if(states[il] <= states[il-1]){

                ctx.log<<"[rank "<<ctx.my_rank<<"]\n";
                for (int64_t il2 = std::max(il - 4, static_cast<int64_t>(0)); il2<=il; il2++) 
                    ctx.log<<"states["<<il2<<"]="<< states[il2]<<"\n";

                ctx.log<<std::endl;
                throw std::logic_error("broken state exchange: order not preserved");
            }
        }
#endif
    
     }
};


template<RealOrCplx coeff_t, Basis B>
struct MPILazyOpSumBase {
    using Scalar = coeff_t;
    explicit MPILazyOpSumBase(
            const B& local_basis_, const SymbolicOpSum<coeff_t>& ops_,
            MPIctx& context_
            ) : basis(local_basis_), ops(ops_), ctx(context_)
     {
    }

    // MPILazyOpSumBase operator=(const MPILazyOpSumBase& other) = delete;

	// Core evaluator 
    // Applies y = A x (sets y=0 first)
	void evaluate(const coeff_t* x, coeff_t* y)
    {
		std::fill(y, y + basis.dim(), coeff_t(0));
        this->evaluate_add(x, y);
	}


    // Returns a plan to modify the basis such that load is evenly balanced
    BasisTransferWisdom find_optimal_basis_load();

    // allocates send/receive buffers for MPI alltoall
    // based on current matrix structure
    // By defualt allocate nothing
    virtual void allocate_temporaries() =0;

    // Does y += A*x, where y[i] and x[i] are both indexed from the start of the local block
	virtual void evaluate_add(const coeff_t* x, coeff_t* y) =0; 

protected:
    void evaluate_add_diagonal(const coeff_t* x, coeff_t* y) const;

	const B& basis;
	const SymbolicOpSum<coeff_t> ops;
    MPIctx& ctx;

private:
    static constexpr double APPLY_TOL=1e-15;

    void inplace_bucket_sort(std::vector<ZBasisBase::state_t>& states,
        std::vector<coeff_t>& c,
        std::vector<int>& bucket_sizes,
        std::vector<int>& bucket_starts
        ) const;


};


template<RealOrCplx coeff_t, Basis B>
struct MPILazyOpSumBatched : public MPILazyOpSumBase<coeff_t, B> {

    explicit MPILazyOpSumBatched(
            const B& local_basis_, const SymbolicOpSum<coeff_t>& ops_,
            MPIctx& context_
      ) : MPILazyOpSumBase<coeff_t, B>(local_basis_, ops_, context_)
     {
    }

    void evaluate_add(const coeff_t* x, coeff_t* y) override {
        this->evaluate_add_diagonal(x, y);
        evaluate_add_off_diag_batched(x, y);
    }

    void allocate_temporaries() override;

protected:
    void evaluate_add_off_diag_batched(const coeff_t* x, coeff_t* y);
    // metadata
    std::vector<coeff_t> send_dy; // contiguous buffer
    std::vector<ZBasisBST::state_t> send_state; 
    std::vector<MPI_Count> send_displs;
    std::vector<MPI_Count> send_counts;

    std::vector<coeff_t> recv_dy;
    std::vector<ZBasisBST::state_t> recv_state;
    std::vector<MPI_Count> recv_displs;
    std::vector<MPI_Count> recv_counts;
};


template<RealOrCplx coeff_t, Basis B>
struct MPILazyOpSumPipe : public MPILazyOpSumBase<coeff_t, B> {

    explicit MPILazyOpSumPipe(
            const B& local_basis_, const SymbolicOpSum<coeff_t>& ops_,
            MPIctx& context_
      ) : MPILazyOpSumBase<coeff_t, B>(local_basis_, ops_, context_)
     {
    }


    void allocate_temporaries() override {};

    void evaluate_add(const coeff_t* x, coeff_t* y) override {
        this->evaluate_add_diagonal(x, y);
        evaluate_add_off_diag_pipeline(x, y);
    }

protected:
    void evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y) const;

};



    // Double-buffered communication state
//template<typename coeff_t>
//struct OperatorCommState {
//    std::vector<MPI_Request> requests;
//    std::vector<std::vector<coeff_t>> send_dy;
//    std::vector<std::vector<ZBasisBase::state_t>> send_states;
//    std::vector<std::vector<ZBasisBase::state_t>> recv_states_bufs;
//    std::vector<std::vector<coeff_t>> recv_dy_bufs;
//    std::vector<int> recv_sources;
//
//    void reserve(const std::vector<int>& sendcounts, 
//            const std::vector<int>& recvcounts){
//        for (int r=0; r<sendcounts.size(); r++){
//            send_states[r].reserve(sendcounts[r]);
//            send_dy[r].reserve(sendcounts[r]);
//
//            recv_dy_bufs[r].reserve(recvcounts[r]);
//            recv_states_bufs[r].reserve(recvcounts[r]);
//        }
//    }
//    
//    void clear_for_reuse() {
//        for (auto& v : send_states) v.clear();
//        for (auto& v : send_dy) v.clear();
//        recv_states_bufs.clear();
//        recv_dy_bufs.clear();
//        recv_sources.clear();
//        requests.clear();
//    }
//};


template<typename coeff_t>
class OperatorCommState {
    using state_t = ZBasisBase::state_t;

    // big, raw-memory buffers (manuyally allocated)
    std::vector<coeff_t> send_dy_bufs;
    std::vector<state_t> send_states_bufs;

    std::vector<state_t> recv_states_bufs;
    std::vector<coeff_t> recv_dy_bufs;

    std::vector<size_t> send_pos; // write-head position (relative to send_displs)
    std::vector<size_t> send_counts; // allocated size of each ranks's buffer
    std::vector<size_t> send_displs; // displacement of each rank's buffer

    std::vector<size_t> recv_counts;
    std::vector<size_t> recv_displs;

public:

    std::vector<MPI_Request> requests;

    OperatorCommState() {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        send_counts.resize(world_size, 0);
        send_pos.resize(world_size, 0);
        recv_counts.resize(world_size, 0);
        send_displs.resize(world_size, 0);
        recv_displs.resize(world_size, 0);

        requests.reserve(2 * world_size);
    }

    void reserve(size_t worst_case_send, size_t worst_case_recv){
        // reserves for worst case
        send_dy_bufs.reserve(worst_case_send);
        send_states_bufs.reserve(worst_case_send);

        recv_dy_bufs.reserve(worst_case_recv);
        recv_states_bufs.reserve(worst_case_recv);
    }



    void reserve_send_resize_recv(const std::vector<size_t>& sendcounts_, 
            const std::vector<size_t>& recvcounts_){
        send_counts = sendcounts_;
        recv_counts = recvcounts_;

        assert(send_counts.size() == recv_counts.size());
        size_t curr_send_total=0;
        for (size_t r=0; r<send_counts.size(); r++){
            send_pos[r] = 0;
            send_displs[r] = curr_send_total;
            curr_send_total += send_counts[r];
        }

        size_t curr_recv_total=0;
        for (size_t r=0; r<recv_counts.size(); r++){
            recv_displs[r] = curr_recv_total;
            curr_recv_total += recv_counts[r];
        }

        // Use resize but check capacity to avoid reallocation
        assert(curr_send_total <= send_dy_bufs.capacity() && 
               "Exceeded reserved send capacity!");
        assert(curr_recv_total <= recv_dy_bufs.capacity() && 
               "Exceeded reserved recv capacity!");


        send_dy_bufs.resize(curr_send_total);
        send_states_bufs.resize(curr_send_total);

        recv_dy_bufs.resize(curr_recv_total);
        recv_states_bufs.resize(curr_recv_total);

    }


    auto get_send_count(int rank) const { return send_counts[rank]; }
    auto get_recv_count(int rank) const { return recv_counts[rank]; }

    std::pair<coeff_t*, state_t*> get_send_buffers(int rank){
        return {send_dy_bufs.data()+send_displs[rank],
                send_states_bufs.data()+send_displs[rank]};
    }

    std::pair<coeff_t*, state_t*> get_recv_buffers(int rank){
        return {recv_dy_bufs.data()+recv_displs[rank],
                recv_states_bufs.data()+recv_displs[rank]};
    }

    void sendbuf_push_back(int rank, coeff_t c, const state_t& psi){

        assert(send_pos[rank] < send_counts[rank]);
        auto j = send_displs[rank] + send_pos[rank];
        send_pos[rank]++;

        send_states_bufs[j]=psi;
        send_dy_bufs[j]=c;
    }
    
    void clear_for_reuse() {
        send_states_bufs.resize(0);
        send_dy_bufs.resize(0);
        recv_states_bufs.resize(0);
        recv_dy_bufs.resize(0);
        std::fill(send_pos.begin(), send_pos.end(), 0);
        std::fill(send_counts.begin(), send_counts.end(), 0);
        std::fill(recv_counts.begin(), recv_counts.end(), 0);
        requests.resize(0);
    }
};


template<RealOrCplx coeff_t, Basis B>
struct MPILazyOpSumPipePrealloc : public MPILazyOpSumBase<coeff_t, B> {

    explicit MPILazyOpSumPipePrealloc(
            const B& local_basis_, const SymbolicOpSum<coeff_t>& ops_,
            MPIctx& context_
      ) : MPILazyOpSumBase<coeff_t, B>(local_basis_, ops_, context_)
     {
    }


    void allocate_temporaries() override ;

    void evaluate_add(const coeff_t* x, coeff_t* y) override {
        this->evaluate_add_diagonal(x, y);
        evaluate_add_off_diag_pipeline(x, y);
    }

protected:

    // Communication pattern cache
    struct CommMetadataCache {
        std::vector<std::vector<size_t>> sendcounts_per_op;  // [op_idx][rank]
        std::vector<std::vector<size_t>> recvcounts_per_op;  // [op_idx][rank]
        bool is_initialized = false;
    } comm_cache;
    
    OperatorCommState<coeff_t> comm_buffers[2];

    void evaluate_add_off_diag_pipeline(const coeff_t* x, coeff_t* y);

};


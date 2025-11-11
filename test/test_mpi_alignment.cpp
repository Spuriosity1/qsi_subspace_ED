#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <pyro_tree_mpi.hpp>

void test_vtree_node_layout() {
    vtree_node_t dummy;

    MPI_Aint base, disp[3];
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.state_thus_far, &disp[0]);
    MPI_Get_address(&dummy.curr_spin, &disp[1]);
    MPI_Get_address(&dummy.num_spinon_pairs, &disp[2]);
    for (int i = 0; i < 3; i++) disp[i] -= base;

    printf("== vtree_node_t layout ==\n");
    printf("offsetof(state_thus_far) = %zu, MPI disp = %ld\n",
           offsetof(vtree_node_t, state_thus_far), (long)disp[0]);
    printf("offsetof(curr_spin)       = %zu, MPI disp = %ld\n",
           offsetof(vtree_node_t, curr_spin), (long)disp[1]);
    printf("offsetof(num_spinon_pairs)= %zu, MPI disp = %ld\n",
           offsetof(vtree_node_t, num_spinon_pairs), (long)disp[2]);
    printf("sizeof(vtree_node_t) = %zu\n", sizeof(vtree_node_t));

    assert(offsetof(vtree_node_t, state_thus_far) == (size_t)disp[0]);
    assert(offsetof(vtree_node_t, curr_spin)       == (size_t)disp[1]);
    assert(offsetof(vtree_node_t, num_spinon_pairs)== (size_t)disp[2]);
}

void test_vtree_node_serialization(MPI_Datatype node_type) {
    vtree_node_t node_send = {0};
    vtree_node_t node_recv = {0};

    // Fill with deterministic pattern
    memset(&node_send, 0xAB, sizeof(node_send));

    int bufsize;
    MPI_Pack_size(1, node_type, MPI_COMM_WORLD, &bufsize);
    unsigned char *buf = static_cast<unsigned char*>(malloc(bufsize));

    int pos = 0;
    MPI_Pack(&node_send, 1, node_type, buf, bufsize, &pos, MPI_COMM_WORLD);
    assert(pos <= bufsize);

    // Unpack to another struct
    pos = 0;
    MPI_Unpack(buf, bufsize, &pos, &node_recv, 1, node_type, MPI_COMM_WORLD);

    // Compare byte-for-byte
    if (memcmp(&node_send, &node_recv, sizeof(vtree_node_t)) != 0) {
        printf("❌ MPI pack/unpack mismatch!\n");
        for (size_t i = 0; i < sizeof(vtree_node_t); i++) {
            if (((unsigned char*)&node_send)[i] != ((unsigned char*)&node_recv)[i]) {
                printf("  Byte %zu: send=%02X recv=%02X\n", i,
                       ((unsigned char*)&node_send)[i],
                       ((unsigned char*)&node_recv)[i]);
            }
        }
    } else {
        printf("✅ MPI pack/unpack identical\n");
    }

    free(buf);
}


void test_vtree_node_sendrecv(MPI_Datatype node_type) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vtree_node_t node;
    memset(&node, rank ? 0xBB : 0xAA, sizeof(node));

    if (rank == 0)
        MPI_Send(&node, 1, node_type, 1, 0, MPI_COMM_WORLD);
    else if (rank == 1) {
        vtree_node_t recv;
        MPI_Recv(&recv, 1, node_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("✅ MPI send/recv successful, first bytes: %02X %02X %02X %02X\n",
               ((unsigned char*)&recv)[0], ((unsigned char*)&recv)[1],
               ((unsigned char*)&recv)[2], ((unsigned char*)&recv)[3]);
    }
}

// packet test
void test_packet_layout() {
    packet dummy;

    MPI_Aint base, disp[2];
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.available, &disp[0]);
    MPI_Get_address(&dummy.state, &disp[1]);
    for (int i = 0; i < 2; i++) disp[i] -= base;

    printf("== packet layout ==\n");
    printf("offsetof(available) = %zu, MPI disp = %ld\n",
           offsetof(packet, available), (long)disp[0]);
    printf("offsetof(state)     = %zu, MPI disp = %ld\n",
           offsetof(packet, state), (long)disp[1]);
    printf("sizeof(packet)      = %zu\n", sizeof(packet));

    assert(offsetof(packet, available) == (size_t)disp[0]);
    assert(offsetof(packet, state)     == (size_t)disp[1]);
}


void test_packet_serialization(MPI_Datatype packet_type) {
    packet send_pkt, recv_pkt;
    memset(&send_pkt, 0xCD, sizeof(send_pkt));
    memset(&recv_pkt, 0x00, sizeof(recv_pkt));

    send_pkt.available = 42;  // Just to distinguish data

    int bufsize;
    MPI_Pack_size(1, packet_type, MPI_COMM_WORLD, &bufsize);
    unsigned char *buf = static_cast<unsigned char*>(malloc(bufsize));

    int pos = 0;
    MPI_Pack(&send_pkt, 1, packet_type, buf, bufsize, &pos, MPI_COMM_WORLD);
    assert(pos <= bufsize);

    pos = 0;
    MPI_Unpack(buf, bufsize, &pos, &recv_pkt, 1, packet_type, MPI_COMM_WORLD);

    if (memcmp(&send_pkt, &recv_pkt, sizeof(packet)) != 0) {
        printf("❌ MPI pack/unpack mismatch for packet!\n");
        for (size_t i = 0; i < sizeof(packet); i++) {
            unsigned char s = ((unsigned char*)&send_pkt)[i];
            unsigned char r = ((unsigned char*)&recv_pkt)[i];
            if (s != r)
                printf("  Byte %zu: send=%02X recv=%02X\n", i, s, r);
        }
    } else {
        printf("✅ MPI pack/unpack identical for packet\n");
    }

    free(buf);
}


void test_packet_sendrecv(MPI_Datatype packet_type) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        printf("⚠️  Skipping packet send/recv test (need at least 2 ranks)\n");
        return;
    }

    packet pkt, recv_pkt, tmp_pkt;
    if (rank == 0){
        pkt.state.state_thus_far.uint64[0] = 0xF000000F;
        pkt.state.state_thus_far.uint64[1] = 0xF008800F;
        pkt.state.curr_spin = 4;
        pkt.state.num_spinon_pairs = 11;
        pkt.available = 0;
    } else {   
        pkt.state.state_thus_far.uint64[0] = 0xABABCDCD;
        pkt.state.state_thus_far.uint64[1] = 0xFEFE0101;
        pkt.state.curr_spin = 1000;
        pkt.state.num_spinon_pairs = 0;
        pkt.available = 999;
    }


    MPI_Sendrecv(&pkt, 1, packet_type, (rank +1) % size, 123,
            &recv_pkt, 1, packet_type, (rank + size - 1) % size,
            123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&recv_pkt, 1, packet_type, (rank +size -1) % size, 123,
            &tmp_pkt, 1, packet_type, (rank + size + 1) % size,
            123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if ( memcmp(&tmp_pkt, &pkt, sizeof(tmp_pkt)) != 0 ){      
        throw std::logic_error("Round trip was not a no-op!");
    }

    assert(pkt.state.state_thus_far == tmp_pkt.state.state_thus_far);

}



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    test_vtree_node_layout();
    MPI_Datatype t = create_vtree_node_type();
    test_vtree_node_serialization(t);
    test_vtree_node_sendrecv(t);

    test_packet_layout();
    MPI_Datatype p = create_packet_type();
    test_packet_serialization(p);

    test_packet_sendrecv(p);


    MPI_Finalize();
    return 0;
}

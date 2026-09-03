// getconfig — dump the MPI and HDF5 build/runtime diagnostics that actually
// bite us in practice (library versions, and crucially which HDF5 I/O filters
// are compiled in, since a missing deflate encoder silently writes basis files
// uncompressed).
//
//   getconfig            # run standalone
//   mpirun -n 1 getconfig
#include <cstdio>
#include <mpi.h>
#include <hdf5.h>

namespace {

void print_mpi() {
    std::printf("== MPI ==\n");

    int std_major = 0, std_minor = 0;
    MPI_Get_version(&std_major, &std_minor);
    std::printf("  standard         : %d.%d\n", std_major, std_minor);

    char libver[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
    int liblen = 0;
    if (MPI_Get_library_version(libver, &liblen) == MPI_SUCCESS && liblen > 0) {
        // library version strings are often multi-line; keep to the first line
        for (int i = 0; i < liblen; ++i) {
            if (libver[i] == '\n' || libver[i] == '\r') { libver[i] = '\0'; break; }
        }
        std::printf("  library          : %s\n", libver);
    }

    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::printf("  comm world size  : %d (this rank %d)\n", size, rank);
}

void print_one_filter(const char* name, H5Z_filter_t id) {
    htri_t avail = H5Zfilter_avail(id);
    if (avail <= 0) {
        std::printf("  %-10s : NOT AVAILABLE\n", name);
        return;
    }
    unsigned info = 0;
    H5Zget_filter_info(id, &info);
    const bool enc = info & H5Z_FILTER_CONFIG_ENCODE_ENABLED;
    const bool dec = info & H5Z_FILTER_CONFIG_DECODE_ENABLED;
    std::printf("  %-10s : available  (encode: %s, decode: %s)\n",
                name, enc ? "yes" : "NO", dec ? "yes" : "NO");
}

void print_hdf5() {
    std::printf("== HDF5 ==\n");

    unsigned maj = 0, min = 0, rel = 0;
    H5get_libversion(&maj, &min, &rel);
    std::printf("  runtime library  : %u.%u.%u\n", maj, min, rel);
    std::printf("  compiled headers : %d.%d.%d\n",
                H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE);
    if (H5check_version(H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE) < 0)
        std::printf("  WARNING: header/runtime version mismatch\n");

    std::printf("  I/O filters:\n");
    print_one_filter("deflate", H5Z_FILTER_DEFLATE);   // gzip — what basis writes use
    print_one_filter("shuffle", H5Z_FILTER_SHUFFLE);
    print_one_filter("szip",    H5Z_FILTER_SZIP);
    print_one_filter("nbit",    H5Z_FILTER_NBIT);
    print_one_filter("scaleoff", H5Z_FILTER_SCALEOFFSET);
    print_one_filter("fletcher", H5Z_FILTER_FLETCHER32);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        print_mpi();
        std::printf("\n");
        print_hdf5();
        std::fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}

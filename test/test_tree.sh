#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

data_dir="test/data"
tmp_dir="test/tmp/shards"
exec_dir="build/components/build_basis"

stem="pyro_2_2_2x0,4,4b4,0,4b4,4,0b1"
lfile="test/lattice_files/$stem.json"
ref_outfile_h5="${data_dir}/${stem}.reference.basis.h5"
mkdir -p "$tmp_dir"

function h5diff_cmp() {
    output=$(h5diff -c "$1" "$2" 2>&1)
    code=$?
    if [[ "$output" == *"Not comparable"* ]]; then
        return 1
    else
        return $code
    fi
}

function check_diff() {
    if h5diff_cmp $1 "$ref_outfile_h5"; then
    echo -e "\033[32;1m[hdf5] $2 test passed\033[0m"
    else
    echo -e "\033[31;1m[hdf5] $2 test failed\033[0m:\
        h5diff $1 $ref_outfile_h5 != 0"
    return 1
    fi
}

out_file="$tmp_dir/merged-1t-${stem}"
"${exec_dir}/sbsearch" -j 1 "$lfile" -o "$tmp_dir"
"${exec_dir}/merge_shards" "$tmp_dir/manifest-$stem.json" -o "${out_file}"
check_diff "${out_file}.h5" "single-thread"


out_file="$tmp_dir/merged-4t-${stem}"
"${exec_dir}/sbsearch" -j 4 "$lfile" -o "$tmp_dir"
"${exec_dir}/merge_shards" "$tmp_dir/manifest-$stem.json" -o "${out_file}"
check_diff "${out_file}.h5"  "multi-thread"


out_file="$tmp_dir/merged-mpi-1-${stem}"
"${exec_dir}/sbsearch_mpi" "$lfile" -o "$tmp_dir"
"${exec_dir}/merge_shards" "$tmp_dir/manifest-$stem.json" -o "${out_file}"
check_diff "${out_file}.h5"  "MPI 1-job"


out_file="$tmp_dir/merged-mpi-4-${stem}"
mpirun -n 4 "${exec_dir}/sbsearch_mpi" "$lfile" -o "$tmp_dir"
"${exec_dir}/merge_shards" "$tmp_dir/manifest-$stem.json" -o "${out_file}"
check_diff "${out_file}.h5"  "MPI 4-job"


# 8490  build/sbsearch ../lattice_files/pyro_2,0,0_0,3,0_0,0,3.json -b 100 -o ~/tmp -j 4
# 8491  build/sbsearch ../lattice_files/pyro_1,0,0_0,2,0_0,0,8.json -b 100 -o ~/tmp -j 4
# 8492  build/sbsearch ../lattice_files/pyro_1,0,0_0,2,0_0,0,9.json -b 100 -o ~/tmp -j 4
# 8504  build/sbsearch ../lattice_files/pyro_1,0,0_0,2,0_0,0,4.json -j 8 -o ~/tmp
# 8510  build/sbsearch ../lattice_files/pyro_1,0,0_0,2,0_0,0,2.json -j 8 -o ~/tmp
# 9885  build/components/build_basis/sbsearch_mpi ../lattice_files/pyro_2,0,0_0,2,0_0,0,2.json 0 --tmp_outpath ../tmp

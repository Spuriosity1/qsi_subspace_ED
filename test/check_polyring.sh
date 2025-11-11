#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ $# -ne 2 ]; then
    echo "Usage: $0 <path-to-lattice-json> <seed-hi>,<seed-lo>"
    exit 1
fi

lfile="$1"
stem="$(basename "$lfile" .json)"
data_dir="test/data"
tmp_dir="test/tmp/shards"
exec_dir="build/components/build_basis"

mkdir -p "$tmp_dir"

names=()
outputs=()

function h5diff_cmp() {
    output=$(h5diff -c "$1" "$2" 2>&1)
    code=$?
    if [[ "$output" == *"Not comparable"* ]]; then
        return 1
    else
        return $code
    fi
}

function check_agreement() {
    local base="$1"
    local test="$2"
    local label="$3"
    if h5diff_cmp "$base" "$test"; then
        echo -e "\033[32;1m[hdf5] $label agrees with single-threaded output\033[0m"
    else
        echo -e "\033[31;1m[hdf5] $label disagrees with single-threaded output\033[0m"
        return 1
    fi
}

run_test() {
    local name="$1"
    shift
    local out_file="$tmp_dir/merged-${name}-${stem}"

    echo "=== Running: $name ==="
    echo "Command: $@"
    time "$@"

    echo "Command: ${exec_dir}/merge_shards $tmp_dir/manifest-$stem.json -o ${out_file}"
    "${exec_dir}/merge_shards" "$tmp_dir/manifest-$stem.json" -o "${out_file}"

    names+=("$name")
    outputs+=("${out_file}.h5")

    # Clean up only shard files, not merged output
    find "$tmp_dir" -type f ! -name "merged-*" -delete
    echo
}


seed="$2"

run_test "single-thread" "build/polyring_basis_mpi" "$lfile" "$seed" "$tmp_dir"
run_test "MPI 4-job" mpirun -n 4 "build/polyring_basis_mpi" "$lfile" "$seed" "$tmp_dir"

echo "=== Comparing outputs to single-threaded version ==="
base="${outputs[0]}"
for i in "${!names[@]}"; do
    if [[ "${names[$i]}" != "single-thread" ]]; then
        check_agreement "$base" "${outputs[$i]}" "${names[$i]}"
    fi
done


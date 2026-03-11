#!/bin/bash

# Configurable parameters
N=${1:-50}                  # Number of datasets to process
HDF5_FILE="../lattice_files/pyro_2,0,0_0,3,0_0,0,4.0.basis.partitioned.h5"
JSON_FILE="../lattice_files/pyro_2,0,0_0,3,0_0,0,4.json"

NRANKS=${2:-1}
OUTPUT_CSV="results_n${NRANKS}.csv"

echo "Processing $N files, $NRANKS ranks"

# Write CSV header
echo "basis_size,max_memory_kb,time_real_s,time_user_s,time_sys_s,dataset_name" > "$OUTPUT_CSV"

# Read all datasets from h5ls
datasets=($(h5ls -r "$HDF5_FILE" | grep '^/' | awk '{print $1}'))

# Shuffle datasets
shuf_datasets=($(shuf -e "${datasets[@]}"))

# Initialize counter
success_count=0


max_size=80000000
min_size=10000000

for dataset in "${shuf_datasets[@]}"; do
    # Stop if we have N successful outputs
    if [ "$success_count" -ge "$N" ]; then
        break
    fi

    # Extract the number of rows (e.g., 5912 from {5912/Inf, 2})
    size=$(h5ls -r "$HDF5_FILE" | grep "$dataset" | awk -F'[{/,}]' '{print $3}')

    # Skip if size is empty
    if [ -z "$size" ]; then
        echo "Warning: Could not determine size for $dataset, skipping."
        continue
    fi
    # skip if wrong size
    if [[ $size -gt $max_size || $size -lt $min_size ]]; then
        continue
    fi

    echo "Processing $dataset (size $size rows)"

    # Run command with /usr/bin/time and capture output
    output=$(/usr/bin/time -v mpirun -n $NRANKS build/bench/bench_apply_mpi "$JSON_FILE" --sector "$dataset" --trim 2>&1)

    # Extract time and memory info
    time_real=$(echo "$output" | grep "Elapsed (wall clock) time" | awk '{print $5}')
    time_user=$(echo "$output" | grep "User time" | awk '{print $4}')
    time_sys=$(echo "$output" | grep "System time" | awk '{print $4}')
    max_memory_kb=$(echo "$output" | grep "Maximum resident set size" | awk '{print $6}')

    # Write to CSV
    echo "$size,$max_memory_kb,$time_real,$time_user,$time_sys,$dataset" >> "$OUTPUT_CSV"

    # Increment success counter
    ((success_count++))
done

echo "Successfully processed $success_count datasets. Results saved to $OUTPUT_CSV"


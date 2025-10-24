#!/bin/bash

# Benchmark runtime of gen_spinon_basis with a variable number of threads.
# Operatrs on increasingly large unit cells, 1 x 2 x j for j=1...10.

if [ -z "$1" ]; then
    echo "Usage: $0 <n_threads>"
    exit 1
fi


benchfile="$(date +"%Y-%m-%dT%H-%M-%S")_time.txt"

for j in `seq 2 10`; do
    start=`perl -MTime::HiRes=time -e 'printf "%.9f\n", time'`
    cmd="bin/gen_spinon_basis bench/test_data/pyro_1,0,0_0,2,0_0,0,$j.json --n_threads $1"
    echo $cmd >> $benchfile
    $cmd > /dev/null
    end=`perl -MTime::HiRes=time -e 'printf "%.9f\n", time'`
	runtime=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.6f\n", e - s }')
    echo "Runtime: $runtime s" >> $benchfile
    echo "$runtime"
done

rm bench/test_data/*.h5

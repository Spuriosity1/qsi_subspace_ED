#!/bin/bash

for j in `seq 2 10`; do
    start=`perl -MTime::HiRes=time -e 'printf "%.9f\n", time'`
	build/gen_spinon_basis bench/test_data/pyro_1,0,0_0,2,0_0,0,$j.json > /dev/null
    end=`perl -MTime::HiRes=time -e 'printf "%.9f\n", time'`
	runtime=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.6f\n", e - s }')
    echo "$runtime"
done

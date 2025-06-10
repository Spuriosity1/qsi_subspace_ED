#!/bin/bash

for j in `seq 1 5`; do
    start=`perl -MTime::HiRes=time -e 'printf "%.9f\n", time'`
	build/gen_spinon_basis bench/pyro_1,0,0_0,2,0_0,0,$j.json > /dev/null
    end=`perl -MTime::HiRes=time -e 'printf "%.9f\n", time'`
    runtime=$( echo "$end - $start" | bc -l )
    echo $runtime
done

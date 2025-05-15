#!/bin/bash

lfile="../lattice_files/pyro_2_2_2x0,4,4b4,0,4b4,4,0b1.json"


function timeit {
	echo "$@"
	local start=`perl -MTime::HiRes=time -e 'printf "%.9f\n", time'`
	"$@" > /dev/null
	local stop=`perl -MTime::HiRes=time -e 'printf "%.9f\n", time'`
    echo $( echo "$stop - $start" | bc -l )
}

timeit ../build/gen_spinon_basis $lfile

for nt in `seq 1 8`; do
	timeit ../build/gen_spinon_basis_parallel $lfile $nt 
done;







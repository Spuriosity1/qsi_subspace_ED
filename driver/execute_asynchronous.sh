#!/bin/bash

# Driver script for running many different MC simulations in parallel

# usage: execute_parallel PLANFILE NUM_THREADS



if [[ $# -lt 2 ]]; then
    echo Usage: execute_parallel PLANFILE NUM_THREADS
    exit 1
fi

NUM_THREADS=$2


# PLANFILE should be in shell script format,
# ../bin/executable arg1 arg2 ...

arr=()
while IFS= read -r line; do
   arr+=("$line")
done <"${1}"

NTASKS="${#arr[@]}"
echo "Scheduling $NTASKS jobs over $NUM_THREADS cores"

mkdir -p rundata

for i in `seq 0 $(( $NUM_THREADS - 1 ))`; do
    (
        for j in `seq $i $NUM_THREADS $(( $NTASKS - 1 ))`; do
            echo $j >> "${1}".track
            eval "${arr[$j]} > rundata/$i.$j.out 2> rundata/$i.$j.err"
        done
    ) & 
    echo "[ thread $i ] started process $!"
done

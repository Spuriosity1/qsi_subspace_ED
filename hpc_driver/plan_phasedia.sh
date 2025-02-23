#!/bin/bash

# Check if the user provided an input file
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <latspec> <initial_index>"
    exit 1
fi

lattice=`basename $1`

sector_decomp="../basis_partitions/$lattice"

if [ ! -d "$sector_decomp" ]; then
	echo "Error: sector decomposition not found at $sector_decomp"
	exit 1
fi

sector_list=`ls $sector_decomp`;


latfile="../lattice_files/$lattice.json"



index=$2
for p in $sector_list; do

	BASE_CMD="python3 ../scripts/phase_dia.py $latfile --db_repo ../../ed_data/ --basis_file ../basis_partitions/$lattice/$p"
       	echo $BASE_CMD -x -3 -X 0 -d 0.1 --index $index
        let index+=1
       	echo $BASE_CMD -x -0.1 -X 3 -d 0.1 --index $index
        let index+=1
done

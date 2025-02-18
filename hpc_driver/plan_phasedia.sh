#!/bin/bash

# Check if the user provided an input file
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <latspec>"
    exit 1
fi

lattice=`basename $1`

sector_decomp="../basis_partitions/$lattice"

if [ ! -d "$sector_decomp" ]; then
	echo "Error: sector decomposition not found at $sector_decomp"
	exit 1
fi

latfile="../lattice_files/$lattice.json"


for p in `ls $sector_decomp`; do
       	echo "python3 ../scripts/phase_dia.py $latfile -x -3 -X 0 -d 0.1 --db_repo ../../ed_data/ --basis_file ../basis_partitions/$lattice/$p"
       	echo "python3 ../scripts/phase_dia.py $latfile -x -0 -X 3 -d 0.1 --db_repo ../../ed_data/ --basis_file ../basis_partitions/$lattice/$p"
done

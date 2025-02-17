#!/bin/bash

#lattice="pyro_1_3_6x0,-20,-4b4,20,0b0,-8,0b1"
#lattice="pyro_1_3_6x-12,-4,-8b0,-4,-4b4,0,4b1"
lattice="pyro_1_3_3x0,4,4b4,0,4b4,4,0b1"

latfile="../lattice_files/$lattice.json"

rm phase_dia.plan

for p in `ls ../basis_partitions/$lattice`; do
       	echo "python3 ../scripts/phase_dia.py $latfile -x -3 -X 3 -d 0.1 --db_path ../../ed_data/results.db --basis_file ../basis_partitions/$lattice/$p" >> phase_dia.plan
done

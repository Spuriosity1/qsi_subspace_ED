#!/bin/bash
ldir="test/lattice_files"
timefile="$(date +"%Y-%m-%dT%H-%M-%S")_time.txt"
time build/gen_spinon_basis $ldir/pyro_2_2_3x0,4,4b4,0,4b4,4,0b1.json 0 test | tee $timefile


lfile="$ldir/pyro_2_3_3x0,4,4b4,0,4b4,4,0b1.json"
build/gen_spinon_basis $lfile &
pid=$!
samplefile="$(date +"%Y-%m-%dT%H-%M-%S")_profile.txt"
sample $pid 20 -file $samplefile
echo $pid

less $samplefile

kill $pid





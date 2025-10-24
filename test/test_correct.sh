#!/bin/bash

LATSTEM="pyro_2,0,0_0,2,0_0,0,3"

mkdir ../tmp_data

bin/gen_spinon_basis "test/lattice_files/$LATSTEM.json"
bin/diag_DOQSI_ham "test/lattice_files/$LATSTEM.json" --Jpm 0.02 --B 0.1 0.1 0.1 -n 1 -N 1 --algorithm mfeig0 -k 20 --rng_seed 11 --atol -9 --rtol -9 -o ../tmp_data

LOGFILE=test/tmp/test_correct.log

REF_EIGFILE="test/data/REFERENCE_Jpm=0.0200%Bx=0.1000%By=0.1000%Bz=0.1000%.eigs.h5"
NEW_EIGFILE="../tmp_data/Jpm=0.0200%Bx=0.1000%By=0.1000%Bz=0.1000%.eigs.h5"

h5diff -d 1e-8 -r $REF_EIGFILE $NEW_EIGFILE  /eigenvalues /eigenvalues > $LOGFILE
if [ $? -ne 0 ]; then echo "Eigenvalues differ, see $LOGFILE"; fi


h5diff -d 1e-4 -r $REF_EIGFILE $NEW_EIGFILE  /eigenvectors /eigenvectors >> $LOGFILE
if [ $? -ne 0 ]; then echo "Eigenvectors differ, see $LOGFILE"; fi

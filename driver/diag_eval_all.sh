#!/bin/bash

for f in `ls ../out/misaligned_B/222/*.eigs.h5`; do
    echo $f
    if [ ! -f "${f%.h5}.out.h5" ]; 
    then bin/eval_observables $f; fi;  
done

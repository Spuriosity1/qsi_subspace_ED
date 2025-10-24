#!/bin/bash

XPATH="../bin/bench"

function benchit () {
    echo $1
    valgrind $XPATH/$1 0
    valgrind $XPATH/$1 1000
    valgrind $XPATH/$1 10000
}

benchit umap_memtest
benchit map_memtest
benchit vector_memtest
benchit ankerl_memtest



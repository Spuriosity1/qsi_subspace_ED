#!/usr/bin/env julia

include("calc_Cv_ftlm.jl")
LinearAlgebra.BLAS.set_num_threads(1)


if length(ARGS) < 3
    println("Usage: process_one_Cv.jl <matrix.mat.mtx> <dense> <out_file> [-v]")
    return
end

mtx_file=ARGS[1]
dense= first(ARGS[2]) in ['t', 'T', 'y', 'Y']
out_file=ARGS[3]

v="-v" in ARGS

try
    process_file(mtx_file, dense, out_file, verbose=v) 
catch
    @error "Error processing $(mtx_file)"
end

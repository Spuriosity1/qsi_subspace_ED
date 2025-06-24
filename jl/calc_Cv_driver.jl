#!/usr/bin/env julia

include("calc_Cv_ftlm.jl")
using JLD2, Printf, FilePathsBase, ProgressMeter




function process_file(mtx_file, dense, out_file)

    n_samples = 200
    Tmin = 1e-5

    kry_settings = Lanczos(krylovdim=20)
    steps = 300


    H = MatrixMarket.mmread(mtx_file)
    H isa SparseMatrixCSC || error("Matrix is not sparse")

    if dense || size(H, 1) < 100
        β, Evals = dense_thermal_evolution(H, H; Tmin=Tmin, steps=steps)
        jldsave(out_file; beta=β, E_expect=Evals, n_samples=1, method="dense")
    else
        β, Evals = thermal_evolution(H, H; Tmin=Tmin, 
            steps=300,
            n_samples=n_samples,
            kry_settings=kry_settings,
            seed=1000)

        jldsave(out_file; beta=β, E_expect=Evals, n_samples=n_samples, method="sparse", steps=steps)
    end

end

function main()
    if length(ARGS) < 1
        println("Usage: process_shape.jl <shape> [--force] [--dense]")
        return
    end

    shape = ARGS[1]
    force = "--force" in ARGS
    dense = "--dense" in ARGS
    

    INDIR = joinpath(@__DIR__, "..","..", "mtx", shape)
    ODIR = joinpath(@__DIR__, "..","..", "out", "Cv_DOQSI", shape)
    isdir(ODIR) || mkpath(ODIR)

    files = filter(f -> endswith(f, ".mat.mtx"), readdir(INDIR; join=true))

    println("Checking $(length(files)) infiles")

    prog = Progress(length(files))
    Threads.@threads for i in eachindex(files)
#    for i in eachindex(files)
        next!(prog)
        mtx_file = files[i]

        fname = basename(mtx_file)
        base = replace(fname, r"\.mat\.mtx$" => "")
        out_file = joinpath(ODIR, base * ".jld2")

        @info "Processing: $fname"
        if isfile(out_file) && !force
            @info "Skipping existing file: $out_file"
            continue
        end
        try
            process_file(mtx_file, dense, out_file) 
        catch e
            @error "Error processing $(mtx_file)"
        end
    end
    finish!(prog)
end

println("Begin")
main()


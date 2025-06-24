#!/usr/bin/env julia

include("calc_Cv_ftlm.jl")
using JLD2, Printf, FilePathsBase

function main()
    if length(ARGS) < 1
        println("Usage: process_shape.jl <shape> [--force]")
        exit(1)
    end

    shape = ARGS[1]
    force = "--force" in ARGS

    INDIR = joinpath(@__DIR__, "..","..", "mtx", shape)
    ODIR = joinpath(@__DIR__, "..","..", "out", "Cv_DOQSI", shape)
    isdir(ODIR) || mkpath(ODIR)

    files = filter(f -> endswith(f, ".mat.mtx"), readdir(INDIR; join=true))
    n_samples = 20
    Tmin = 1e-6

    kry_settings = Lanczos(krylovdim=20)

    Threads.@threads for i in eachindex(files)
    mtx_file = files[i]
        fname = basename(mtx_file)
        base = replace(fname, r"\.mat\.mtx$" => "")
        out_file = joinpath(ODIR, base * ".jld2")

        if isfile(out_file) && !force
            @info "Skipping existing file: $out_file"
            continue
        end

        @info "Processing: $fname"
        try
            H = MatrixMarket.mmread(mtx_file)
            H isa SparseMatrixCSC || error("Matrix is not sparse")


            if size(H, 1) < 5000
                β, Evals = dense_thermal_evolution(H, H; Tmin=Tmin)
                jldsave(out_file; beta=β, E_expect=Evals, n_samples=1, method="dense")
            else
                β, Evals = thermal_evolution(H, H; Tmin=Tmin,
                    n_samples=n_samples,
                    kry_settings=kry_settings)

                jldsave(out_file; beta=β, E_expect=Evals, n_samples=n_samples, method="sparse")
            end

        catch e
            @error "Error processing $mtx_file: $e"
        end
    end
end

main()


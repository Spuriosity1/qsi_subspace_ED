#!/usr/bin/env julia
using HDF5

include("calc_Cv_ftlm.jl")
using Printf, FilePathsBase, ProgressMeter


PROJROOT= joinpath(@__DIR__, "..")

function rename_mtx_as_out(ODIR, mtx_file)
    fname = basename(mtx_file)
    base = replace(fname, r"\.mat\.mtx$" => "")
    return joinpath(ODIR, base * ".jld2")
end

function process_all(files, ODIR, dense=false) prog = Progress(length(files))
    Threads.@threads for i in eachindex(files)
        next!(prog)
        mtx_file = files[i]

        out_file = rename_mtx_as_out(ODIR, mtx_file)

        @info "Processing: $(basename(mtx_file))"
        try
            process_file(mtx_file, dense, out_file) 
        catch
            @error "Error processing $(mtx_file)"
        end
    end
    finish!(prog)
end

function list_and_sort_basis_datasets(filename::String)
    basis_datasets = Dict{String, Int}()

    h5open(filename, "r") do file
        for name in keys(file)
            # Check if it starts with "basis" and is a dataset
            if startswith(name, "basis")
                dset = file[name]
                basis_datasets[name] = length(dset)
            end
        end
    end

    # Sort the dataset names by their size (ascending)
    return basis_datasets
end

proc_one=joinpath(@__DIR__, "..", "jl", "process_one_Cv.jl")

function print_all(files, ODIR, script_name, dense=false)
    open(script_name, "a") do io
        for mtx_file in files
            out_file = rename_mtx_as_out(ODIR, mtx_file)

            cmd = `julia -t 1 $proc_one $mtx_file $dense $out_file`
            println(io, join(cmd.exec, " "))
        end
    end
    run(`chmod +x $script_name`)
    println("Wrote run script to $script_name")
end

function main()
    if length(ARGS) < 1
        println("Usage: process_shape.jl <lattice> <Jpm> <Bx> <By> <Bz> [--sectors] [--force] [--dense]")
        return
    end

    lattice_file = ARGS[1]
    Jpm_s = ARGS[2]
    B =ARGS[3:5]

    force = "--force" in ARGS
    dense = "--dense" in ARGS
    use_sectors = "--sectors" in ARGS
    
    lattice_stem, ext = splitext(lattice_file)
    if ext != ".json"
        @error "Lattice file $(lattice_file) invalid: must be a .json"
    end
    lname = basename(lattice_stem)
    
    if use_sectors 
        basis_file = lattice_stem * ".0.basis.partitioned.h5"
    else
        basis_file = lattice_stem * ".0.basis.h5"
    end


    dset_sizes = list_and_sort_basis_datasets(basis_file)

    # generate all the .mtx files (<30 mins total)
    for dset_name in keys(dset_sizes)
        INDIR = joinpath(PROJROOT,"..", "mtx", lname*dset_name)
        exe = joinpath(PROJROOT, "build", "build_hamiltonian")
        
        isdir(INDIR) || mkpath(INDIR)
        cmd=`$(exe) $(lattice_file) --Jpm $Jpm_s --B $B -o $(INDIR)`
        if use_sectors
            cmd = `$cmd --sector $dset_name`
        end
        println(cmd)
        run(cmd)
    end


    script_name = joinpath(@__DIR__, "run_Cv_$(lname).plan")
    rm(script_name)


    # run the easy ones
    for (dset_name, size) in dset_sizes
        INDIR = joinpath(@__DIR__, "..","..", "mtx", lname*dset_name)
        ODIR = joinpath(@__DIR__, "..","..", "out", "Cv_DOQSI", lname*dset_name)
        isdir(ODIR) || mkpath(ODIR)
        files = []
        for file in filter(f -> endswith(f, ".mat.mtx"), readdir(INDIR; join=true))
            out_file = rename_mtx_as_out(ODIR, file)
            if isfile(out_file) && !force
                @info "Skipping existing file: $out_file"
                continue
            end
            push!(files, file)
        end

        println("Processing $(length(files)) infiles")

        if size < 1_000
            process_all(files, ODIR, dense)
        else
            print_all(files, ODIR, script_name, dense)
        end
    end
end

println("Begin")
main()


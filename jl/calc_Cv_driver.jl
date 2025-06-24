#!/usr/bin/env julia

include("calc_Cv_ftlm.jl")
using Printf, FilePathsBase, ProgressMeter

function rename_mtx_as_out(ODIR, mtx_file)
    fname = basename(mtx_file)
    base = replace(fname, r"\.mat\.mtx$" => "")
    return joinpath(ODIR, base * ".jld2")
end

#function process_all(files, ODIR, dense=false)
#    prog = Progress(length(files))
#    Threads.@threads for i in eachindex(files)
#        next!(prog)
#        mtx_file = files[i]
#
#        out_file = rename_mtx_as_out(ODIR, mtx_file)
#
#        @info "Processing: $(basename(mtx_file))"
#        try
#            process_file(mtx_file, dense, out_file) 
#        catch
#            @error "Error processing $(mtx_file)"
#        end
#    end
#    finish!(prog)
#end


function print_all(files, ODIR, shape, dense=false)
    script_name = joinpath(@__DIR__, "run_Cv_$(shape).sh")
    open(script_name, "w") do io
        println(io, "#!/bin/bash\n")
        for mtx_file in files
            out_file = rename_mtx_as_out(ODIR, mtx_file)

            cmd = `julia -t 1 jl/process_one_Cv.jl $mtx_file $dense $out_file`
            println(io, join(cmd.exec, " "))
        end
    end
    run(`chmod +x $script_name`)
    println("Wrote run script to $script_name")
end

function main()
    if length(ARGS) < 1
        println("Usage: process_shape.jl <shape> [--force] [--dense] [--calculate]")
        return
    end

    shape = ARGS[1]
    force = "--force" in ARGS
    dense = "--dense" in ARGS
#    calculate = "--calculate" in ARGS
    

    INDIR = joinpath(@__DIR__, "..","..", "mtx", shape)
    ODIR = joinpath(@__DIR__, "..","..", "out", "Cv_DOQSI", shape)
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
    
#    if calculate
#        process_all(files, ODIR, dense)
#    else
    print_all(files, ODIR, shape, dense)
#    end
end

println("Begin")
main()


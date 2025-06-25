
import Pkg; Pkg.activate("$(@__DIR__)/../")

using PackageCompiler

create_sysimage([:JLD2, :Printf, :FilePathsBase, :MatrixMarket, 
:SparseArrays, :KrylovKit, :Random],
    sysimage_path="CvSysimage.so",
    precompile_execution_file="calc_Cv_ftlm.jl")



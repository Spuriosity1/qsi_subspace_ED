import Pkg; Pkg.activate("$(@__DIR__)/../")

using MatrixMarket
using SparseArrays
using LinearAlgebra
using KrylovKit
using Random
using SparseArrays

"""
    build_and_load_hamiltonian(exe::String, lattice_file::String;
                               Jpm::Float64, B::Tuple{Float64,Float64,Float64}, output_dir::String)

Runs the C++ executable to build the Hamiltonian, parses the output,
and loads the resulting sparse matrix from the MatrixMarket file.

Returns:
- `H::SparseMatrixCSC`: the loaded sparse Hamiltonian
"""
function build_and_load_hamiltonian(mtx_path::String)

    # Load and return sparse matrix
    H = MatrixMarket.mmread(mtx_path)
    if !(H isa SparseMatrixCSC)
        error("Loaded matrix is not sparse.")
    end

    return H
end

"""
    thermal_evolution(H, O; Tmin, n=1, steps=100, krylovdim=30, seed=nothing)

Imaginary-time evolve from β = 0 to β = 1/Tmin, computing the expectation
⟨O⟩(β) = ⟨ψ(β)|O|ψ(β)⟩ / ⟨ψ(β)|ψ(β)⟩

Returns:
- βs::Vector{Float64}: inverse temperatures
- Evals::Matrix{Float64}: n x steps matrix of observable expectations
"""
function thermal_evolution(H::SparseMatrixCSC, O::Union{SparseMatrixCSC, Function};
                           Tmin::Float64, n_samples::Int = 1, steps::Int = 100,
                           kry_settings::Lanczos = Lanczos(),
                           seed = nothing)

    N = size(H, 1)
    β_max = 1 / Tmin
    dβ = β_max / steps
    βs = collect(0.0:dβ:β_max)

    if seed !== nothing
        Random.seed!(seed)
    end

    sums = zeros(length(βs))


    ψ = randn(N,n_samples) + 1im * randn(N,n_samples)
    for i in 1:n_samples
        ψ[:,i] ./= norm(ψ[:,i])
    end

    @assert norm(H - adjoint(H)) < 1e-8

    for (j, β) in enumerate(βs)
        all_ok = true

        for i in 1:n_samples
            if j > 1
                ψ[:,i], info = exponentiate(H, -dβ, ψ[:,i], kry_settings)

                if info.converged ==0
                    all_ok=false
                    @warn "Terminating early (β = $(β))"
                    break
                end
            end
            Ov = (O isa Function) ? O(ψ[:,i]) : O * ψ[:,i]
            tmp = real(dot(ψ[:,i], Ov)) / real(dot(ψ[:,i], ψ[:,i]))
            sums[j] += tmp
        end
        if !all_ok 
            break 
        end
    end

    return βs, sums
end


"""
     dense_thermal_evolution(H::SparseMatrixCSC, O::Union{SparseMatrixCSC, Function};
    steps::Int = 100, Tmin::Float64,)

Imaginary-time evolve from β = 0 to β = 1/Tmin, computing the expectation exactly

Returns:
- βs::Vector{Float64}: inverse temperatures
- Evals::Matrix{Float64}: n x steps matrix of observable expectations
"""
function dense_thermal_evolution(H::SparseMatrixCSC, O::Union{SparseMatrixCSC, Function};
    steps::Int = 100, Tmin::Float64,)

    β_max = 1 / Tmin
    dβ = β_max / steps
    βs = collect(0.0:dβ:β_max)

    H_dense = Matrix(H)

    F = eigen!(H_dense)
    Es = zeros(length(βs))

    for (j, β) in enumerate(βs)
        Z = sum( exp.(-β * F.values) )
        
        Es[j] = sum( F.values .* exp.(-β * F.values) ) / Z
    end

    return βs, Es
end




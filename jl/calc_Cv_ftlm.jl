using MatrixMarket
using SparseArrays
using LinearAlgebra
using KrylovKit
using Random
using JLD2, Printf

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
                           Tmin::Float64, Tmax::Float64=1., n_samples::Int = 1, steps::Int = 100,
                           kry_settings::Lanczos = Lanczos(),
                           seed = nothing)

    vals, vecs, info = eigsolve(H, 1, :LM, issymmetric=true)
    eigval0 = vals[1]

    H -= eigval0 * I # shift the eogvals to avoid overflow


    N = size(H, 1)
    β_max = 1 / Tmin
    β_min = 1 / Tmax

    βs = exp.(range(log(β_min), log(β_max), steps))

    if seed !== nothing
        Random.seed!(seed)
    end

    sums = zeros(length(βs))


    ψ = randn(N,n_samples) + 1im * randn(N,n_samples)
    for i in 1:n_samples
        ψ[:,i] ./= norm(ψ[:,i])
    end

    @assert norm(H - adjoint(H)) < 1e-8
    old_β=0
    for (j, β) in enumerate(βs)
        all_ok = true

        for i in 1:n_samples
            if j > 1
                dβ = β - old_β
                old_β = β
                ψ[:,i], info = exponentiate(H, -dβ, ψ[:,i], kry_settings)

                ψ[:,i] ./= norm(ψ[:,i])

                if info.converged ==0
                    all_ok=false
                    @warn "Terminating early (β = $(β))"
                    break
                end
            end
            Ov = (O isa Function) ? O(ψ[:,i]) : O * ψ[:,i]
            tmp = real(dot(ψ[:,i], Ov)) #/ real(dot(ψ[:,i], ψ[:,i]))
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
    Tmin::Real, Tmax::Real=1., 
    steps::Int = 100, 
    atol=1e-16)

    β_max = 1 / Tmin
    β_min = 1 / Tmax
   
    βs = exp.(range(log(β_min), log(β_max), steps))

    H_dense = Matrix(H)

    spectrum = real.(eigvals(H_dense))
    Es = zeros(length(βs))
    min_E = first(spectrum)
       

    for (j, β) in enumerate(βs)
        w = exp.(-β .* (spectrum.-min_E))
        Z = sum(w)
        Eβ = dot(spectrum, w)
        Es[j] = Eβ / (Z + eps())
    end

    return βs, Es
end



function process_file(mtx_file::String, dense::Bool, out_file::String)
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
            steps=steps,
            n_samples=n_samples,
            kry_settings=kry_settings,
            seed=1000)

        jldsave(out_file; beta=β, E_expect=Evals, n_samples=n_samples, method="sparse", steps=steps)
    end

end

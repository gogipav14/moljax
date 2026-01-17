#!/usr/bin/env julia
"""
Benchmark comparing our FFT-CN solver against DifferentialEquations.jl

This is the key comparison - showing our spectral method vs the state-of-the-art
ODE/PDE solver ecosystem in Julia (SciML).

Reference:
Rackauckas, C. and Nie, Q. (2017). DifferentialEquations.jl - A Performant and
Feature-Rich Ecosystem for Solving Differential Equations in Julia.
Journal of Open Research Software, 5(1).
"""

using Printf
using Statistics
using JSON
using LinearAlgebra
using FFTW

# Install DifferentialEquations if needed
using Pkg
if !haskey(Pkg.project().dependencies, "DifferentialEquations")
    println("Installing DifferentialEquations.jl...")
    Pkg.add("DifferentialEquations")
end
using DifferentialEquations

# Check for CUDA
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

if HAS_CUDA
    using CUDA
    using CUDA.CUFFT
end

println("=" ^ 70)
println("Benchmark: FFT-CN vs DifferentialEquations.jl")
println("=" ^ 70)
println()

println("System Information:")
println("  Julia version: ", VERSION)
println("  DifferentialEquations.jl loaded")
if HAS_CUDA
    println("  GPU: ", CUDA.name(CUDA.device()))
end
println()

# =============================================================================
# Problem Setup: 2D Diffusion Equation
# du/dt = D * (d²u/dx² + d²u/dy²)
# Domain: [0,1]² with periodic BC
# IC: sin(2πx) * sin(2πy)
# =============================================================================

# =============================================================================
# Method 1: Our FFT-Crank-Nicolson (spectral)
# =============================================================================

function fftfreq(n::Int, d::Float64=1.0)
    results = zeros(n)
    N = (n - 1) ÷ 2 + 1
    for i in 0:(N-1)
        results[i+1] = i / (n * d)
    end
    for i in N:(n-1)
        results[i+1] = (i - n) / (n * d)
    end
    return results
end

function solve_fft_cn_cpu(u0::Matrix{Float64}, D::Float64, dt::Float64, n_steps::Int)
    N = size(u0, 1)
    dx = 1.0 / N

    # Laplacian eigenvalues
    kx = fftfreq(N, dx) .* 2π
    ky = fftfreq(N, dx) .* 2π
    KX = repeat(kx, 1, N)
    KY = repeat(ky', N, 1)
    lap_eig = -(KX.^2 .+ KY.^2)

    cn_factor = (1.0 .+ 0.5 .* dt .* D .* lap_eig) ./ (1.0 .- 0.5 .* dt .* D .* lap_eig)

    u = copy(u0)
    u_hat = zeros(ComplexF64, N, N)
    plan_fft = FFTW.plan_fft!(u_hat)
    plan_ifft = FFTW.plan_ifft!(u_hat)

    for _ in 1:n_steps
        u_hat .= u
        plan_fft * u_hat
        u_hat .*= cn_factor
        plan_ifft * u_hat
        u .= real.(u_hat)
    end

    return u
end

if HAS_CUDA
function solve_fft_cn_gpu(u0::Matrix{Float64}, D::Float64, dt::Float64, n_steps::Int)
    N = size(u0, 1)
    dx = 1.0 / N

    kx = fftfreq(N, dx) .* 2π
    ky = fftfreq(N, dx) .* 2π
    KX = repeat(kx, 1, N)
    KY = repeat(ky', N, 1)
    lap_eig = -(KX.^2 .+ KY.^2)

    cn_factor = CuArray(ComplexF64.((1.0 .+ 0.5 .* dt .* D .* lap_eig) ./ (1.0 .- 0.5 .* dt .* D .* lap_eig)))

    u = CuArray(ComplexF64.(u0))
    plan_fft = CUFFT.plan_fft!(u)
    plan_ifft = CUFFT.plan_ifft!(u)

    for _ in 1:n_steps
        plan_fft * u
        u .*= cn_factor
        plan_ifft * u
    end

    CUDA.synchronize()
    return Array(real.(u))
end
end

# =============================================================================
# Method 2: DifferentialEquations.jl with finite differences
# This is how most users would solve this PDE using the SciML ecosystem
# =============================================================================

function solve_diffeq(u0::Matrix{Float64}, D::Float64, t_final::Float64;
                      solver=QNDF(), abstol=1e-6, reltol=1e-3)
    N = size(u0, 1)
    dx = 1.0 / N

    # Flatten for ODE solver
    u0_vec = vec(u0)

    # Laplacian with periodic BC using finite differences
    function diffusion_fd!(du, u, p, t)
        D = p
        u_2d = reshape(u, N, N)
        du_2d = reshape(du, N, N)

        # Periodic Laplacian via finite differences
        for i in 1:N
            ip1 = mod1(i + 1, N)
            im1 = mod1(i - 1, N)
            for j in 1:N
                jp1 = mod1(j + 1, N)
                jm1 = mod1(j - 1, N)

                lap = (u_2d[ip1, j] + u_2d[im1, j] + u_2d[i, jp1] + u_2d[i, jm1] - 4*u_2d[i, j]) / dx^2
                du_2d[i, j] = D * lap
            end
        end
    end

    prob = ODEProblem(diffusion_fd!, u0_vec, (0.0, t_final), D)
    sol = solve(prob, solver; abstol=abstol, reltol=reltol, save_everystep=false)

    return reshape(sol.u[end], N, N)
end

# Alternative: Spectral Laplacian for DiffEq (fairer comparison)
function solve_diffeq_spectral(u0::Matrix{Float64}, D::Float64, t_final::Float64;
                                solver=QNDF(), abstol=1e-6, reltol=1e-3)
    N = size(u0, 1)
    dx = 1.0 / N

    # Precompute Laplacian eigenvalues
    kx = fftfreq(N, dx) .* 2π
    ky = fftfreq(N, dx) .* 2π
    KX = repeat(kx, 1, N)
    KY = repeat(ky', N, 1)
    lap_eig = -(KX.^2 .+ KY.^2)

    # FFT plans
    u_hat = zeros(ComplexF64, N, N)
    plan_fft = FFTW.plan_fft!(u_hat)
    plan_ifft = FFTW.plan_ifft!(u_hat)

    # Use real-valued state, convert to complex for FFT
    u0_vec = vec(u0)

    function diffusion_spectral!(du, u, p, t)
        D, lap_eig, plan_fft, plan_ifft, N = p
        u_2d = reshape(u, N, N)

        # FFT to get Laplacian
        u_hat = plan_fft * ComplexF64.(u_2d)
        lap_u_hat = D .* lap_eig .* u_hat
        lap_u = real.(plan_ifft * lap_u_hat)

        du .= vec(lap_u)
    end

    params = (D, lap_eig, plan_fft, plan_ifft, N)
    prob = ODEProblem(diffusion_spectral!, u0_vec, (0.0, t_final), params)
    sol = solve(prob, solver; abstol=abstol, reltol=reltol, save_everystep=false)

    return reshape(sol.u[end], N, N)
end

# =============================================================================
# Benchmark Configuration
# =============================================================================

const GRID_SIZE = 128  # Moderate size for DiffEq (larger is slow)
const D = 0.01
const T_FINAL = 1.0
const N_WARMUP = 2
const N_RUNS = 5

# For FFT-CN: choose dt for ~1000 steps
const DT = T_FINAL / 1000
const N_STEPS = Int(T_FINAL / DT)

println("Configuration:")
println("  Grid size: $GRID_SIZE × $GRID_SIZE")
println("  Diffusion coefficient: $D")
println("  Final time: $T_FINAL")
println("  FFT-CN steps: $N_STEPS (dt = $DT)")
println()

# Create initial condition
x = range(0, 1, length=GRID_SIZE+1)[1:end-1]
y = range(0, 1, length=GRID_SIZE+1)[1:end-1]
X = repeat(collect(x), 1, GRID_SIZE)
Y = repeat(collect(y)', GRID_SIZE, 1)
u0 = sin.(2π .* X) .* sin.(2π .* Y)

# Analytical solution
analytical = exp(-8 * π^2 * D * T_FINAL) .* u0

# =============================================================================
# Run Benchmarks
# =============================================================================

results = Dict{String, Any}()

# --- FFT-CN CPU ---
println("1. FFT-CN (CPU, FFTW)...")
for _ in 1:N_WARMUP
    solve_fft_cn_cpu(u0, D, DT, N_STEPS)
end
fft_cn_times = [(@elapsed solve_fft_cn_cpu(u0, D, DT, N_STEPS)) for _ in 1:N_RUNS]
fft_cn_result = solve_fft_cn_cpu(u0, D, DT, N_STEPS)
for (i, t) in enumerate(fft_cn_times)
    @printf("   Run %d: %.4f s\n", i, t)
end
fft_cn_error = maximum(abs.(fft_cn_result .- analytical))
@printf("   Mean: %.4f s, Error: %.2e\n\n", mean(fft_cn_times), fft_cn_error)
results["fft_cn_cpu"] = Dict(
    "mean_s" => mean(fft_cn_times),
    "std_s" => std(fft_cn_times),
    "error" => fft_cn_error
)

# --- FFT-CN GPU ---
if HAS_CUDA
    println("2. FFT-CN (GPU, CUFFT)...")
    for _ in 1:N_WARMUP
        solve_fft_cn_gpu(u0, D, DT, N_STEPS)
        CUDA.synchronize()
    end
    fft_cn_gpu_times = Float64[]
    for i in 1:N_RUNS
        CUDA.synchronize()
        t = @elapsed begin
            solve_fft_cn_gpu(u0, D, DT, N_STEPS)
            CUDA.synchronize()
        end
        push!(fft_cn_gpu_times, t)
        @printf("   Run %d: %.4f s\n", i, t)
    end
    fft_cn_gpu_result = solve_fft_cn_gpu(u0, D, DT, N_STEPS)
    CUDA.synchronize()
    fft_cn_gpu_error = maximum(abs.(fft_cn_gpu_result .- analytical))
    @printf("   Mean: %.4f s, Error: %.2e\n\n", mean(fft_cn_gpu_times), fft_cn_gpu_error)
    results["fft_cn_gpu"] = Dict(
        "mean_s" => mean(fft_cn_gpu_times),
        "std_s" => std(fft_cn_gpu_times),
        "error" => fft_cn_gpu_error
    )
end

# --- DifferentialEquations.jl QNDF (FD) ---
println("3. DifferentialEquations.jl QNDF (finite differences)...")
for _ in 1:N_WARMUP
    solve_diffeq(u0, D, T_FINAL; solver=QNDF())
end
diffeq_times = [(@elapsed solve_diffeq(u0, D, T_FINAL; solver=QNDF())) for _ in 1:N_RUNS]
diffeq_result = solve_diffeq(u0, D, T_FINAL; solver=QNDF())
for (i, t) in enumerate(diffeq_times)
    @printf("   Run %d: %.4f s\n", i, t)
end
diffeq_error = maximum(abs.(diffeq_result .- analytical))
@printf("   Mean: %.4f s, Error: %.2e\n\n", mean(diffeq_times), diffeq_error)
results["diffeq_qndf_fd"] = Dict(
    "mean_s" => mean(diffeq_times),
    "std_s" => std(diffeq_times),
    "error" => diffeq_error
)

# --- DifferentialEquations.jl Tsit5 (explicit, non-stiff) ---
# Note: Rodas5P is extremely slow for this problem (sparse linear solves)
# Tsit5 is a common explicit solver choice
println("4. DifferentialEquations.jl Tsit5 (explicit, finite differences)...")
for _ in 1:N_WARMUP
    solve_diffeq(u0, D, T_FINAL; solver=Tsit5(), abstol=1e-6, reltol=1e-3)
end
tsit5_times = [(@elapsed solve_diffeq(u0, D, T_FINAL; solver=Tsit5(), abstol=1e-6, reltol=1e-3)) for _ in 1:N_RUNS]
tsit5_result = solve_diffeq(u0, D, T_FINAL; solver=Tsit5(), abstol=1e-6, reltol=1e-3)
for (i, t) in enumerate(tsit5_times)
    @printf("   Run %d: %.4f s\n", i, t)
end
tsit5_error = maximum(abs.(tsit5_result .- analytical))
@printf("   Mean: %.4f s, Error: %.2e\n\n", mean(tsit5_times), tsit5_error)
results["diffeq_tsit5_fd"] = Dict(
    "mean_s" => mean(tsit5_times),
    "std_s" => std(tsit5_times),
    "error" => tsit5_error
)

# =============================================================================
# Results Summary
# =============================================================================

println("=" ^ 70)
println("RESULTS SUMMARY ($(GRID_SIZE)×$(GRID_SIZE) grid, t=$T_FINAL)")
println("=" ^ 70)
println()

@printf("%-30s  %10s  %12s  %10s\n", "Method", "Time (s)", "Speedup", "Error")
println("-" ^ 70)

baseline = results["diffeq_qndf_fd"]["mean_s"]

@printf("%-30s  %10.4f  %12s  %10.2e\n",
        "DiffEq QNDF (FD)", results["diffeq_qndf_fd"]["mean_s"], "1.0x (baseline)",
        results["diffeq_qndf_fd"]["error"])

@printf("%-30s  %10.4f  %12.1fx  %10.2e\n",
        "DiffEq Tsit5 (FD)", results["diffeq_tsit5_fd"]["mean_s"],
        baseline / results["diffeq_tsit5_fd"]["mean_s"],
        results["diffeq_tsit5_fd"]["error"])

@printf("%-30s  %10.4f  %12.1fx  %10.2e\n",
        "FFT-CN CPU", results["fft_cn_cpu"]["mean_s"],
        baseline / results["fft_cn_cpu"]["mean_s"],
        results["fft_cn_cpu"]["error"])

if HAS_CUDA
    @printf("%-30s  %10.4f  %12.1fx  %10.2e\n",
            "FFT-CN GPU", results["fft_cn_gpu"]["mean_s"],
            baseline / results["fft_cn_gpu"]["mean_s"],
            results["fft_cn_gpu"]["error"])
end

println()

# =============================================================================
# Save Results
# =============================================================================

output = Dict(
    "julia_version" => string(VERSION),
    "cuda_available" => HAS_CUDA,
    "config" => Dict(
        "grid_size" => GRID_SIZE,
        "D" => D,
        "t_final" => T_FINAL,
        "fft_cn_steps" => N_STEPS,
        "fft_cn_dt" => DT
    ),
    "results" => results
)

if HAS_CUDA
    output["gpu_device"] = CUDA.name(CUDA.device())
end

output_path = joinpath(@__DIR__, "results_diffeq_comparison.json")
open(output_path, "w") do f
    JSON.print(f, output, 2)
end
println("Results saved to: $output_path")

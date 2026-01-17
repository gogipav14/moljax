#!/usr/bin/env julia
"""
Scaling benchmark: Test Julia FFT-CN performance across grid sizes.
Equivalent to Python benchmark_scaling.py.

Tests: 64², 128², 256², 512², 1024²
"""

using Printf
using Statistics
using JSON
using FFTW
using LinearAlgebra

# Check for CUDA availability
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

println("=" ^ 60)
println("Julia FFT-CN Scaling Benchmark")
println("=" ^ 60)
println()

# =============================================================================
# Solver Functions (same as benchmark_cpu_gpu.jl)
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

function create_laplacian_eigenvalues(N::Int, dx::Float64)
    kx = fftfreq(N, dx) .* 2π
    ky = fftfreq(N, dx) .* 2π
    KX = repeat(kx, 1, N)
    KY = repeat(ky', N, 1)
    return -(KX.^2 .+ KY.^2)
end

function solve_diffusion_cpu(u0::Matrix{Float64}, D::Float64, dt::Float64, n_steps::Int)
    N = size(u0, 1)
    dx = 1.0 / N

    lap_eig = create_laplacian_eigenvalues(N, dx)
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

function create_laplacian_eigenvalues_gpu(N::Int, dx::Float64)
    kx = fftfreq(N, dx) .* 2π
    ky = fftfreq(N, dx) .* 2π
    KX = repeat(kx, 1, N)
    KY = repeat(ky', N, 1)
    lap_eig = -(KX.^2 .+ KY.^2)
    return CuArray(lap_eig)
end

function solve_diffusion_gpu(u0::Matrix{Float64}, D::Float64, dt::Float64, n_steps::Int)
    N = size(u0, 1)
    dx = 1.0 / N

    lap_eig = create_laplacian_eigenvalues_gpu(N, dx)
    cn_factor = CuArray(ComplexF64.((1.0 .+ 0.5 .* dt .* D .* Array(lap_eig)) ./ (1.0 .- 0.5 .* dt .* D .* Array(lap_eig))))

    # Work in complex space, use in-place FFT plans
    u = CuArray(ComplexF64.(u0))

    # Create in-place CUFFT plans (much faster than allocating each call)
    plan_fft = CUFFT.plan_fft!(u)
    plan_ifft = CUFFT.plan_ifft!(u)

    for _ in 1:n_steps
        plan_fft * u      # In-place forward FFT
        u .*= cn_factor   # Apply CN factor
        plan_ifft * u     # In-place inverse FFT
    end

    CUDA.synchronize()
    return Array(real.(u))
end

end # HAS_CUDA

# =============================================================================
# Benchmark Configuration
# =============================================================================

const GRID_SIZES = [64, 128, 256, 512, 1024]
const D = 0.01  # Match JAX benchmark
const N_STEPS = 1000  # Match JAX benchmark (1000 steps)
const N_WARMUP = 2
const N_RUNS = 5

println("Configuration:")
println("  Grid sizes: ", GRID_SIZES)
println("  Diffusion coefficient: $D")
println("  Steps per run: $N_STEPS")
println("  Warmup runs: $N_WARMUP")
println("  Benchmark runs: $N_RUNS")
println()

# =============================================================================
# Run Scaling Tests
# =============================================================================

results = Dict{Int, Dict{String, Any}}()

for N in GRID_SIZES
    println("-" ^ 40)
    println("Grid size: $N × $N ($(N^2) DOFs)")
    println("-" ^ 40)

    dx = 1.0 / N
    dt = 0.1 * dx^2 / D  # CFL-like condition

    # Create initial condition
    x = range(0, 1, length=N+1)[1:end-1]
    y = range(0, 1, length=N+1)[1:end-1]
    X = repeat(collect(x), 1, N)
    Y = repeat(collect(y)', N, 1)
    u0 = sin.(2π .* X) .* sin.(2π .* Y)

    # CPU benchmark
    println("  CPU warmup...")
    for _ in 1:N_WARMUP
        solve_diffusion_cpu(u0, D, dt, N_STEPS)
    end

    cpu_times = Float64[]
    for i in 1:N_RUNS
        t = @elapsed solve_diffusion_cpu(u0, D, dt, N_STEPS)
        push!(cpu_times, t)
    end
    cpu_mean = mean(cpu_times)
    @printf("  CPU: %.4f s (%.2f ms/step)\n", cpu_mean, 1000*cpu_mean/N_STEPS)

    result = Dict(
        "N" => N,
        "DOFs" => N^2,
        "dt" => dt,
        "cpu_mean_s" => cpu_mean,
        "cpu_times_s" => cpu_times
    )

    # GPU benchmark
    if HAS_CUDA
        println("  GPU warmup...")
        for _ in 1:N_WARMUP
            solve_diffusion_gpu(u0, D, dt, N_STEPS)
        end

        gpu_times = Float64[]
        for i in 1:N_RUNS
            CUDA.synchronize()
            t = @elapsed begin
                solve_diffusion_gpu(u0, D, dt, N_STEPS)
                CUDA.synchronize()
            end
            push!(gpu_times, t)
        end
        gpu_mean = mean(gpu_times)
        speedup = cpu_mean / gpu_mean
        @printf("  GPU: %.4f s (%.2f ms/step)\n", gpu_mean, 1000*gpu_mean/N_STEPS)
        @printf("  Speedup: %.1fx\n", speedup)

        result["gpu_mean_s"] = gpu_mean
        result["gpu_times_s"] = gpu_times
        result["speedup"] = speedup
    end

    results[N] = result
    println()
end

# =============================================================================
# Summary Table
# =============================================================================

println("=" ^ 60)
println("SCALING SUMMARY")
println("=" ^ 60)
println()

if HAS_CUDA
    @printf("%-8s  %-10s  %-10s  %-10s  %-8s\n", "N", "DOFs", "CPU (s)", "GPU (s)", "Speedup")
    println("-" ^ 50)
    for N in GRID_SIZES
        r = results[N]
        @printf("%-8d  %-10d  %-10.4f  %-10.4f  %-8.1fx\n",
                N, r["DOFs"], r["cpu_mean_s"], r["gpu_mean_s"], r["speedup"])
    end
else
    @printf("%-8s  %-10s  %-10s\n", "N", "DOFs", "CPU (s)")
    println("-" ^ 30)
    for N in GRID_SIZES
        r = results[N]
        @printf("%-8d  %-10d  %-10.4f\n", N, r["DOFs"], r["cpu_mean_s"])
    end
end
println()

# =============================================================================
# Save Results
# =============================================================================

output = Dict(
    "julia_version" => string(VERSION),
    "cuda_available" => HAS_CUDA,
    "config" => Dict(
        "grid_sizes" => GRID_SIZES,
        "D" => D,
        "n_steps" => N_STEPS,
        "n_runs" => N_RUNS
    ),
    "results" => [results[N] for N in GRID_SIZES]
)

if HAS_CUDA
    output["gpu_device"] = CUDA.name(CUDA.device())
end

output_path = joinpath(@__DIR__, "results_scaling.json")
open(output_path, "w") do f
    JSON.print(f, output, 2)
end
println("Results saved to: $output_path")

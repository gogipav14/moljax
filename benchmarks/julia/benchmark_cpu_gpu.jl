#!/usr/bin/env julia
"""
Benchmark comparing Julia CPU vs GPU for FFT-based Crank-Nicolson diffusion solver.
Equivalent to the JAX/PyTorch/NumPy benchmark in Python.

Outputs JSON results for paper table integration.
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
println("Julia FFT-CN Benchmark: CPU vs GPU")
println("=" ^ 60)
println()

# System info
println("System Information:")
println("  Julia version: ", VERSION)
println("  Threads: ", Threads.nthreads())
println("  BLAS threads: ", BLAS.get_num_threads())
if HAS_CUDA
    println("  CUDA available: true")
    println("  GPU: ", CUDA.name(CUDA.device()))
    println("  CUDA runtime: ", CUDA.runtime_version())
else
    println("  CUDA available: false")
end
println()

# =============================================================================
# CPU Implementation
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

    # Use FFTW plans for efficiency
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

# =============================================================================
# GPU Implementation
# =============================================================================

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

const N = 256
const D = 0.1
const DT = 0.001
const N_STEPS = 1000
const N_WARMUP = 3
const N_RUNS = 10

println("Benchmark Configuration:")
println("  Grid size: $N × $N")
println("  Diffusion coefficient: $D")
println("  Time step: $DT")
println("  Steps: $N_STEPS")
println("  Warmup runs: $N_WARMUP")
println("  Benchmark runs: $N_RUNS")
println()

# Create initial condition
x = range(0, 1, length=N+1)[1:end-1]
y = range(0, 1, length=N+1)[1:end-1]
X = repeat(x, 1, N)
Y = repeat(y', N, 1)
u0 = sin.(2π .* X) .* sin.(2π .* Y)

# =============================================================================
# CPU Benchmark
# =============================================================================

println("Running CPU benchmark...")

# Warmup
for _ in 1:N_WARMUP
    solve_diffusion_cpu(u0, D, DT, N_STEPS)
end

# Benchmark
cpu_times = Float64[]
for i in 1:N_RUNS
    t = @elapsed result_cpu = solve_diffusion_cpu(u0, D, DT, N_STEPS)
    push!(cpu_times, t)
    @printf("  Run %2d: %.4f s\n", i, t)
end

cpu_mean = mean(cpu_times)
cpu_std = std(cpu_times)
@printf("CPU: %.4f ± %.4f s\n\n", cpu_mean, cpu_std)

# =============================================================================
# GPU Benchmark
# =============================================================================

gpu_mean = NaN
gpu_std = NaN
gpu_times = Float64[]

if HAS_CUDA
    println("Running GPU benchmark...")

    # Warmup (includes JIT compilation of CUDA kernels)
    for _ in 1:N_WARMUP
        solve_diffusion_gpu(u0, D, DT, N_STEPS)
    end

    # Benchmark
    for i in 1:N_RUNS
        CUDA.synchronize()
        t = @elapsed begin
            result_gpu = solve_diffusion_gpu(u0, D, DT, N_STEPS)
            CUDA.synchronize()
        end
        push!(gpu_times, t)
        @printf("  Run %2d: %.4f s\n", i, t)
    end

    gpu_mean = mean(gpu_times)
    gpu_std = std(gpu_times)
    @printf("GPU: %.4f ± %.4f s\n\n", gpu_mean, gpu_std)
else
    println("GPU benchmark skipped (CUDA not available)")
    println()
end

# =============================================================================
# Results Summary
# =============================================================================

println("=" ^ 60)
println("RESULTS SUMMARY")
println("=" ^ 60)
@printf("CPU (FFTW):  %.4f ± %.4f s\n", cpu_mean, cpu_std)
if HAS_CUDA
    @printf("GPU (CUFFT): %.4f ± %.4f s\n", gpu_mean, gpu_std)
    speedup = cpu_mean / gpu_mean
    @printf("GPU Speedup: %.1fx\n", speedup)
end
println()

# =============================================================================
# Save Results
# =============================================================================

results = Dict(
    "julia_version" => string(VERSION),
    "n_threads" => Threads.nthreads(),
    "cuda_available" => HAS_CUDA,
    "config" => Dict(
        "N" => N,
        "D" => D,
        "dt" => DT,
        "n_steps" => N_STEPS,
        "n_runs" => N_RUNS
    ),
    "cpu" => Dict(
        "mean_s" => cpu_mean,
        "std_s" => cpu_std,
        "times_s" => cpu_times
    )
)

if HAS_CUDA
    results["gpu"] = Dict(
        "device" => CUDA.name(CUDA.device()),
        "mean_s" => gpu_mean,
        "std_s" => gpu_std,
        "times_s" => gpu_times
    )
    results["speedup"] = cpu_mean / gpu_mean
end

# Save to JSON
output_path = joinpath(@__DIR__, "results_cpu_gpu.json")
open(output_path, "w") do f
    JSON.print(f, results, 2)
end
println("Results saved to: $output_path")

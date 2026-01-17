"""
GPU (CUDA) version of FFT-based Crank-Nicolson solver.
Uses CUDA.jl and CUFFT for GPU-accelerated FFT.
"""
module FFTCNSolverGPU

using CUDA
using CUDA.CUFFT

export solve_diffusion_gpu, fftfreq

"""
Compute FFT sample frequencies (equivalent to numpy.fft.fftfreq).
"""
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

"""
Create Laplacian eigenvalues for periodic domain.
"""
function create_laplacian_eigenvalues(N::Int, dx::Float64)
    kx = fftfreq(N, dx) .* 2π
    ky = fftfreq(N, dx) .* 2π
    KX = repeat(kx, 1, N)
    KY = repeat(ky', N, 1)
    return -(KX.^2 .+ KY.^2)
end

"""
Solve 2D diffusion equation using FFT-Crank-Nicolson on GPU.

Arguments:
- u0: Initial condition (N×N matrix)
- D: Diffusion coefficient
- dt: Time step
- n_steps: Number of time steps

Returns: Final solution as CPU Array
"""
function solve_diffusion_gpu(u0::Matrix{Float64}, D::Float64, dt::Float64, n_steps::Int)
    N = size(u0, 1)
    dx = 1.0 / N

    # Compute eigenvalues on CPU, transfer to GPU
    lap_eig = create_laplacian_eigenvalues(N, dx)
    cn_factor = ComplexF64.((1.0 .+ 0.5 .* dt .* D .* lap_eig) ./ (1.0 .- 0.5 .* dt .* D .* lap_eig))
    cn_factor_gpu = CuArray(cn_factor)

    # Transfer initial condition to GPU as complex
    u = CuArray(ComplexF64.(u0))

    # Create in-place CUFFT plans
    plan_fft! = CUFFT.plan_fft!(u)
    plan_ifft! = CUFFT.plan_ifft!(u)

    # Time stepping
    for _ in 1:n_steps
        plan_fft! * u           # In-place FFT
        u .*= cn_factor_gpu     # Apply CN factor
        plan_ifft! * u          # In-place IFFT
    end

    CUDA.synchronize()
    return Array(real.(u))
end

end # module

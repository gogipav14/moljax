"""
FFT-based Crank-Nicolson solver for 2D diffusion equation.
Equivalent to the JAX implementation in moljax.

Supports both CPU (FFTW) and GPU (CUDA.CUFFT).
"""
module FFTCNSolver

using FFTW

export solve_diffusion_cpu, create_laplacian_eigenvalues, fftfreq

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
Solve 2D diffusion equation using FFT-Crank-Nicolson on CPU.

Arguments:
- u0: Initial condition (N×N matrix)
- D: Diffusion coefficient
- dt: Time step
- n_steps: Number of time steps
- lap_eig: Precomputed Laplacian eigenvalues (optional)

Returns: Final solution (N×N matrix)
"""
function solve_diffusion_cpu(u0::Matrix{Float64}, D::Float64, dt::Float64,
                             n_steps::Int; lap_eig=nothing)
    N = size(u0, 1)
    dx = 1.0 / N

    # Precompute Laplacian eigenvalues if not provided
    if lap_eig === nothing
        lap_eig = create_laplacian_eigenvalues(N, dx)
    end

    # Crank-Nicolson amplification factor
    cn_factor = (1.0 .+ 0.5 .* dt .* D .* lap_eig) ./ (1.0 .- 0.5 .* dt .* D .* lap_eig)

    # Work in complex space for FFT
    u = ComplexF64.(u0)

    # Create in-place FFT plans
    plan_fft! = FFTW.plan_fft!(u)
    plan_ifft! = FFTW.plan_ifft!(u)

    # Time stepping
    for _ in 1:n_steps
        plan_fft! * u          # In-place FFT
        u .*= cn_factor        # Apply CN factor in Fourier space
        plan_ifft! * u         # In-place IFFT
    end

    return real.(u)
end

end # module

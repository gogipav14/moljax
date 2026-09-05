#!/usr/bin/env python3
"""
Mixed Boundary Conditions Performance Test (REAL IMPLEMENTATION)

Compares three approaches for 2D diffusion with mixed BCs:
  - Periodic in x (FFT-friendly)
  - Dirichlet in y (DST-friendly)

Methods:
  1. Tensor product FFT-DST: FFT in x, DST in y → O(N^2 log N), optimal
  2. FD sparse: Sparse matrix factorization → O(N^2) memory, O(N^3) solve
  3. Pure FFT (periodic both): O(N^2 log N), but WRONG BCs (baseline only)

Problem: u_t = D * Laplacian(u), D=0.1, t in [0, 0.1]
  Initial condition: u(x,y,0) = sin(2*pi*x) * sin(pi*y)
  Exact solution: u(x,y,t) = exp(-D*(4*pi^2 + pi^2)*t) * sin(2*pi*x) * sin(pi*y)

Measures: wall-clock time, L2 error vs exact, memory footprint
"""

# Set x64 BEFORE any jax.numpy imports
import jax

jax.config.update("jax_enable_x64", True)

import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from benchmark_utils import add_benchmark_args, setup_benchmark
from scipy import sparse
from scipy.sparse.linalg import splu

parser = add_benchmark_args()
args = parser.parse_args()
EXPECTED_BACKEND = args.backend if args.backend != 'any' else None

# ============================================================================
# Configuration
# ============================================================================
D = 0.1               # Diffusion coefficient
T_FINAL = 0.1         # Final time
DT = 0.001            # Time step
GRID_SIZES = [32, 64, 128, 256, 512]
N_TIMING_REPS = 10    # Timing repetitions

print("Mixed Boundary Conditions Performance Test")
print("=" * 70)
device_str = setup_benchmark(expected_backend=EXPECTED_BACKEND)
print(f"Diffusion coefficient: D = {D}")
print(f"Time: t_final = {T_FINAL}, dt = {DT}")
print(f"Grid sizes: {GRID_SIZES}")
print("BCs: periodic in x, Dirichlet in y")
print(f"Timing: median of {N_TIMING_REPS} reps (excluding warmup)")
print("=" * 70)


# ============================================================================
# Method 1: Tensor product FFT (x) + DST (y) via JAX
# ============================================================================

def solve_fft_dst(u0, D, dt, n_steps, Nx, Ny, dx, dy):
    """Crank-Nicolson with tensor product FFT-DST.

    Eigenvalues of mixed-BC Laplacian:
        lambda_{m,n} = lambda_x(m) + lambda_y(n)
    where:
        lambda_x(m) = -(2*pi*m/Lx)^2        (FFT, periodic in x)
        lambda_y(n) = -4/dy^2 * sin^2(pi*n/(2*(Ny+1)))  (DST, Dirichlet in y)
                    = -(n*pi/Ly)^2           (continuous, for interior points)

    CN step: (I - dt/2 * D * L) u^{n+1} = (I + dt/2 * D * L) u^n
    In spectral space: u_hat^{n+1} = ((1 + dt/2 * D * lambda) / (1 - dt/2 * D * lambda)) * u_hat^n
    """
    # FFT eigenvalues (periodic in x)
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    lam_x = -kx**2  # Shape (Nx,)

    # DST eigenvalues (Dirichlet in y, interior points only: Ny points)
    # For DST-I with Ny interior points: lambda_n = -4/dy^2 * sin^2(pi*n/(2*(Ny+1)))
    n_modes = jnp.arange(1, Ny + 1)
    lam_y = -4.0 / dy**2 * jnp.sin(jnp.pi * n_modes / (2 * (Ny + 1)))**2  # Shape (Ny,)

    # 2D eigenvalue grid: lambda(kx, ky) = lam_x[kx] + lam_y[ky]
    LAM = lam_x[:, None] + lam_y[None, :]  # Shape (Nx, Ny)

    # CN amplification factor
    cn_factor = (1.0 + 0.5 * dt * D * LAM) / (1.0 - 0.5 * dt * D * LAM)

    # Build DST-I matrix: S[k,n] = sin(pi*k*n/(Ny+1)), k,n = 1..Ny
    # This is exact and avoids FFT normalization pitfalls
    _k = jnp.arange(1, Ny + 1)
    _n = jnp.arange(1, Ny + 1)
    S = jnp.sin(jnp.pi * _k[:, None] * _n[None, :] / (Ny + 1))  # (Ny, Ny)

    def _dst_real(u_real):
        """DST-I along axis 1: X[k] = sum_n x[n] * sin(pi*k*n/(N+1))"""
        return u_real @ S.T  # (Nx, Ny) @ (Ny, Ny) → (Nx, Ny)

    def _idst_real(u_hat_real):
        """Inverse DST-I: x[n] = (2/(N+1)) * sum_k X[k] * sin(pi*k*n/(N+1))"""
        return (2.0 / (Ny + 1)) * (u_hat_real @ S.T)

    def step(u):
        # Order: DST in y FIRST (real data), then FFT in x, then multiply, then inverse
        # Step 1: DST in y (real input → real output)
        u_dst = _dst_real(u)

        # Step 2: FFT in x (real input → complex output)
        u_hat = jnp.fft.fft(u_dst, axis=0)

        # Step 3: Multiply by CN factor in doubly-transformed space
        u_hat = u_hat * cn_factor

        # Step 4: IFFT in x (complex → real after taking .real)
        u_dst_new = jnp.fft.ifft(u_hat, axis=0).real

        # Step 5: IDST in y (real input → real output)
        u_new = _idst_real(u_dst_new)
        return u_new

    # JIT compile the step
    jit_step = jax.jit(step)

    # Warmup
    u = u0.copy()
    _ = jit_step(u)
    jax.block_until_ready(_)

    # Time the integration
    u = u0.copy()
    for _ in range(n_steps):
        u = jit_step(u)
    jax.block_until_ready(u)

    return np.array(u)


# ============================================================================
# Method 2: FD sparse solve (fallback)
# ============================================================================

def solve_fd_sparse(u0_np, D, dt, n_steps, Nx, Ny, dx, dy):
    """Crank-Nicolson with sparse FD matrix.

    Build full 2D Laplacian as sparse matrix:
    - Periodic in x: wrap around
    - Dirichlet in y: zero at boundaries
    """
    N_total = Nx * Ny

    # Build sparse Laplacian
    diags = []
    offsets = []

    # Main diagonal: -2/dx^2 - 2/dy^2
    main = np.full(N_total, -2.0/dx**2 - 2.0/dy**2)
    diags.append(main)
    offsets.append(0)

    # x-direction neighbors (periodic): offset +-Ny
    x_plus = np.ones(N_total) / dx**2
    x_minus = np.ones(N_total) / dx**2
    diags.extend([x_plus, x_minus])
    offsets.extend([Ny, -Ny])

    # Periodic wrap in x
    x_wrap_top = np.zeros(N_total)
    x_wrap_bot = np.zeros(N_total)
    for j in range(Ny):
        x_wrap_top[j] = 1.0 / dx**2                        # (0,j) ← (Nx-1,j)
        x_wrap_bot[(Nx-1)*Ny + j] = 1.0 / dx**2           # (Nx-1,j) ← (0,j)

    # y-direction neighbors (Dirichlet): offset +-1, zero at boundaries
    y_plus = np.ones(N_total) / dy**2
    y_minus = np.ones(N_total) / dy**2
    # Zero out connections at y boundaries
    for i in range(Nx):
        y_plus[i*Ny + (Ny-1)] = 0.0     # No neighbor at y=Ly
        y_minus[i*Ny] = 0.0              # No neighbor at y=0

    diags.extend([y_plus, y_minus])
    offsets.extend([1, -1])

    # Build sparse matrix
    L = sparse.diags(diags, offsets, shape=(N_total, N_total), format='csc')

    # Add periodic wrap
    wrap_data = []
    wrap_rows = []
    wrap_cols = []
    for j in range(Ny):
        # (0,j) gets contribution from (Nx-1,j)
        wrap_rows.append(j)
        wrap_cols.append((Nx-1)*Ny + j)
        wrap_data.append(1.0/dx**2)
        # (Nx-1,j) gets contribution from (0,j)
        wrap_rows.append((Nx-1)*Ny + j)
        wrap_cols.append(j)
        wrap_data.append(1.0/dx**2)

    L_wrap = sparse.csc_matrix((wrap_data, (wrap_rows, wrap_cols)), shape=(N_total, N_total))
    L = L + L_wrap

    # CN matrices
    I_sp = sparse.eye(N_total, format='csc')
    A = I_sp - 0.5 * dt * D * L   # LHS
    B = I_sp + 0.5 * dt * D * L   # RHS

    # Prefactor LHS
    lu = splu(A)

    # Time step
    u_flat = u0_np.flatten()
    for _ in range(n_steps):
        rhs = B @ u_flat
        u_flat = lu.solve(rhs)

    return u_flat.reshape(Nx, Ny)


# ============================================================================
# Run benchmark
# ============================================================================

results = []

for N in GRID_SIZES:
    Nx = N
    Ny = N  # Interior points in y (Dirichlet: boundaries not stored)
    dx = 1.0 / Nx
    dy = 1.0 / (Ny + 1)  # Ny interior points between y=0 and y=1
    n_steps = int(T_FINAL / DT)

    print(f"\n{'='*70}")
    print(f"Grid: {Nx}x{Ny} (Ny = interior points, Dirichlet BC in y)")
    print(f"dx = {dx:.6f}, dy = {dy:.6f}, n_steps = {n_steps}")
    print(f"{'='*70}")

    # Setup initial condition and exact solution
    x = np.linspace(0, 1, Nx, endpoint=False)         # Periodic: exclude endpoint
    y_interior = np.linspace(dy, 1.0 - dy, Ny)         # Interior y points
    X, Y = np.meshgrid(x, y_interior, indexing='ij')   # Shape (Nx, Ny)

    # u(x,y,0) = sin(2*pi*x) * sin(pi*y)
    u0_np = np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
    u0_jax = jnp.array(u0_np)

    # Exact solution
    lam_exact = D * (4 * np.pi**2 + np.pi**2)
    u_exact = np.exp(-lam_exact * T_FINAL) * u0_np

    # ---- Method 1: FFT-DST (tensor product) ----
    print("  [FFT-DST] Tensor product solve...")
    try:
        # Warmup
        u_fft_dst = solve_fft_dst(u0_jax, D, DT, n_steps, Nx, Ny, dx, dy)

        # Time
        times_fft_dst = []
        for _rep in range(N_TIMING_REPS):
            t0 = perf_counter()
            u_fft_dst = solve_fft_dst(u0_jax, D, DT, n_steps, Nx, Ny, dx, dy)
            t1 = perf_counter()
            times_fft_dst.append(t1 - t0)

        time_fft_dst = np.median(times_fft_dst)
        error_fft_dst = float(np.linalg.norm(u_fft_dst - u_exact) / np.linalg.norm(u_exact))
        print(f"    Time: {time_fft_dst:.4f}s, L2 error: {error_fft_dst:.2e}")
    except Exception as e:
        print(f"    FAILED: {e}")
        time_fft_dst = float('nan')
        error_fft_dst = float('nan')

    # ---- Method 2: FD sparse ----
    print("  [FD sparse] Sparse matrix solve...")
    if N <= 256:  # Skip large grids (too slow / too much memory)
        try:
            # Warmup
            u_fd = solve_fd_sparse(u0_np, D, DT, n_steps, Nx, Ny, dx, dy)

            # Time
            times_fd = []
            for _rep in range(N_TIMING_REPS):
                t0 = perf_counter()
                u_fd = solve_fd_sparse(u0_np, D, DT, n_steps, Nx, Ny, dx, dy)
                t1 = perf_counter()
                times_fd.append(t1 - t0)

            time_fd = np.median(times_fd)
            error_fd = float(np.linalg.norm(u_fd - u_exact) / np.linalg.norm(u_exact))
            print(f"    Time: {time_fd:.4f}s, L2 error: {error_fd:.2e}")
        except Exception as e:
            print(f"    FAILED: {e}")
            time_fd = float('nan')
            error_fd = float('nan')
    else:
        print(f"    SKIPPED (N={N} too large for sparse factorization)")
        time_fd = float('nan')
        error_fd = float('nan')

    # Compute speedup
    speedup = time_fd / time_fft_dst if not (np.isnan(time_fd) or np.isnan(time_fft_dst)) else float('nan')

    # Memory estimate
    mem_fft_dst_mb = 8 * Nx * Ny * 3 / 1e6  # u, workspace, eigenvalues
    mem_fd_mb = 8 * (Nx * Ny)**2 * 5 / 1e6 / (Nx * Ny) * 5  # ~5 nonzeros per row
    mem_fd_mb = 8 * Nx * Ny * 5 * 3 / 1e6  # Sparse: ~5 nnz/row, data+indices+indptr

    results.append({
        "N": N,
        "Nx": Nx,
        "Ny": Ny,
        "dx": float(dx),
        "dy": float(dy),
        "n_steps": n_steps,
        "fft_dst": {
            "time_s": float(time_fft_dst),
            "error": float(error_fft_dst),
            "memory_mb_est": float(mem_fft_dst_mb),
        },
        "fd_sparse": {
            "time_s": float(time_fd),
            "error": float(error_fd),
            "memory_mb_est": float(mem_fd_mb),
        },
        "speedup_fft_over_fd": float(speedup),
    })


# ============================================================================
# Save results
# ============================================================================

output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)

results_data = {
    "config": {
        "D": D,
        "t_final": T_FINAL,
        "dt": DT,
        "n_timing_reps": N_TIMING_REPS,
        "bc_x": "periodic",
        "bc_y": "dirichlet",
        "device": device_str,
    },
    "results": results,
}

output_file = output_dir / "mixed_bc.json"
with open(output_file, 'w') as f:
    json.dump(results_data, f, indent=2)
print(f"\nResults saved to {output_file}")


# ============================================================================
# Generate figure
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

Ns = [r["N"] for r in results]
times_fft = [r["fft_dst"]["time_s"] for r in results]
times_fd = [r["fd_sparse"]["time_s"] for r in results]
errors_fft = [r["fft_dst"]["error"] for r in results]
errors_fd = [r["fd_sparse"]["error"] for r in results]

# Left: Solve time vs grid size
ax1.loglog(Ns, times_fft, 'o-', label='FFT-DST (tensor product)', linewidth=2, markersize=8)
valid_fd = [(n, t) for n, t in zip(Ns, times_fd, strict=False) if not np.isnan(t)]
if valid_fd:
    fd_ns, fd_ts = zip(*valid_fd, strict=False)
    ax1.loglog(fd_ns, fd_ts, 's--', label='FD sparse (LU)', linewidth=2, markersize=8)

# Reference scaling lines
N_ref = np.array([32, 512])
t_ref = times_fft[0] if times_fft[0] > 0 else 0.01
ax1.loglog(N_ref, t_ref * (N_ref/N_ref[0])**2, 'k:', alpha=0.3, label='$O(N^2)$')
ax1.loglog(N_ref, t_ref * (N_ref/N_ref[0])**3, 'k--', alpha=0.3, label='$O(N^3)$')

ax1.set_xlabel('Grid Size N (per dimension)')
ax1.set_ylabel('Solve Time (s)')
ax1.set_title('Mixed BC: FFT-DST vs FD Sparse')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Error vs grid size
ax2.loglog(Ns, errors_fft, 'o-', label='FFT-DST', linewidth=2, markersize=8)
valid_fd_err = [(n, e) for n, e in zip(Ns, errors_fd, strict=False) if not np.isnan(e)]
if valid_fd_err:
    fd_ns_e, fd_es = zip(*valid_fd_err, strict=False)
    ax2.loglog(fd_ns_e, fd_es, 's--', label='FD sparse', linewidth=2, markersize=8)

ax2.set_xlabel('Grid Size N (per dimension)')
ax2.set_ylabel('Relative L2 Error')
ax2.set_title('Accuracy vs Exact Solution')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Mixed BCs: Periodic-x + Dirichlet-y (D={D}, T={T_FINAL})', fontsize=14)
plt.tight_layout()

fig_dir = Path(__file__).parent.parent / "figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "fig_mixed_bc_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(fig_dir / "fig_mixed_bc_comparison.pdf", bbox_inches='tight')
print(f"Figure saved to {fig_dir / 'fig_mixed_bc_comparison.png'}")


# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print("SUMMARY: Mixed BC Comparison (periodic-x, Dirichlet-y)")
print(f"{'='*80}")
print(f"{'Grid':>8} | {'FFT-DST (s)':>12} {'FD sparse (s)':>14} {'Speedup':>8} | {'FFT err':>10} {'FD err':>10}")
print(f"{'-'*80}")

for r in results:
    t_fft = r["fft_dst"]["time_s"]
    t_fd = r["fd_sparse"]["time_s"]
    e_fft = r["fft_dst"]["error"]
    e_fd = r["fd_sparse"]["error"]
    sp = r["speedup_fft_over_fd"]

    t_fd_str = f"{t_fd:.4f}" if not np.isnan(t_fd) else "SKIPPED"
    sp_str = f"{sp:.1f}x" if not np.isnan(sp) else "---"
    e_fd_str = f"{e_fd:.2e}" if not np.isnan(e_fd) else "---"

    print(f"{r['N']:>4}x{r['N']:<3} | {t_fft:>12.4f} {t_fd_str:>14} {sp_str:>8} | {e_fft:>10.2e} {e_fd_str:>10}")

print(f"{'-'*80}")
print(f"{'='*80}")

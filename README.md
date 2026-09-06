# moljax

**GPU-Portable Adaptive Method-of-Lines in JAX via AD-JVP Newton-Krylov and Spectral/FFT Operators**

[![CI](https://github.com/gogipav14/moljax/actions/workflows/ci.yml/badge.svg)](https://github.com/gogipav14/moljax/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2026.110205-blue.svg)](https://doi.org/10.1016/j.cpc.2026.110205)

## Paper

This repository accompanies:

> **moljax: GPU-accelerated method of lines for stiff reaction-diffusion PDEs with FFT preconditioning**
> Gorgi Pavlov. *Computer Physics Communications* **326** (2026) 110205.
> [doi:10.1016/j.cpc.2026.110205](https://doi.org/10.1016/j.cpc.2026.110205)

The published results correspond to release `v1.0.0` (commit `25cd9a3`).
To reproduce them:

```bash
conda env create -f environment.yml && conda activate moljax
bash benchmarks/run_all.sh          # ~30 min on RTX 5060
python benchmarks/plot_main_figures.py
```

See [REPRODUCE.md](REPRODUCE.md) for details, including the CPU-only path.
If you just want to *use* the library rather than reproduce the paper, use
`environment-current.yml`, which tracks the current dependency stack.

---

`moljax` is a JAX-native method-of-lines framework for solving stiff PDEs with:

- **JIT-compiled adaptive time stepping** with accept/reject control flow on GPU/TPU
- **Matrix-free Newton-Krylov** using automatic differentiation for machine-precision Jacobian-vector products
- **FFT/DST/DCT spectral operators** enabling O(N log N) implicit solves and exponential integrators
- **Physics-aware preconditioning** reducing Krylov iterations by 10-100×

## Installation

moljax is not on PyPI. Install from a clone:

```bash
git clone https://github.com/gogipav14/moljax.git
cd moljax
pip install -e ".[dev,viz]"    # or: pip install -e .   (library only)
```

Requires Python 3.10+ and JAX 0.4.20+ (`pyproject.toml`); for the pinned
stacks see `environment.yml` (paper) and `environment-current.yml`.

**float64.** The model factories default to `float64`, and JAX arrays are
`float32` unless 64-bit mode is on. Enable it before the first array is
created, either in code or with `JAX_ENABLE_X64=1` in the environment:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Every snippet below assumes this has run.

## Quick Start

### Gray-Scott with Crank-Nicolson and the FFT preconditioner

```python
import jax.numpy as jnp
from moljax.core import Grid2D, IntegratorType, adaptive_integrate, create_fft_cache
from moljax.core.model import create_gray_scott_model
from moljax.core.preconditioners import create_gray_scott_fft_preconditioner
from moljax.core.utils import get_interior

grid = Grid2D.uniform(64, 64, 0.0, 2.5, 0.0, 2.5)
model = create_gray_scott_model(grid, Du=0.16, Dv=0.08, F=0.04, k=0.06)

# States carry one ghost layer per side: shape (ny + 2, nx + 2).
X, Y = grid.meshgrid(include_ghost=True)
square = (jnp.abs(X - 1.25) < 0.25) & (jnp.abs(Y - 1.25) < 0.25)
y0 = {"u": jnp.where(square, 0.5, 1.0), "v": jnp.where(square, 0.25, 0.0)}

precond = create_gray_scott_fft_preconditioner(create_fft_cache(grid))
result = adaptive_integrate(model, y0, t0=0.0, t_end=20.0, dt0=0.5,
                            method=IntegratorType.CN, preconditioner=precond)
u = get_interior(result.y_final["u"], grid)   # (64, 64), ghosts stripped
print(float(result.t_final), int(result.n_accepted), int(result.status))
```

`adaptive_integrate` runs the whole accept/reject loop inside one
`lax.while_loop`; `result.status == 0` means it reached `t_end`.
`IntegratorType` also offers `EULER`, `SSPRK3`, `RK4`, `BE` and `BDF2`.

### IMEX Strang splitting

Diffusion is solved implicitly in Fourier space, the reaction explicitly,
so the step is not limited by the diffusion CFL:

```python
from moljax.core import adaptive_integrate_imex
from moljax.core.model import create_gray_scott_periodic_fft

model_fft, fft_cache, diffusivities = create_gray_scott_periodic_fft(
    grid, Du=0.16, Dv=0.08, F=0.04, k=0.06)
result = adaptive_integrate_imex(model_fft, y0, 0.0, 20.0, 0.5,
                                 fft_cache, diffusivities, use_strang=True)
```

### ETDRK4 on a periodic 1D grid

Exponential integrators take the FFT-diagonalized linear operator exactly
and treat only the nonlinear term with a Runge-Kutta scheme:

```python
from moljax.core import DiffusionOperator, Grid1D, etd_integrate

grid1d = Grid1D.uniform(128, 0.0, 1.0)
x = grid1d.x_coords()                          # interior, cell-centered coordinates
u0 = {"u": 0.5 + 0.3 * jnp.sin(2 * jnp.pi * x)}
op = DiffusionOperator(grid1d, D=0.01)         # D * Laplacian, diagonalized by the FFT

def reaction(state, t):
    u = state["u"]
    return {"u": 2.0 * u * (1.0 - u)}

t_hist, states = etd_integrate(u0, (0.0, 1.0), dt=0.05, linear_ops={"u": op},
                               nonlinear_rhs=reaction, method="etdrk4")
u_final = states[-1]["u"]                      # (128,)
```

`method` may also be `"etd1"` or `"etd2"`. The tubular reactor of the paper
(Danckwerts boundaries, finite differences) lives in
`benchmarks/benchmark_tubular_reactor.py`; the core package provides
periodic, Dirichlet and Neumann boundaries.

## Features

### Spatial Operators

| Operator | BC Support | Complexity | Accuracy |
|----------|------------|------------|----------|
| FFT Laplacian | Periodic | O(N log N) | Spectral |
| DST Laplacian | Dirichlet | O(N log N) | O(Δx²) |
| DCT Laplacian | Neumann | O(N log N) | O(Δx²) |
| FD Stencils | Any | O(N) | O(Δx²) to O(Δx⁶) |

### Time Integrators

| Method | Type | Order | Best For |
|--------|------|-------|----------|
| RK4 | Explicit | 4 | Non-stiff, CFL-limited |
| SSPRK3 | Explicit | 3 | Hyperbolic, TVD |
| Backward Euler | Implicit | 1 | Very stiff |
| Crank-Nicolson | Implicit | 2 | Diffusion-dominated |
| BDF2 | Implicit | 2 | Stiff ODEs |
| IMEX-Euler | Splitting | 1 | Reaction-diffusion |
| IMEX-Strang | Splitting | 2 | Reaction-diffusion |
| ETDRK4 | Exponential | 4 | Stiff linear + nonlinear |

### Adaptive Time Stepping

Explicit methods are limited by a CFL bound (`CFLParams`); every method
uses a PI controller on the embedded error estimate (`PIDParams`):

```python
from moljax.core.dt_policy import CFLParams, PIDParams

cfl_params = CFLParams(safety=0.9, cfl_diffusion=0.25)   # explicit methods only
pid_params = PIDParams(atol=1e-6, rtol=1e-4, kI=0.7, kP=0.4, dt_max=1.0)
result = adaptive_integrate(model, y0, 0.0, 1.0, 1e-3, method=IntegratorType.RK4,
                            cfl_params=cfl_params, pid_params=pid_params)
```

### Newton-Krylov Solver

The implicit steps call a matrix-free Newton-Krylov solver (GMRES on
AD-computed Jacobian-vector products). It is usable on its own; the residual
maps a state dict to a state dict:

```python
from moljax.core import newton_krylov_solve
from moljax.core.newton_krylov import NKParams

def residual(x):
    return {"u": x["u"] ** 3 - 1.0}

x0 = {"u": jnp.full((grid.ny_total, grid.nx_total), 0.5)}
nk = newton_krylov_solve(residual, x0, grid, params={},
                         nk_params=NKParams(max_newton_iters=20, newton_tol=1e-10))
print(bool(nk.stats.converged), int(nk.stats.newton_iters))
```

Pass `preconditioner=` (for example `create_gray_scott_fft_preconditioner`
above) to precondition the GMRES solve.

## Benchmarks

Run the paper benchmarks:

```bash
cd benchmarks
python benchmark_method_comparison.py    # RK4 vs CN vs IMEX reactor comparison
python generate_convergence_figure.py    # Spectral accuracy verification
python benchmark_numpy_jax_pytorch.py    # Cross-framework validation
```

The scripts expect a GPU by default; `--backend any` runs them on whatever
backend JAX has. `bash benchmarks/run_all.sh` runs the whole suite and
[REPRODUCE.md](REPRODUCE.md) covers the SISC extension.

### Julia Benchmarks

For cross-language reproducibility (NumPy, JAX, Julia yield identical errors):

```bash
cd benchmarks/julia
julia benchmark_diffeq.jl      # Compare with DifferentialEquations.jl
julia fft_cn_solver.jl         # FFT-CN reference implementation
julia fft_cn_solver_gpu.jl     # GPU version (requires CUDA.jl)
```

## Examples

| Example | Description |
|---------|-------------|
| `examples/gray_scott_2d.py` | Turing patterns with explicit/implicit/IMEX comparison |
| `examples/advdiff_multispecies.py` | Multi-species transport with upwind advection |
| `examples/acoustics_1d.py` | Coupled wave equations with SSPRK3 |

Run them from the repository root with the package installed, for example
`python examples/gray_scott_2d.py`; each writes to `./output/`.

## Citation

If you use `moljax` in your research, please cite:

```bibtex
@article{pavlov2026moljax,
  title   = {moljax: {GPU}-accelerated method of lines for stiff reaction-diffusion {PDEs} with {FFT} preconditioning},
  author  = {Pavlov, Gorgi},
  journal = {Computer Physics Communications},
  volume  = {326},
  pages   = {110205},
  year    = {2026},
  doi     = {10.1016/j.cpc.2026.110205}
}
```

[CITATION.cff](CITATION.cff) carries the same record plus the software
citation.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Built on [JAX](https://github.com/google/jax) by Google
- Inspired by [DifferentialEquations.jl](https://diffeq.sciml.ai/) and [Diffrax](https://github.com/patrick-kidger/diffrax)

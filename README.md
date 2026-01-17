# moljax

**GPU-Portable Adaptive Method-of-Lines in JAX via AD-JVP Newton-Krylov and Spectral/FFT Operators**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.25+-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`moljax` is a JAX-native method-of-lines framework for solving stiff PDEs with:

- **JIT-compiled adaptive time stepping** with accept/reject control flow on GPU/TPU
- **Matrix-free Newton-Krylov** using automatic differentiation for machine-precision Jacobian-vector products
- **FFT/DST/DCT spectral operators** enabling O(N log N) implicit solves and exponential integrators
- **Physics-aware preconditioning** reducing Krylov iterations by 10-100×

## Installation

```bash
pip install moljax
```

Or install from source:
```bash
git clone https://github.com/gogipav14/moljax.git
cd moljax
pip install -e .
```

## Quick Start

### 2D Diffusion with FFT-CN (Spectral Accuracy)

```python
import jax.numpy as jnp
from moljax.core import Grid2D, fft_solvers

# Create grid
grid = Grid2D(nx=128, ny=128, Lx=1.0, Ly=1.0, periodic=True)

# Initial condition: sin(2πx)sin(2πy)
x, y = grid.x, grid.y
u0 = jnp.sin(2*jnp.pi*x) * jnp.sin(2*jnp.pi*y)

# FFT-CN solver for diffusion
D = 0.1  # Diffusivity
dt = 0.01
u = fft_solvers.fft_cn_step(u0, D, grid.dx, dt)
```

### Gray-Scott Reaction-Diffusion (Stiff)

```python
from moljax.core import MOLModel, IntegratorType, adaptive_integrate
from moljax.core.model import gray_scott_factory

# Create Gray-Scott model
model = gray_scott_factory(
    nx=256, ny=256,
    Du=2e-5, Dv=1e-5,
    F=0.035, k=0.065
)

# Solve with IMEX (diffusion implicit, reaction explicit)
result = adaptive_integrate(
    model,
    t_span=(0, 10000),
    integrator=IntegratorType.IMEX_STRANG,
    rtol=1e-5, atol=1e-7
)
```

### Tubular Reactor with Danckwerts BC

```python
from moljax.core import Grid1D, operators, stepping

# 1D reactor grid
grid = Grid1D(n=128, L=1.0, bc_type='danckwerts')

# Axial dispersion model: ∂c/∂t = (1/Pe)∂²c/∂z² - ∂c/∂z - Da·c
Pe, Da = 50, 2  # Péclet, Damköhler numbers

# FD operators for non-periodic BC
D2 = operators.laplacian_1d(grid, order=2)
D1 = operators.gradient_1d(grid, upwind=True)

# Time integration
c = stepping.cn_step(c0, dt, lambda c: (1/Pe)*D2@c - D1@c - Da*c)
```

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

```python
from moljax.core.dt_policy import CFLParams, PIDParams

# CFL-based for explicit methods
cfl_params = CFLParams(cfl_target=0.8, safety_factor=0.9)

# PID controller for implicit methods
pid_params = PIDParams(
    rtol=1e-6, atol=1e-8,
    ki=0.25, kp=0.075, kd=0.01  # Gustafsson PI.4.2 controller
)
```

### Newton-Krylov Solver

```python
from moljax.core.newton_krylov import newton_krylov_solve
from moljax.core.preconditioners import DiffusionPreconditioner

# Matrix-free implicit solve with FFT preconditioner
precond = DiffusionPreconditioner(D, grid.dx, dt)
u_new = newton_krylov_solve(
    residual_fn,
    u_guess,
    preconditioner=precond,
    gmres_restart=30,
    rtol=1e-8
)
```

## Benchmarks

Run the paper benchmarks:

```bash
cd benchmarks
python benchmark_method_comparison.py    # RK4 vs CN vs IMEX reactor comparison
python generate_convergence_figure.py    # Spectral accuracy verification
python benchmark_numpy_jax_pytorch.py    # Cross-framework validation
```

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

## Citation

If you use `moljax` in your research, please cite:

```bibtex
@article{moljax2025,
  title={moljax: GPU-Portable Adaptive Method-of-Lines in JAX via AD-JVP Newton--Krylov and Spectral/FFT Operators},
  author={Pavlov, Gorgi},
  journal={Computers \& Chemical Engineering},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Built on [JAX](https://github.com/google/jax) by Google
- Inspired by [DifferentialEquations.jl](https://diffeq.sciml.ai/) and [Diffrax](https://github.com/patrick-kidger/diffrax)

# Reproducing moljax Benchmark Results

This document provides instructions for reproducing the benchmarks in the moljax paper.

## System Requirements

### Tested Configuration (Paper Results)
- **GPU**: NVIDIA GeForce RTX 5060 (8 GB VRAM)
- **CPU**: Intel Core Ultra 5 225F
- **RAM**: 32 GB
- **Python**: 3.12
- **JAX**: 0.4.35 with CUDA 12.x
- **OS**: Ubuntu 22.04 (WSL2)

### Minimum Requirements
- NVIDIA GPU with CUDA support (8+ GB VRAM recommended)
- Python 3.10+
- JAX with GPU support

## Installation

```bash
# Clone repository
git clone https://github.com/gogipav14/moljax.git
cd moljax

# Option 1: Using conda (recommended)
conda env create -f environment.yml
conda activate moljax

# Option 2: Using pip
python -m venv venv
source venv/bin/activate
pip install jax[cuda12] numpy scipy matplotlib diffrax

# Install moljax
pip install -e .
```

## Running Benchmarks

All benchmark scripts are in `benchmarks/`.

```bash
cd benchmarks

# Run all benchmarks (may take 30+ minutes)
python benchmark_vs_scipy.py      # Table 5: SciPy comparison
python benchmark_scaling.py       # Figure: CPU/GPU scaling
python benchmark_gray_scott.py    # Table 6: Gray-Scott methods
python benchmark_gmres_sweep.py   # Table: GMRES iterations
python benchmark_jit_speedup.py   # Figure: JIT speedup
python benchmark_tubular_reactor.py  # Reactor case study
python benchmark_method_comparison.py  # Method comparison
python benchmark_solver_comparison.py  # Solver comparison

# Generate paper figures
python generate_paper_figures.py
```

## Expected Results

Results will vary depending on hardware. Expected ranges below are for RTX 4090 GPU.

### Table 5: SciPy Comparison (128x128 grid, diffusion)

| Method | Expected Time (s) | Expected Range |
|--------|-------------------|----------------|
| JAX FFT-CN (GPU) | 0.17 | 0.15 - 0.25 |
| NumPy FFT-CN (CPU) | 0.21 | 0.18 - 0.30 |
| SciPy RK45 | 20.8 | 18 - 25 |
| SciPy Radau | 649 | 500 - 800 |

**Key metric**: JAX FFT-CN speedup over SciPy Radau should be ~3000-4000x.

### Table 6: Gray-Scott (256x256 grid, t=10000)

| Method | Expected Time (s) | Expected Range |
|--------|-------------------|----------------|
| RK4 (explicit, adaptive) | 100 | 80 - 150 |
| CN (Newton-Krylov) | 22 | 18 - 30 |
| IMEX-Strang | 6.0 | 4 - 10 |
| ETDRK4 | 10.6 | 8 - 15 |

**Key metric**: IMEX-Strang should be 15-20x faster than RK4.

### CPU/GPU Scaling

| Grid | GPU (s) | Speedup vs CPU |
|------|---------|----------------|
| 64x64 | 0.09 | ~1x |
| 128x128 | 0.10 | ~3x |
| 256x256 | 0.22 | ~5x |
| 512x512 | 0.61 | ~11x |
| 1024x1024 | 2.5 | ~18x |

**Key metric**: GPU speedup should increase with problem size, reaching 15-20x at 1024x1024.

### GMRES Iteration Reduction (FFT Preconditioning)

| Stiffness (σ) | No Precond | FFT Precond | Reduction |
|---------------|------------|-------------|-----------|
| 1 | ~40 iters | ~3 iters | ~13x |
| 10 | ~130 iters | ~3 iters | ~40x |
| 100 | ~400 iters | ~4 iters | ~100x |
| 1000 | ~1300 iters | ~5 iters | ~260x |

**Key metric**: FFT preconditioning should reduce GMRES iterations by 10-100x depending on stiffness.

## Troubleshooting

### Out of Memory
- Reduce grid size (e.g., 128x128 instead of 256x256)
- Ensure no other GPU processes running
- Set `JAX_PLATFORM_NAME=cpu` to run on CPU

### Slow Performance
- Verify GPU is being used: `python -c "import jax; print(jax.devices())"`
- Check CUDA version compatibility with JAX
- First run includes JIT compilation; subsequent runs will be faster

### Results Don't Match
- Hardware differences can cause 2-5x variation in absolute times
- **Speedup ratios** (e.g., IMEX vs RK4) should be consistent within ~20%
- Verify all methods reach same final error (check JSON results)

## Output Files

Results are saved to `benchmarks/results/`:
- `scipy_comparison.json` - Table 5 data
- `gray_scott.json` - Table 6 data
- `scaling.json` - CPU/GPU scaling data
- `gmres_sweep.json` - GMRES iteration data
- `reactor_results.json` - Reactor case study

Figures are saved to `figures/` in PDF and PNG format.

## Verification

After running benchmarks, verify consistency:

```bash
# Check that all JSON results exist
ls benchmarks/results/*.json

# Check key metrics match paper claims
python -c "
import json
from pathlib import Path

results_dir = Path('benchmarks/results')

# Check SciPy comparison
with open(results_dir / 'scipy_comparison.json') as f:
    scipy = json.load(f)
    speedup = scipy['results']['scipy_radau']['mean_s'] / scipy['results']['jax_fft_cn']['mean_s']
    print(f'SciPy Radau / JAX FFT-CN speedup: {speedup:.0f}x')

# Check Gray-Scott
with open(results_dir / 'gray_scott.json') as f:
    gs = json.load(f)
    methods = {m['name']: m['time_s'] for m in gs['methods']}
    print(f'IMEX-Strang / RK4 speedup: {methods[\"RK4 (explicit)\"]/methods[\"IMEX-Strang\"]:.1f}x')
"
```

## SISC Extended Benchmark Suite

The SISC resubmission includes 14 additional benchmarks for comprehensive coverage.

### Running the Full SISC Suite

```bash
cd benchmarks

# Run all SISC benchmarks (30-60 minutes)
./run_sisc_suite.sh

# Or run quick subset for testing
./run_sisc_suite.sh --quick
```

### SISC Benchmark Descriptions

| Script | Description | Purpose |
|--------|-------------|---------|
| `bench_iter_vs_grid.py` | GMRES iterations vs grid size (64²-1024²) | Grid scaling validation |
| `bench_iter_vs_dim.py` | Iterations vs dimension (1D/2D/3D at ~1M DOF) | Dimensionality effects |
| `bench_newton_policy_ablation.py` | Newton failure policies (terminate/÷2/÷4/÷8) | Failure handling |
| `bench_precond_variants.py` | Preconditioner comparison (None/Jacobi/FFT) | Ablation study |
| `bench_bc_matrix.py` | FFT/DST/DCT for different BCs | BC coverage |
| `bench_presmooth_rannacher.py` | Rannacher startup for nonsmooth IC | CN oscillation damping |
| `bench_jvp_vs_fd_sweep.py` | AD-JVP vs FD-JVP accuracy (ε sweep) | AD advantage |
| `bench_adjoint_grad_sanity.py` | AD gradient validation vs FD | Sensitivity verification |
| `bench_precision_fp32_fp64.py` | FP32 vs FP64 work-precision | Precision guidance |
| `bench_gpu_memory_scaling.py` | Memory vs grid size and GMRES restart | Memory limits |
| `bench_3d_feasibility.py` | 3D diffusion (32³-128³) | 3D capabilities |
| `bench_imex_vs_fullimplicit_map.py` | Method selection regime map | Method guidance |

### Expected SISC Results

#### E1: GMRES Iterations vs Grid Size (FFT preconditioned)
| Grid | σ=1 | σ=10 | σ=100 |
|------|-----|------|-------|
| 64² | 3-4 | 4-5 | 5-7 |
| 256² | 3-4 | 4-5 | 6-8 |
| 1024² | 4-5 | 5-6 | 7-10 |

**Key finding**: Iterations remain nearly constant across grid sizes (preempts "works only at small grids" criticism).

#### E9: JVP vs FD Accuracy
- Optimal FD epsilon: ~1e-7 (U-shaped error curve)
- AD-JVP: machine precision (~1e-15)
- FD-JVP stagnation: poor epsilon choices cause GMRES stagnation

#### E11: FP32 vs FP64
- FP32 error floor: ~1e-7
- FP32 speedup: 1.5-2x over FP64 (consumer GPU)
- Use FP64 when accuracy < 1e-6 required

### SISC Output Files

Results saved to `benchmarks/results/`:
- `iter_vs_grid.json` - Grid scaling data
- `iter_vs_dim.json` - Dimensional comparison
- `newton_policy_ablation.json` - Failure handling
- `precond_variants.json` - Preconditioner comparison
- `bc_matrix.json` - BC transform timing
- `rannacher_startup.json` - Oscillation damping
- `jvp_vs_fd_sweep.json` - AD vs FD accuracy
- `adjoint_grad_sanity.json` - Gradient validation
- `precision_fp32_fp64.json` - Precision comparison
- `gpu_memory_scaling.json` - Memory analysis
- `3d_feasibility.json` - 3D capability limits
- `imex_vs_fullimplicit_map.json` - Method regime map

Figures saved to `figures/fig_*.pdf`.

## Contact

For questions about reproducing results, please open an issue on GitHub.

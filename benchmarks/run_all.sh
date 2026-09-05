#!/bin/bash
#
# run_all.sh - reproduce every benchmark table in the moljax paper.
#
#   Pavlov, G. "moljax: GPU-accelerated method of lines for stiff
#   reaction-diffusion PDEs with FFT preconditioning."
#   Computer Physics Communications 326 (2026) 110205.
#   doi:10.1016/j.cpc.2026.110205
#
# Usage:
#   bash benchmarks/run_all.sh                 # full suite
#   bash benchmarks/run_all.sh --backend any   # on CPU, or whatever JAX has
#   SKIP_SLOW=1 bash benchmarks/run_all.sh     # omit the multi-hour stages
#   PYTHON=python3.12 bash benchmarks/run_all.sh
#
# --backend gpu|cpu|any (default gpu) is forwarded to every script. The
# scripts check that JAX is running on the expected backend and abort
# otherwise; "any" skips that check. Scripts without argument parsing
# ignore the flag.
#
# Results are written to benchmarks/results/*.json. Figures are produced
# separately by benchmarks/plot_main_figures.py.
#
# Reference runtime on the paper's hardware (RTX 5060, float64):
#   default suite   ~30 min
#   with SKIP_SLOW  ~10 min
# CPU-only reproduction (--backend any) takes roughly 3-4 hours; the
# GPU-only stages (the FFT baselines, the JIT x device factorial and the
# OFAT study) fail without a GPU and the script continues past them.
#
# Every stage is run even if an earlier one fails; the script exits
# nonzero at the end if any stage failed, and prints a summary.

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
SKIP_SLOW="${SKIP_SLOW:-0}"
BACKEND="${BACKEND:-gpu}"

while [ $# -gt 0 ]; do
    case "$1" in
        --backend)   BACKEND="$2"; shift 2 ;;
        --backend=*) BACKEND="${1#--backend=}"; shift ;;
        *)
            echo "run_all.sh: unknown argument: $1" >&2
            echo "usage: bash benchmarks/run_all.sh [--backend gpu|cpu|any]" >&2
            exit 2 ;;
    esac
done

case "$BACKEND" in
    gpu|cpu|any) ;;
    *) echo "run_all.sh: --backend must be gpu, cpu or any (got '$BACKEND')" >&2; exit 2 ;;
esac

FAILED=()
PASSED=()
SKIPPED=()

run() {
    local label="$1"; shift
    local script="$1"; shift

    echo ""
    echo "=================================================================="
    echo ">>> ${label}"
    echo "    ${script} --backend ${BACKEND}"
    echo "=================================================================="
    local start
    start=$(date +%s)
    if "$PYTHON" "$script" --backend "$BACKEND" "$@"; then
        local elapsed=$(( $(date +%s) - start ))
        echo "--- OK (${elapsed}s): ${label}"
        PASSED+=("$label")
    else
        local elapsed=$(( $(date +%s) - start ))
        echo "--- FAILED (${elapsed}s): ${label}" >&2
        FAILED+=("$label")
    fi
}

run_slow() {
    local label="$1"
    if [ "$SKIP_SLOW" = "1" ]; then
        echo ""
        echo ">>> SKIPPED (SKIP_SLOW=1): ${label}"
        SKIPPED+=("$label")
        return
    fi
    run "$@"
}

echo "=================================================================="
echo "moljax benchmark suite"
echo "=================================================================="
echo "Started:  $(date)"
echo "Python:   $($PYTHON --version 2>&1)"
echo "Requested backend: $BACKEND"
"$PYTHON" - <<'EOF'
import jax
print(f"JAX:      {jax.__version__}")
print(f"Devices:  {jax.devices()}")
print(f"Backend:  {jax.default_backend()}")
EOF

# ------------------------------------------------------------------
# Controlled kernel and backend studies (paper Section 5.1)
# ------------------------------------------------------------------
run "GPU FFT baseline: CuPy / nvmath / JAX"   benchmark_fft_3lib.py
run "GPU FFT: CuPy vs JAX (fft2 + rfft2)"     benchmark_cupy_fft.py
run "JIT x device factorial"                  benchmark_jit_factorial.py
run "JIT speedup"                             benchmark_jit_speedup.py

# ------------------------------------------------------------------
# Controlled solver-component ablations (paper Sections 5.4-5.7)
# ------------------------------------------------------------------
run "OFAT single-factor study"                benchmark_ofat.py
run "Component ablation"                      benchmark_ablation.py
run "FFT diagonalization vs sparse direct"    benchmark_fft_vs_sparse.py
run "GMRES / FFT preconditioner sweep"        benchmark_gmres_sweep.py
run "Scaling with problem size"               benchmark_scaling.py
run "SciPy comparison"                        benchmark_vs_scipy.py
run "Solver comparison"                       benchmark_solver_comparison.py

# ------------------------------------------------------------------
# Tubular reactor (paper Section 5.2)
# ------------------------------------------------------------------
run "Tubular reactor Pe-Da sweep"             benchmark_tubular_reactor.py
run "Reactor method comparison"               benchmark_method_comparison.py
run "Reactor vs Diffrax"                      benchmark_reactor_vs_diffrax.py

# ------------------------------------------------------------------
# Stiff reaction-diffusion suite (paper Section 5.6)
# ------------------------------------------------------------------
run      "Gray-Scott"                         benchmark_gray_scott.py
run_slow "Schnakenberg (gamma=1000)"          benchmark_schnakenberg.py
run      "Brusselator"                        benchmark_brusselator.py
run      "Gray-Scott: moljax leg"             benchmark_gray_scott_moljax.py
run      "Gray-Scott: Diffrax leg"            benchmark_gray_scott_diffrax.py

# ------------------------------------------------------------------
# Work-precision studies (paper Figures 6, 8-10)
# ------------------------------------------------------------------
run      "Work-precision: diffusion vs Diffrax" benchmark_diffrax_work_precision.py
run_slow "Work-precision: Gray-Scott"           work_precision_gray_scott.py
run_slow "Work-precision: Brusselator"          work_precision_brusselator.py
run      "Work-precision: reactor (NFE)"        work_precision_reactor_nfe.py

# ------------------------------------------------------------------
# Broadening studies (paper Section 6.4 limitations)
# ------------------------------------------------------------------
run "Variable-coefficient stress test"        benchmark_variable_coeff.py
run "Mixed boundary conditions"               benchmark_mixed_bc.py

# ------------------------------------------------------------------
# Pattern verification (paper Section 5.6)
# ------------------------------------------------------------------
run "Verify patterns vs conventional solver"  verify_patterns_conventional.py

echo ""
echo "=================================================================="
echo "Benchmark suite complete"
echo "=================================================================="
echo "Finished: $(date)"
echo ""
echo "Passed:  ${#PASSED[@]}"
echo "Failed:  ${#FAILED[@]}"
echo "Skipped: ${#SKIPPED[@]}"

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped stages (SKIP_SLOW=1):"
    for s in "${SKIPPED[@]}"; do echo "  - $s"; done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed stages:" >&2
    for s in "${FAILED[@]}"; do echo "  - $s" >&2; done
    echo ""
    echo "Results written so far: results/*.json"
    exit 1
fi

echo ""
echo "Results: benchmarks/results/*.json"
echo "Next:    ${PYTHON} benchmarks/plot_main_figures.py"

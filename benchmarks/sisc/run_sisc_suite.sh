#!/bin/bash
#
# SISC benchmark suite runner.
#
# Usage: ./run_sisc_suite.sh [--quick] [--backend gpu|cpu|any]
#   --quick             run the three-script subset (E1, E4, E9)
#   --backend BACKEND   forwarded to every script (default gpu); "any"
#                       skips the backend check so the suite runs on CPU
#   PYTHON=python3.12 ./run_sisc_suite.sh   chooses the interpreter
#
# The scripts import benchmark_utils from benchmarks/, so that directory is
# put on PYTHONPATH here. Results are written to benchmarks/results/sisc/
# and figures to benchmarks/figures/.
#
# Every stage is run even if an earlier one fails; the script exits
# nonzero at the end if any stage failed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
BACKEND="${BACKEND:-gpu}"
QUICK_MODE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --quick)     QUICK_MODE=true; shift ;;
        --backend)   BACKEND="$2"; shift 2 ;;
        --backend=*) BACKEND="${1#--backend=}"; shift ;;
        *)
            echo "run_sisc_suite.sh: unknown argument: $1" >&2
            echo "usage: ./run_sisc_suite.sh [--quick] [--backend gpu|cpu|any]" >&2
            exit 2 ;;
    esac
done

case "$BACKEND" in
    gpu|cpu|any) ;;
    *) echo "run_sisc_suite.sh: --backend must be gpu, cpu or any (got '$BACKEND')" >&2; exit 2 ;;
esac

export PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"

RESULTS_DIR="$SCRIPT_DIR/../results/sisc"
FIGURES_DIR="$SCRIPT_DIR/../figures"
mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"

echo "========================================"
echo "SISC Benchmark Suite"
echo "========================================"
echo "Working directory: $SCRIPT_DIR"
echo "Python:            $PYTHON"
echo "Backend:           $BACKEND"
echo "Results:           $RESULTS_DIR"
echo "Figures:           $FIGURES_DIR"
echo ""

if $QUICK_MODE; then
    echo "Running in QUICK mode (subset of benchmarks)"
    echo ""
fi

# Run one benchmark with timing; returns nonzero if the script failed.
run_benchmark() {
    local script=$1
    local name=$2

    echo "----------------------------------------"
    echo "Running: $name"
    echo "Script: $script --backend $BACKEND"
    echo "----------------------------------------"

    local start_time
    start_time=$(date +%s)

    if "$PYTHON" "$script" --backend "$BACKEND"; then
        local duration=$(( $(date +%s) - start_time ))
        echo "OK: $name completed in ${duration}s"
    else
        echo "FAILED: $name" >&2
        return 1
    fi

    echo ""
}

# List of benchmarks
BENCHMARKS=(
    "bench_iter_vs_grid.py:E1: GMRES Iterations vs Grid Size"
    "bench_iter_vs_dim.py:E2: GMRES Iterations vs Dimension"
    "bench_newton_policy_ablation.py:E3: Newton Policy Ablation"
    "bench_precond_variants.py:E4: Preconditioner Variants"
    "bench_bc_matrix.py:E6: Boundary Condition Matrix"
    "bench_presmooth_rannacher.py:E8: Rannacher Startup"
    "bench_jvp_vs_fd_sweep.py:E9: JVP vs FD Sweep"
    "bench_adjoint_grad_sanity.py:E10: Gradient Sanity Check"
    "bench_precision_fp32_fp64.py:E11: FP32 vs FP64"
    "bench_gpu_memory_scaling.py:E12: GPU Memory Scaling"
    "bench_3d_feasibility.py:E13: 3D Feasibility"
    "bench_imex_vs_fullimplicit_map.py:E14: Method Regime Map"
    "bench_reaction_dominant.py:E15: Reaction-Dominant Regime"
)

# Quick mode: run only essential benchmarks
QUICK_BENCHMARKS=(
    "bench_iter_vs_grid.py:E1: GMRES Iterations vs Grid Size"
    "bench_precond_variants.py:E4: Preconditioner Variants"
    "bench_jvp_vs_fd_sweep.py:E9: JVP vs FD Sweep"
)

if $QUICK_MODE; then
    BENCHMARKS=("${QUICK_BENCHMARKS[@]}")
fi

# Run benchmarks
PASSED=0
FAILED=0
TOTAL=${#BENCHMARKS[@]}

echo "Running $TOTAL benchmarks..."
echo ""

for entry in "${BENCHMARKS[@]}"; do
    IFS=':' read -r script name <<< "$entry"

    if run_benchmark "$script" "$name"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# Summary
echo "========================================"
echo "BENCHMARK SUITE COMPLETE"
echo "========================================"
echo "Passed: $PASSED / $TOTAL"
echo "Failed: $FAILED / $TOTAL"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""

# List generated files
echo "Generated result files:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (no JSON files found)"
echo ""
echo "Generated figures:"
ls -la "$FIGURES_DIR"/fig_*.pdf 2>/dev/null || echo "  (no figures found)"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: Some benchmarks failed!" >&2
    exit 1
fi

echo "All benchmarks completed successfully!"

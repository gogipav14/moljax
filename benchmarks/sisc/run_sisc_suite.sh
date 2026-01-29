#!/bin/bash
#
# SISC Benchmark Suite Runner
# Executes all new benchmarks for the SISC resubmission
#
# Usage: ./run_sisc_suite.sh [--quick]
#   --quick: Run subset of benchmarks for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create results directory
mkdir -p results

echo "========================================"
echo "SISC Benchmark Suite"
echo "========================================"
echo "Working directory: $SCRIPT_DIR"
echo "Results will be saved to: $SCRIPT_DIR/results/"
echo ""

# Check for quick mode
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo "Running in QUICK mode (subset of benchmarks)"
    echo ""
fi

# Function to run benchmark with timing
run_benchmark() {
    local script=$1
    local name=$2

    echo "----------------------------------------"
    echo "Running: $name"
    echo "Script: $script"
    echo "----------------------------------------"

    start_time=$(date +%s)

    if python3 "$script"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✓ $name completed in ${duration}s"
    else
        echo "✗ $name FAILED"
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
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

# Summary
echo "========================================"
echo "BENCHMARK SUITE COMPLETE"
echo "========================================"
echo "Passed: $PASSED / $TOTAL"
echo "Failed: $FAILED / $TOTAL"
echo ""
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""

# List generated files
echo "Generated result files:"
ls -la results/*.json 2>/dev/null || echo "  (no JSON files found)"
echo ""
echo "Generated figures:"
ls -la ../figures/fig_*.pdf 2>/dev/null || echo "  (no figures found)"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "WARNING: Some benchmarks failed!"
    exit 1
fi

echo "All benchmarks completed successfully!"

#!/usr/bin/env bash
# Run all strategy sweeps in FP64 then FP32, combine results into sweep_results_combined.csv.
# Must be launched from the benchmarks/ directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(realpath "$SCRIPT_DIR/..")"
PYTHON="$(which python)"

FP64_REFACTOR_EVERY_CSV="$SCRIPT_DIR/sweep_results_fp64_refactor_every.csv"
FP64_BASE_CASE_CSV="$SCRIPT_DIR/sweep_results_fp64_base_case.csv"
FP64_ITER0_ONLY_CSV="$SCRIPT_DIR/sweep_results_fp64_iter0_only.csv"
FP64_REFACTOR_N_CSV="$SCRIPT_DIR/sweep_results_fp64_refactor_n.csv"

FP32_REFACTOR_EVERY_CSV="$SCRIPT_DIR/sweep_results_fp32_refactor_every.csv"
FP32_BASE_CASE_CSV="$SCRIPT_DIR/sweep_results_fp32_base_case.csv"
FP32_ITER0_ONLY_CSV="$SCRIPT_DIR/sweep_results_fp32_iter0_only.csv"
FP32_REFACTOR_N_CSV="$SCRIPT_DIR/sweep_results_fp32_refactor_n.csv"

COMBINED_CSV="$SCRIPT_DIR/sweep_results_combined.csv"

echo "=== Installing package in FP64 mode ==="
CMAKE_BUILD_PARALLEL_LEVEL=1 CUDA_REAL_DOUBLE=1 uv pip install -ve "$PACKAGE_DIR"

echo ""
echo "=== Running benchmark sweeps (FP64) ==="
"$PYTHON" "$SCRIPT_DIR/run_benchmark_sweep.py"         --output "$FP64_REFACTOR_EVERY_CSV"
# "$PYTHON" "$SCRIPT_DIR/run_base_case_factors_sweep.py" --output "$FP64_BASE_CASE_CSV"
"$PYTHON" "$SCRIPT_DIR/run_iter0_only_sweep.py"        --output "$FP64_ITER0_ONLY_CSV"
"$PYTHON" "$SCRIPT_DIR/run_refactor_every_n_sweep.py"  --output "$FP64_REFACTOR_N_CSV"

echo ""
echo "=== Installing package in FP32 mode ==="
CMAKE_BUILD_PARALLEL_LEVEL=1 CUDA_REAL_FLOAT=1 uv pip install -ve "$PACKAGE_DIR"

echo ""
echo "=== Running benchmark sweeps (FP32) ==="
"$PYTHON" "$SCRIPT_DIR/run_benchmark_sweep.py"         --output "$FP32_REFACTOR_EVERY_CSV"
# "$PYTHON" "$SCRIPT_DIR/run_base_case_factors_sweep.py" --output "$FP32_BASE_CASE_CSV"
"$PYTHON" "$SCRIPT_DIR/run_iter0_only_sweep.py"        --output "$FP32_ITER0_ONLY_CSV"
"$PYTHON" "$SCRIPT_DIR/run_refactor_every_n_sweep.py"  --output "$FP32_REFACTOR_N_CSV"

echo ""
echo "=== Combining results ==="
"$PYTHON" - <<EOF
import pandas as pd

csvs = {
    "fp64_refactor_every": "$FP64_REFACTOR_EVERY_CSV",
    "fp64_iter0_only":     "$FP64_ITER0_ONLY_CSV",
    "fp64_refactor_n":     "$FP64_REFACTOR_N_CSV",
    "fp32_refactor_every": "$FP32_REFACTOR_EVERY_CSV",
    "fp32_iter0_only":     "$FP32_ITER0_ONLY_CSV",
    "fp32_refactor_n":     "$FP32_REFACTOR_N_CSV",
}

frames = []
for label, path in csvs.items():
    try:
        df = pd.read_csv(path)
        frames.append(df)
        print(f"{label:25s}: {len(df)} rows")
    except FileNotFoundError:
        print(f"{label:25s}: MISSING ({path})")

combined = pd.concat(frames, ignore_index=True)
combined.to_csv("$COMBINED_CSV", index=False)
print(f"\nCombined: {len(combined)} rows -> $COMBINED_CSV")
EOF

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Sweep runner for the direct_refactor_every_n strategy.

Sweeps over (nb_iter, refactor_period) pairs. Only pairs where
refactor_period < nb_iter are meaningful (otherwise equivalent to
direct_iter0_only). Pairs where refactor_period >= nb_iter are skipped.

Usage (from gpusim2grid/benchmarks/):
    python run_refactor_every_n_sweep.py
    python run_refactor_every_n_sweep.py --output results_refactor_n.csv
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
import time
import gpusim2grid
from _aux_sweep import DO_SLEEP_BETWEEN

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
GRIDS = ["case14", "case118", "case300", "case6515rte"]
BATCH_SIZES = [1, 10, 100, 1000]
# More iterations than direct_refactor_every: refactoring is infrequent
NB_ITERS = [2, 3, 4, 6]
# N values for refactor_every_n; only used when N < nb_iter
REFACTOR_PERIODS = [2, 3]
MAX_CONT_CPU = 500
MAX_CONT_GPU = 10_000

# ---------------------------------------------------------------------------
# GPU memory probe — extend batch sizes on high-memory devices
# ---------------------------------------------------------------------------
MB_PER_BATCH_SLOT = 6_000
LARGE_BATCH_TIERS = [
    (3_000,  22_000),
    (10_000, 75_000),
]

# ---------------------------------------------------------------------------
# Regex patterns for parsing benchmark stdout
# ---------------------------------------------------------------------------
RE_LIGHTSIM = re.compile(r"In addition.*?(\d+\.?\d+) ms / pf\)")
RE_FP = re.compile(r"\((fp(?:32|64))\)")
RE_PER_CTG = re.compile(r"Per contingency\s+:\s+(\d+\.?\d+) ms/ctg")
RE_REAL_E2E = re.compile(
    r"Real end-to-end totals.*?Total real time\s+:\s+(\d+\.?\d+) ms.*?Per contingency\s+:\s+(\d+\.?\d+) ms/ctg",
    re.DOTALL,
)
RE_GPU = re.compile(r"GPU used:\s*(.+)")
RE_EFF_BATCH = re.compile(r"Effective batch size used:\s*(\d+)")
RE_N_CONTS = re.compile(r"n cont = (\d+)")
RE_DIFF = re.compile(r"^Difference for contingency ", re.MULTILINE)
RE_FLOW_ERR = re.compile(r"^Flow \(A\) max difference for contingency \d+ \(\w+\): (\S+) A", re.MULTILINE)


def parse_output(stdout: str, nb_iter: int, batch_size: int, grid_name: str,
                 refactor_period: int) -> dict | None:
    """Extract benchmark metrics from a single run's stdout.

    Returns None if any required field is missing (failed run).
    """
    ls_matches = RE_LIGHTSIM.findall(stdout)
    if not ls_matches:
        return None
    lightsim_ms = ls_matches[0]

    fp_matches = RE_FP.findall(stdout)
    if not fp_matches:
        return None
    fp_precision = fp_matches[0]

    per_ctg_matches = RE_PER_CTG.findall(stdout)
    if not per_ctg_matches:
        return None
    per_ctg_ms = per_ctg_matches[0]

    eff_batch_matches = RE_EFF_BATCH.findall(stdout)
    eff_batch = eff_batch_matches[0] if eff_batch_matches else ""

    n_conts_matches = RE_N_CONTS.findall(stdout)
    n_conts = n_conts_matches[0] if n_conts_matches else ""

    gpu_matches = RE_GPU.findall(stdout)
    gpu_name = gpu_matches[0].strip() if gpu_matches else ""

    e2e_match = RE_REAL_E2E.search(stdout)
    real_total_ms = e2e_match.group(1) if e2e_match else ""
    real_per_ctg_ms = e2e_match.group(2) if e2e_match else ""

    flow_errs = [float(v) for v in RE_FLOW_ERR.findall(stdout)]
    max_flow_err = f"{max(flow_errs):.3e}" if flow_errs else ""

    return {
        "gpu": gpu_name,
        "grid_name": grid_name,
        "lightsim_ms_per_contingency": lightsim_ms,
        "fp_precision": fp_precision,
        "strategy": "direct_refactor_every_n",
        "refactor_period": refactor_period,
        "nb_iter_nr": nb_iter,
        "batch_size": batch_size,
        "effective_batch_size": eff_batch,
        "n_contingencies_gpu": n_conts,
        "gpu_ms_per_contingency": per_ctg_ms,
        "real_total_ms": real_total_ms,
        "real_per_contingency_ms": real_per_ctg_ms,
        "nb_differences": len(RE_DIFF.findall(stdout)),
        "max_flow_err_A": max_flow_err,
    }


def run_one(grid_name: str, batch_size: int, nb_iter: int, refactor_period: int,
            nb_max_cont_cpu: int, nb_max_cont_gpu: int, benchmarks_dir: Path) -> dict | None:
    """Spawn one benchmark subprocess and return parsed metrics (or None on failure)."""
    cmd = [
        sys.executable,
        str(benchmarks_dir / "contingency_analysis.py"),
        "--grid_name", grid_name,
        "--batch_size", str(batch_size),
        "--nb_iter", str(nb_iter),
        "--nb_max_cont_cpu", str(nb_max_cont_cpu),
        "--nb_max_cont_gpu", str(nb_max_cont_gpu),
        "--strategy", "direct_refactor_every_n",
        "--refactor_period", str(refactor_period),
    ]
    print(
        f"  Running: grid={grid_name:10s}  batch_size={batch_size:5d}  "
        f"nb_iter={nb_iter:2d}  refactor_period={refactor_period}",
        flush=True,
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(benchmarks_dir),
    )

    if result.returncode != 0:
        print(f"    [FAILED] returncode={result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    stderr: {line}")
        return None

    row = parse_output(result.stdout, nb_iter=nb_iter, batch_size=batch_size,
                       grid_name=grid_name, refactor_period=refactor_period)
    if row is None:
        print(f"    [PARSE ERROR] could not extract metrics from stdout")
        print("    stdout tail:")
        for line in result.stdout.strip().splitlines()[-10:]:
            print(f"      {line}")
    else:
        print(
            f"    lightsim={row['lightsim_ms_per_contingency']} ms/ctg  "
            f"gpu={row['gpu_ms_per_contingency']} ms/ctg  "
            f"{row['fp_precision']}"
        )
    return row


def query_gpu_total_memory_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def extend_batch_sizes_for_gpu(base_sizes: list[int]) -> list[int]:
    vram_mb = query_gpu_total_memory_mb()
    if vram_mb is None:
        print("  [GPU probe] nvidia-smi unavailable — using default batch sizes")
        return base_sizes

    print(f"  [GPU probe] Total VRAM: {vram_mb} MiB ({vram_mb/1024:.1f} GiB)")
    sizes = list(base_sizes)
    for bs, threshold_mb in LARGE_BATCH_TIERS:
        if vram_mb >= threshold_mb:
            sizes.append(bs)
            print(f"  [GPU probe] Adding batch_size={bs:6d}  "
                  f"(estimated {bs * MB_PER_BATCH_SLOT // 1000:.0f} GB, "
                  f"threshold {threshold_mb // 1000} GB)")
    return sizes


def main():
    parser = argparse.ArgumentParser(
        description="Sweep runner for the direct_refactor_every_n strategy"
    )
    parser.add_argument("--nb_iters", nargs="+", type=int, default=NB_ITERS,
                        help="NR iteration counts to sweep (default: 4 8 12)")
    parser.add_argument("--refactor_periods", nargs="+", type=int, default=REFACTOR_PERIODS,
                        help="Refactor period values N to sweep (default: 2 5 10)")
    parser.add_argument("--output", default="sweep_results_refactor_n.csv",
                        help="Output CSV file (default: sweep_results_refactor_n.csv)")
    parser.add_argument("--grids", nargs="+", default=GRIDS, help="Grid names to sweep")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=BATCH_SIZES,
                        help="Batch sizes to sweep")
    parser.add_argument("--nb_max_cont_cpu", type=int, default=MAX_CONT_CPU)
    parser.add_argument("--nb_max_cont_gpu", type=int, default=MAX_CONT_GPU)
    args = parser.parse_args()

    benchmarks_dir = Path(__file__).parent.resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = benchmarks_dir / output_path

    package_version = gpusim2grid.__version__

    csv_columns = [
        "gpu",
        "package_version",
        "grid_name",
        "lightsim_ms_per_contingency",
        "fp_precision",
        "strategy",
        "refactor_period",
        "nb_iter_nr",
        "batch_size",
        "effective_batch_size",
        "n_contingencies_gpu",
        "gpu_ms_per_contingency",
        "real_total_ms",
        "real_per_contingency_ms",
        "speedup_vs_l2g",
        "nb_differences",
        "max_flow_err_A",
    ]

    rows = []

    print("\n=== GPU memory probe ===")
    effective_batch_sizes = extend_batch_sizes_for_gpu(args.batch_sizes)
    args.batch_sizes = effective_batch_sizes

    # Only sweep (nb_iter, refactor_period) pairs where refactor_period < nb_iter
    valid_pairs = [
        (n, p) for n in args.nb_iters for p in args.refactor_periods if p < n
    ]
    total = len(args.grids) * len(valid_pairs) * len(args.batch_sizes)
    done = 0

    print(f"\n=== Strategy: direct_refactor_every_n ===")
    print(f"Valid (nb_iter, refactor_period) pairs: {valid_pairs}")

    for grid_name in args.grids:
        print(f"\n  --- Grid: {grid_name} ---")
        for nb_iter, refactor_period in valid_pairs:
            print(f"\n    -- nb_iter={nb_iter}, refactor_period={refactor_period} --")
            for batch_size in args.batch_sizes:
                done += 1
                print(f"[{done}/{total}]", end=" ")
                row = run_one(grid_name, batch_size, nb_iter, refactor_period,
                              args.nb_max_cont_cpu, args.nb_max_cont_gpu, benchmarks_dir)
                if row is not None:
                    row["package_version"] = package_version
                    try:
                        speedup = float(row["lightsim_ms_per_contingency"]) / float(row["real_per_contingency_ms"])
                        row["speedup_vs_l2g"] = f"{speedup:.3f}"
                    except (ValueError, ZeroDivisionError):
                        row["speedup_vs_l2g"] = "N/A"
                    rows.append(row)
                if DO_SLEEP_BETWEEN:
                    time.sleep(60)
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_columns)
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"\nDone. {len(rows)}/{total} runs succeeded.")
    print(f"Results written to: {output_path}")

    if rows:
        col_widths = {col: max(len(col), max(len(str(r[col])) for r in rows)) for col in csv_columns}
        header = "  ".join(col.ljust(col_widths[col]) for col in csv_columns)
        sep = "  ".join("-" * col_widths[col] for col in csv_columns)
        print(f"\n{header}")
        print(sep)
        for row in rows:
            print("  ".join(str(row[col]).ljust(col_widths[col]) for col in csv_columns))


if __name__ == "__main__":
    main()

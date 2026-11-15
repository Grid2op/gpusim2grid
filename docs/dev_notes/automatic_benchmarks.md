# Automatic benchmarks on Modal, published on Read the Docs

Status: **plan, nothing implemented yet.** Written as a handoff for a Claude Code instance that
has a local CUDA GPU: build and debug everything **locally first**, then on Modal as little as
possible (a Modal debugging loop costs money on every iteration).

## 1. Decisions already taken (do not reopen them)

- **Hardware: one A100 40 GB on Modal, FP64 only.** No FP32 build, no other GPU for now (H100
  maybe later). The benchmark wheel is compiled for the A100 only: `-DCMAKE_CUDA_ARCHITECTURES=80`,
  CUDA 13, FP64.
- **GPU only. No CPU baseline, nothing timed on a CPU on Modal.** lightsim2grid (CPU) may be used
  *locally* to check correctness while developing, never as a timed baseline in the results.
- **No single power flow solve** in the comparison, for any tool (meaningless on a GPU).
- **What is compared:**
  - gpusim2grid against itself: `linear_solver='cudss'` vs `'own_lu'`, and `outer_loops=False` vs
    `True`.
  - **ExaPF** (Julia, GPU): **time series** only (batched power flow with varying loads / gens).
  - **p3s** (parallel pandapower, its **GPU** Newton): **time series** and **contingency analysis**.
- **Fully automatic, no human step, no manual dispatch:** triggered by **publishing a GitHub
  release**.
- **No commit made by a bot.** The results never go on a branch (not even an unmerged data
  branch: it would pass the DCO check, but it is not in its spirit). They are a **release
  asset** (`benchmarks.json` attached to the release), which Read the Docs downloads at build
  time. The repo is public, so the asset URL needs no credentials.
- The docs for version `vX.Y` show the numbers of release `vX.Y`.

## 2. Overview

```
release published (GitHub)
  └─ workflow benchmarks.yml
       ├─ job build-wheel   GPU-less runner, CUDA 13, FP64, sm_80   → artifact (same run)
       └─ job benchmark     needs: build-wheel
            ├─ modal run ci/modal/benchmarks.py   (A100-40GB)   → benchmarks.json (returned to the runner)
            ├─ gh release upload $TAG benchmarks.json --clobber
            └─ POST readthedocs API: rebuild version $TAG
Read the Docs build of $TAG
  └─ docs/conf.py hook: download benchmarks.json of $TAG → generate tables + plots → docs/benchmarks.rst
```

The one rule that keeps local debugging possible: **everything that measures is a plain local
CLI.** The Modal script only builds the image, mounts the wheel and calls that CLI; the workflow
only calls the Modal script. So the whole suite, the ExaPF and p3s runners and the docs page can be
developed and checked on a local GPU without Modal.

## 3. Files to create

| File | Role |
|---|---|
| `benchmarks/suite/run_suite.py` | Entry point. Runs the whole matrix (or a `--quick` subset), writes one `benchmarks.json`. Runs locally or inside Modal, unchanged. |
| `benchmarks/suite/grids.py` | Downloads the grids (pinned), converts them for each tool, caches them. |
| `benchmarks/suite/scenarios.py` | Generates the time-series scenarios deterministically (seed), in a tool-neutral format. |
| `benchmarks/suite/g2g_runner.py` | gpusim2grid workloads (time series, contingency analysis) × the knobs. |
| `benchmarks/suite/exapf_runner.jl` + `exapf_runner.py` | ExaPF time series; the `.py` calls `julia` as a subprocess and reads back JSON. |
| `benchmarks/suite/p3s_runner.py` | p3s time series and contingency analysis. |
| `benchmarks/suite/schema.py` | Builds and checks the results JSON (section 5). |
| `ci/modal/benchmarks.py` | Modal wrapper, modelled on `ci/modal/gpu_tests.py`. |
| `.github/workflows/benchmarks.yml` | The release-triggered workflow. |
| `docs/_ext/benchmark_results.py` | Sphinx hook: fetches the JSON and generates the page. |
| `docs/benchmarks.rst` | The page (includes the generated part), added to the `index.rst` toctree. |

Keep the existing `benchmarks/*.py` scripts as they are. Reuse their code where it fits:
`benchmarks/outer_loops.py` shows how a grid with the outer loops is built from pypowsybl
(`init_from_pypowsybl(olf_rules=True)`), and `benchmarks/injection_sweep.py` shows the per-element
scenarios.

## 4. The benchmark matrix

### 4.1 Grids

- **One source for every tool: public MATPOWER cases**, pinned to a commit. Suggested source:
  [pglib-opf](https://github.com/power-grid-lib/pglib-opf), e.g. `case1354_pegase`,
  `case2869_pegase`, `case9241_pegase`, and one larger case if the run time allows
  (`case13659_pegase`). ExaPF reads `.m` files directly. pandapower (p3s) and lightsim2grid get
  them through pandapower's MATPOWER converter.
- **Outer loops** need a grid solved with lightsim2grid's `NROuter_*` algorithm, i.e. built
  through pypowsybl (`init_from_pypowsybl(olf_rules=True)`). **To verify:** whether pypowsybl
  imports MATPOWER cases (I believe it reads the binary `.mat` format only, so a `.m` → `.mat` step
  may be needed). If that does not work, use pypowsybl's built-in `ieee118` / `ieee300` for the
  outer-loops axis only, and say so on the page.
- Converters disagree on details (transformer models, shunts, generator limits). Never trust a
  conversion blindly: every tool's solution is checked against the others (section 4.5).

### 4.2 Workloads

| Workload | gpusim2grid | ExaPF | p3s |
|---|---|---|---|
| Time series, N scenarios of load / gen variation | `InjectionSweepGPU` (`set_injections_from_elements`) | batched power flow | GPU time series |
| Contingency analysis, N-1 on branches | `ContingencyAnalysisGPU` | — | GPU contingency analysis |

- **Scenarios** (`scenarios.py`): log-normal multipliers on each load's P and Q (same factor for
  both, as `benchmarks/injection_sweep.py` does) and generator P rescaled to follow the total
  load. Stored once per grid in MATPOWER order (per bus with `Pd ≠ 0` for loads, per generator for
  gens), as `.npz`. Each runner maps them to its own element order. Every tool gets **the same
  scenarios**.
- **Sizes:** two values of N for each workload, e.g. 1 000 and 10 000 scenarios. The contingency
  list is every branch whose loss keeps the grid connected, capped (e.g. 10 000) and identical for
  gpusim2grid and p3s.

### 4.3 gpusim2grid knobs

For every grid and workload:

- `linear_solver`: `'cudss'`, `'own_lu'`.
- `outer_loops`: `False`, `True` (on a grid solved with `NROuter_*`). `True` requires
  `scheduling='continuous'`, `nb_iter_per_round=1` and `ir_n_steps=1` (see CLAUDE.md "Outer
  loops"). For the `False` rows, use the default that is fastest and report which one it is.
- Not a knob: FP64 only, one strategy per row (stated in the JSON).

### 4.4 Fairness settings (same problem for every tool)

- **The cross-tool rows use a model every tool supports.** As far as I know ExaPF has a single
  slack and no reactive limits (**verify**), so the gpusim2grid rows compared with it run
  `use_distributed_slack=False`, `outer_loops=False`. Same for p3s, according to what its GPU
  solver supports (**verify**). The gpusim2grid-only rows (outer loops etc.) can use the full model.
- **Same convergence tolerance** on every tool, stated in the JSON (lightsim2grid takes its `tol`
  in MVA, gpusim2grid in per unit: see CLAUDE.md "Batch scheduling").
- **Timing**: `gpusim2grid.warmup()` first. Separate setup time (construction, analysis) from run
  time and report both. Median of 5 repetitions after one untimed run. For ExaPF, the Julia
  compilation must not be timed: run once, then time. GPU work synchronized before stopping the
  clock (for ExaPF: `CUDA.@sync`).
- Record per row: number of converged scenarios / contingencies. A tool that converges fewer rows
  is not faster just because it stops earlier.

### 4.5 Correctness checks (in every run, in the JSON)

- For each (grid, workload), the max |ΔV| (complex, per unit) between each tool and gpusim2grid
  with cuDSS on a sample of rows (e.g. 100). Above a threshold (1e-6, to be adjusted on the first
  local runs), the row is flagged on the page, not dropped.
- `cudss` vs `own_lu`, `outer_loops` vs none: also compared (the outer loops change the answer on
  purpose; report the difference, do not flag it).

## 5. Results JSON (`benchmarks.json`)

One file per run. `schema.py` builds and validates it; the docs page reads only this.

```json
{
  "schema_version": 1,
  "gpusim2grid": {"version": "0.3.0", "commit": "<sha>", "tag": "v0.3.0", "is_fp32": false},
  "date_utc": "2026-11-15T12:00:00Z",
  "hardware": {"gpu_name": "NVIDIA A100-SXM4-40GB", "gpu_memory_mib": 40960,
               "driver": "...", "cuda_runtime": "13.x", "requested_gpu": "A100-40GB"},
  "versions": {"lightsim2grid": "1.3.1", "cudss": "...", "pypowsybl": "...",
               "julia": "...", "ExaPF": "...", "CUDA.jl": "...", "p3s": "...", "pandapower": "..."},
  "grids": [{"name": "case9241_pegase", "source": "pglib-opf@<sha>", "n_bus": 9241, "n_branch": 16049}],
  "results": [
    {"grid": "case9241_pegase", "workload": "time_series", "n_rows": 10000,
     "tool": "gpusim2grid", "options": {"linear_solver": "cudss", "outer_loops": false,
                                        "use_distributed_slack": false, "scheduling": "continuous"},
     "setup_ms": 0.0, "run_ms": {"median": 0.0, "min": 0.0, "max": 0.0, "n_rep": 5},
     "n_converged": 10000, "max_dv_vs_ref": 0.0, "flagged": false, "notes": ""}
  ],
  "skipped": [{"tool": "p3s", "grid": "...", "workload": "...", "reason": "..."}]
}
```

A tool that fails or is unavailable gets a `skipped` entry with the reason: **one tool failing
must never lose the whole run** (catch per row, write what was measured).

## 6. Modal wrapper (`ci/modal/benchmarks.py`)

Modelled on `ci/modal/gpu_tests.py` (same image pattern, same `LD_LIBRARY_PATH` handling, the same
check that a GPU is visible and the precision is right).

- `gpu="A100-40GB"`, one entry, no fallback list.
- **Check what was actually given**: `nvidia-smi --query-gpu=name,memory.total,clocks.max.sm
  --format=csv` goes into `hardware`. I believe Modal may run an A100-40GB request on an 80 GB
  card (**verify** on Modal's GPU docs page: modal.com was not reachable from the session that
  wrote this note). If it does: keep the results and say so on the page, do not fail the run.
- **Image** (cached, never rebuilt by a code change): the PyPI CUDA 13 libraries, cuDSS,
  lightsim2grid, pypowsybl, pandapower, p3s; **Julia** (juliaup) with ExaPF, CUDA.jl and their
  dependencies **instantiated and precompiled in the image**, from a committed
  `Project.toml` + `Manifest.toml` (pin everything: a CUDA.jl update against Modal's driver is the
  most likely way this breaks on its own). The grids can be downloaded into the image too.
- The wheel and `benchmarks/suite/` are mounted at container start (`add_local_dir`, no copy), as
  in `gpu_tests.py`.
- The remote function runs `run_suite.py` and **returns the JSON as its result**: the local
  entrypoint writes it to `--out`. No Modal volume.
- `timeout` on the function (e.g. 60 min) so a hang cannot cost more than that.
- **Check the wheel's commit** against the commit being benchmarked (store it in the wheel or pass
  it), so results can never be attributed to the wrong code.

## 7. Workflow (`.github/workflows/benchmarks.yml`)

- `on: release: types: [published]`. No `workflow_dispatch`.
- **Job `build-wheel`**: a copy of the CUDA 13 × FP64 leg of `build.yml` (same image, same pip
  requirements: keep them in sync with `pyproject.toml`), with
  `CMAKE_ARGS: -DCMAKE_CUDA_ARCHITECTURES=80`. Uploads the wheel as an artifact of this run.
- **Job `benchmark`** (`needs: build-wheel`): checkout at the release tag, download the wheel,
  `pip install modal`, `modal run ci/modal/benchmarks.py --out benchmarks.json`, then
  `gh release upload "$TAG" benchmarks.json --clobber`, then ask Read the Docs to rebuild that
  version:
  `POST https://readthedocs.org/api/v3/projects/gpusim2grid/versions/<version-slug>/builds/` with
  `Authorization: Token $READTHEDOCS_TOKEN`. (**Verify**: the project slug, and how RTD turns a
  tag into a version slug, and that the version is active.) The rebuild is needed because RTD
  builds a new tag immediately, before the benchmark has finished.
- `permissions: contents: write` on the `benchmark` job only (for the upload); `contents: read`
  everywhere else.
- Secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (exist), `READTHEDOCS_TOKEN` (to be created by
  the maintainer). Fail with a clear error if one is missing, as `gpu-tests` does.

## 8. Docs page (`docs/_ext/benchmark_results.py`, `docs/benchmarks.rst`)

- A small Sphinx extension registered in `conf.py`, run at `builder-inited`. It writes
  `docs/_generated/benchmarks_results.rst` (tables) and PNG plots under `docs/_static/_generated/`
  (matplotlib: add it to `docs/requirements.txt`; the plot style should follow the repo's other
  figures if there are any). `docs/benchmarks.rst` includes the generated file.
- **Where the JSON comes from**, in order:
  1. `G2G_BENCH_RESULTS=<path>` (a local file): local development and tests.
  2. On RTD, a tag build (`READTHEDOCS_VERSION_TYPE == "tag"`, **verify** the variable names):
     `https://github.com/Grid2op/gpusim2grid/releases/download/<tag>/benchmarks.json`.
  3. Otherwise (`latest`, branches, PR builds): the latest release,
     `https://github.com/Grid2op/gpusim2grid/releases/latest/download/benchmarks.json`, labelled as
     such ("numbers from release vX.Y").
- **The docs build must never fail because of this**: short timeout (~20 s), and any error
  (no network, 404, bad JSON, wrong `schema_version`) gives a page saying "benchmark results not
  available yet for this version" plus a docs-build warning.
- The page always states: release / commit, date, GPU actually used, every package version, the
  fairness settings (4.4), and the flagged rows.
- **Content**: one table per (grid, workload), tools / options as rows, setup and run time,
  throughput (rows per second), converged count, max |ΔV|. Plots: run time vs N per tool, and
  cudss vs own_lu, outer loops vs none.

## 9. Local test plan (on a CUDA machine, before any Modal run)

Each step has a pass criterion. Do them in order; do not go to Modal before step 6 passes.

1. **Build** gpusim2grid FP64 for the local GPU (`source env_compile.sh`,
   `CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=native pip install --no-build-isolation .`) and run
   `pytest tests/python/test_outer_loops_batch.py -rs`. Pass: the outer-loop tests **run** (not
   skipped on `have_ls2g_outer_loops`) with the PyPI `lightsim2grid==1.3.1`. (Its headers were
   checked to have everything the outer loops need; this is the first real build against them.)
2. **Grids and scenarios**: `python benchmarks/suite/grids.py` and `scenarios.py`. Pass: every
   grid loads in lightsim2grid and its base case converges; for the pypowsybl path, the grid
   solves with `NROuter_*`; scenarios are bit-identical between two runs with the same seed.
3. **gpusim2grid runner**, `run_suite.py --quick --tools gpusim2grid` (smallest grid, small N).
   Pass: valid JSON; every option combination runs; locally only, V matches lightsim2grid's
   CPU solve of a few rows to ~1e-8 (this check is not part of the published results).
4. **ExaPF runner**: install Julia (juliaup), `julia --project=benchmarks/suite -e 'using Pkg;
   Pkg.instantiate()'`, then `run_suite.py --quick --tools exapf`. Pass: same scenarios solved,
   |ΔV| vs gpusim2grid (single slack) under the threshold; the timed run excludes compilation
   (second run not much slower than the third). **Verify ExaPF's current API** for batched power
   flow with per-scenario loads (I believe `BlockPolarForm` over k scenarios; check its docs and
   pin the version in the Manifest).
5. **p3s runner**: **first identify p3s precisely** (PyPI / GitHub, licence, that it has a GPU
   Newton for time series and for contingency analysis, which model it solves). If it cannot be
   found or has no GPU path: stop there for p3s, write it in `skipped` and report back, do not
   substitute another tool. Pass: |ΔV| vs gpusim2grid under the threshold on both workloads, same
   contingency list.
6. **Full local run** `run_suite.py --out /tmp/bench.json` on the full matrix (the local GPU will
   not be an A100: the numbers do not matter, the run time does). Pass: completes, and the run
   time scaled to an A100 fits in the Modal timeout with margin. Trim the matrix if it does not.
7. **Docs**: `G2G_BENCH_RESULTS=/tmp/bench.json make -C docs html`, then without the variable and
   without network. Pass: the page renders with the tables and plots in the first case, the
   "not available" page and no build failure in the second.
8. **The Modal image, without Modal**: reproduce it in Docker
   (`docker run --gpus all python:3.10-slim`, the same `pip install` list and Julia setup as the
   image, the wheel and `benchmarks/suite/` mounted) and run `run_suite.py --quick` in it. Pass:
   runs. This catches most image problems (missing libraries, `LD_LIBRARY_PATH`, Julia
   precompilation) for free.

## 10. Modal test plan (cheap first)

1. `modal run ci/modal/benchmarks.py --quick` with `G2G_GPU=T4` (the cheapest GPU) and a wheel
   built with `-DCMAKE_CUDA_ARCHITECTURES=75`: builds the image once (CPU, a few minutes) and runs
   the quick subset. Pass: JSON comes back with the right `hardware`.
2. One full run on `A100-40GB` with the sm_80 wheel. Pass: JSON complete, check the GPU actually
   given and the total run time (cost).
3. Workflow: test it on a **pre-release** of a test tag (create and delete it afterwards), with the
   RTD rebuild step pointed at that version. Pass: asset attached, RTD build of that version shows
   the page. Then delete the test release and tag.

## 11. Things to verify (unknown when this note was written)

- p3s: what exactly it is, its GPU APIs, the model it solves (section 9 step 5).
- ExaPF: current API for batched power flow, its model (single slack? reactive limits?).
- pypowsybl's MATPOWER import (`.m` or `.mat`), for the outer-loops grids.
- Modal: whether `A100-40GB` can be upgraded to 80 GB; current price per hour.
- Read the Docs: project slug, tag → version slug, `READTHEDOCS_VERSION*` variable names, and
  that tag versions are activated automatically (an automation rule may be needed).
- Whether GitHub release-asset URLs serve CORS headers (only relevant if someone wants the page to
  load the JSON in the browser instead of at build time; the plan does not need it).

## 12. Repository rules that apply (see CLAUDE.md)

- Every commit signed off by a human (DCO), the assistant credited with `Assisted-by:`.
- A `CHANGELOG.rst` entry (`[ADDED]`) for the benchmark workflow and docs page.
- `build.yml`, `pyproject.toml` and the new workflow's requirements kept in sync.

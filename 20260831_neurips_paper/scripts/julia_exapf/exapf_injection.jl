#!/usr/bin/env julia

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Julia/ExaPF.jl counterpart to python cpu baselines (pandapower, pypowsybl / olf, lightsim2grid, rustpower):
# runs the exact same injection scenarios (exported beforehand by
# ../export_data_for_julia.py into .npy files) through ExaPF's batched Newton-Raphson power
# flow, on CPU or GPU (CUDA + CUDSS), and writes out benchmark timings (json) and complex
# bus voltages (npy) in the same layout/convention as the python scripts, so they can be
# loaded and compared from python/notebooks exactly like the other tools' results.
#
# Usage:
#   julia --project=. exapf_injection.jl --data-dir ../../matpower_injection_data/exapf_data_sable_12000 --backend cpu
#   julia --project=. exapf_injection.jl --data-dir ../../matpower_injection_data/exapf_data_sable_12000 --backend gpu --batch-size 500 --save-voltages
#   julia --project=. -t 8 exapf_injection.jl --data-dir ../../matpower_injection_data/exapf_data_sable_12000 --backend cpu --nthreads 8
#
# --nthreads (CPU only, see run_sweep/process_chunk_group): parallelizes the outer scenario
# loop -- the --batch-size chunk sweep over one grid's own scenarios -- across Julia tasks.
# Requires launching Julia itself with at least as many threads (`julia -t N` /
# JULIA_NUM_THREADS=N); it never parallelizes the case_order loop over grids (each grid is
# still processed one at a time, smallest first), and is always forced to 1 on --backend gpu.
#
# --max-scen: solve only the first N scenarios of each case's own --data-dir export (0 = all
# available). Lets base_launch_exapf.sh export one --data-dir per grid at that grid's own top
# --nthreads and reuse it for every smaller --nthreads via a smaller --max-scen, instead of
# re-exporting per (grid, nthreads) pair -- mirrors powermodels_injection.jl's own --max-scen.
#
# time_series timing fields (build_time/factorization_time/modif_time/solve_time/total_time/
# wall_time):
#   build_time:         sum of Jacobian/NetworkStack construction (build_problem_from_polar),
#                       one call per distinct chunk size actually solved (normally once per
#                       worker -- see process_chunk_group).
#   factorization_time:  sum of the matching one-time KLU symbolic+numeric factorizations
#                        (ExaPF.default_linear_solver) for those same calls.
#   modif_time:  sum of set_scenario_data! calls (uploading each chunk's load/gen/v_init into
#                the batched stack) -- ExaPF's equivalent of olf's update_loads/
#                update_generators or rustpower's build_sbus.
#   solve_time:  sum of ExaPF.solve! calls (the actual Newton-Raphson iterations).
#   total_time:  build_time + factorization_time + modif_time + solve_time. Under --nthreads > 1
#                this sums concurrently-running workers' own CPU time, not elapsed time (see
#                run_sweep's docstring) -- same aggregate-vs-elapsed caveat every other backend's
#                total_time carries.
#   wall_time:   elapsed time measured only around the actual per-worker dispatch, i.e. around
#                build_problem_from_polar + set_scenario_data! + ExaPF.solve! for every chunk --
#                run_sweep's own return value.
#
# Jacobian construction and KLU factorization are deliberately kept INSIDE wall_time/total_time,
# not excluded as one-time "grid loading" the way every backend's actual file parse is (see
# below). pp_injection.py/olf_injection.py/rustpower_injection.py/powermodels_injection.jl have
# no comparable concept to compare against at all -- each rebuilds its Jacobian and refactorizes
# completely from scratch on every single scenario (pp.runpp(), pypl.run_ac(),
# NewtonSolver.setup_context(), PowerModels.compute_ac_pf()), so for them build+factorize simply
# IS the per-scenario cost, always fully counted. ls_injection.py's InjectionSweepCPP looks
# closer (Ybus/topology stays fixed across the sweep, `DirectRefactorEvery`: factor once,
# refactor every iteration), but its own C++ implementation (lightsim2grid's source, not visible
# in ls_injection.py itself) also builds a private, independently-factorized grid per thread
# when nb_thread > 1 -- and that per-thread cost is NOT excluded there either: ls_injection.py
# only ever calls one opaque `ts.compute_Vs(...)`, self-timed via `ts.total_time()`/
# `ts.preprocessing_time()`/`ts.solver_time()`, with no visibility into, let alone exclusion of,
# whatever per-thread setup happens inside it. So keeping build_time/factorization_time inside
# ExaPF's own wall_time/total_time here is the accurate match to that precedent, not a deviation
# from it: excluding it would make a higher --nthreads run look artificially
# cheap even though it is doing (and, ideally, parallelizing) strictly more total setup work.
# What IS excluded, as a genuinely fixed, batch-size/nthreads-independent cost, is only the raw
# case-file *parse* (`ExaPF.load_polar`, see `load_case`): done exactly once per grid in
# run_sweep, before the wall clock starts, and its result (`polar`) shared read-only by every
# worker's own build_problem_from_polar call -- reported separately, outside time_series, as
# tmp_res["grid_load"]["load_time"].
#
# JIT-compile cost is a separate concern from the above and is paid once, globally, in main()'s
# warmup block (see there) before any case is timed -- unrelated to the load_time exclusion and 
# is done similarly than for other tools.
#
# --max-chunks (GPU only, see run_sweep): caps a grid's --batch-size sweep to the first N
# chunks instead of running every one (default 128, 0 = no cap). Every chunk of a grid has the
# same nblocks on GPU, so per-chunk timing is essentially constant and a prefix's average is a
# good estimate of the full sweep's -- this trades a bit of statistical noise for a lot of wall
# time on grids with many chunks. Ignored on --backend cpu. nb_solved/nb_total in the result
# json reflect only the chunks actually run; "n_chunks_total"/"chunks_capped" record whether
# and how much the sweep was capped.
#
# Robustness: a case that errors out (singular Jacobian, a corrupt/oversized case, ...) is
# caught, logged into the result json under that case's "error" key, and does not stop the
# rest of the sweep. On GPU, --batch-size splits a case's scenarios into chunks solved
# sequentially (bounding peak GPU memory instead of building one giant block-diagonal system
# for all nscen scenarios at once); a chunk that itself OOMs is also caught and skipped (its
# slice of the voltage output, if requested, is left as NaN) -- but an out-of-memory error
# specifically (see is_oom_error) is treated differently from every other error: it stops
# that case's sweep immediately (remaining chunks would just re-fail the same allocation) AND
# stops the whole case_order loop, since case_order is smallest-grid-first, so every
# remaining case is at least as likely to OOM. Ctrl-C (InterruptException) is never caught.
#
# GPU-only non-convergence bailout (see run_sweep, CONSECUTIVE_DIVERGE_LIMIT, and the
# README's "GPU batched-solve non-convergence at specific --batch-size values" section):
# separately from OOM, a chunk can also just fail to converge (has_converged=false, no
# exception) -- observed on GPU only, and confirmed to be a property of the (grid,
# --batch-size) combination itself (same scenario data converges fine on CPU, and on the same
# grid at other --batch-size values), not the data. That failure mode does not self-recover
# within a (grid, --batch-size) run, so 5 consecutive non-converged chunks (CONSECUTIVE_
# DIVERGE_LIMIT) stop the REST OF THAT GRID's chunks -- but, unlike OOM, do NOT stop the
# case_order loop: this is a GPU-numerics issue tied to that specific grid/batch-size pair,
# not a memory-pressure signal that predicts anything about bigger grids.
#
# Element ordering note (see export_data_for_julia.py's module docstring for the full
# reasoning): ExaPF and lightsim2grid both preserve the raw matpower file's row order for
# buses and generators, so export_data_for_julia.py already produced bus-indexed load arrays
# and (disconnected-generator-filtered) generator-indexed pgen arrays that line up 1:1
# with ExaPF's own internal ordering. No further re-alignment is needed here.
#
# INFO / WARNING: this script has been mainly generated by Claude (sonnet 5), carefully
# reviewed by the authors of the paper. Results have been "sanity checked" when compared with 
# other methods (mainly lightsim2grid).

println("Activating Julia project environment ($(@__DIR__))...")
import Pkg
Pkg.activate(@__DIR__)
println("Project environment activated.")

# Pin every implicit (non-Julia-Threads) thread pool to 1 before any package is loaded, so any
# library that reads these once at __init__ time (rather than lazily) still sees them -- see
# main()'s own LinearAlgebra.BLAS.set_num_threads(1) for the same reasoning applied to BLAS (set
# at runtime there instead, since BLAS.set_num_threads is a real function call, not just an env
# read).
ENV["OMP_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"

println("Loading packages (ArgParse, ExaPF, CUDA, CUDSS, NPZ, JSON, ...) -- first run after a fresh depot/project can take a while (precompilation)...")
using ArgParse
using ExaPF
using CUDA
using CUDSS
using NPZ
using JSON
using LinearAlgebra
using Printf
using Base.Threads
println("Packages loaded.")

const PS = ExaPF.PowerSystem

const TOL_PF = 1e-8
const MAX_ITER = 10
# GPU-only (see run_sweep): case14 vs case_ACTIVSg500 vs ... has shown a --batch-size value
# can make EVERY chunk of a grid fail to converge (has_converged=false, no exception) on GPU
# only -- confirmed same data converges fine on CPU and at other --batch-size values for the
# same grid, so this is not a data/init problem but a GPU-batched-solve one (see cuDSS's own
# "currently in preview" uniform-batch LU path, exercised via ExaPF's nblocks>1 DirectSolver).
# Once this many consecutive chunks fail to converge, further chunks of the same grid at the
# same --batch-size are assumed doomed too; bail out of the grid rather than burn through the
# whole sweep re-failing one chunk at a time.
const CONSECUTIVE_DIVERGE_LIMIT = 5

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data-dir"
            help = "directory produced by export_data_for_julia.py (contains meta.json + per-case .npy files); if omitted, defaults to ../../matpower_injection_data/exapf_data_<sample-data-meth>_<nb-pf> (relative to this script), matching export_data_for_julia.py's own naming convention"
            arg_type = String
            default = ""
        "--sample-data-meth"
            help = "sampling method the data was exported with (fr, sable or pfdelta); only used to build the default --data-dir when --data-dir is not given"
            arg_type = String
            default = "sable"
        "--nb-pf"
            help = "number of powerflows requested at export time; only used to build the default --data-dir when --data-dir is not given"
            arg_type = Int
            default = 12000
        "--max-scen"
            help = "solve only the first N scenarios of each case (0 = all available in --data-dir); lets one --data-dir export (sized at a grid's own top thread count) be reused for every smaller thread count via a smaller --max-scen, instead of re-exporting per thread count -- same role as powermodels_injection.jl's own --max-scen"
            arg_type = Int
            default = 0
        "--backend"
            help = "cpu or gpu"
            arg_type = String
            default = "cpu"
        "--results-dir"
            help = "where to write the benchmark json / _Vs.npy files (matches PATH_RESULTS in ls_injection.py)"
            arg_type = String
            default = joinpath(@__DIR__, "..", "..", "raw_results")
        "--add-to-name"
            help = "customize the name of the experiment"
            arg_type = String
            default = ""
        "--save-voltages"
            help = "save the complex bus voltages for every scenario"
            action = :store_true
        "--warmup-case"
            help = "case file (from the data dir's own case list) used to pay the Julia JIT/GPU-compile cost before timing starts"
            arg_type = String
            default = ""
        "--batch-size"
            help = "split each case's scenarios into chunks of this size, solved sequentially per --nthreads worker (bounds peak memory); 0 (default) = solve all scenarios in one block when --nthreads is 1, or auto-split into ceil(nscen/nthreads) roughly-equal chunks (one per worker) when --nthreads > 1 -- see run_sweep's docstring"
            arg_type = Int
            default = 0
        "--max-chunks"
            help = "GPU backend only (ignored on --backend cpu): cap each grid's scenario sweep to at most this many --batch-size chunks, solving only the first ones and skipping the rest (their slice of --save-voltages output, if requested, is left as NaN, and they don't count towards nb_solved/nb_total). 0 = no cap, run every chunk. Default 128: on GPU every chunk of a grid has the same nblocks (batch-size), so build/factorization/solve cost per chunk is essentially constant -- the timing average over the first --max-chunks chunks converges to the same average the full sweep would report, at a fraction of the wall time. See run_sweep's docstring"
            arg_type = Int
            default = 128
        "--max-iter"
            help = "maximum number of Newton-Raphson iterations"
            arg_type = Int
            default = MAX_ITER
        "--nthreads"
            help = "parallelize the outer scenario-chunk loop (run_sweep's --batch-size chunks, i.e. the time-series/scenario sweep for one grid, NOT the case_order loop over grids) across this many Julia tasks; CPU backend only, always ignored (forced to 1) on --backend gpu. Requires Julia itself to have been started with at least this many threads (e.g. `julia -t N ...` or JULIA_NUM_THREADS=N) -- this flag only caps how many chunks run concurrently, it cannot add OS threads at runtime"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end

function get_backend(name::String)
    name = lowercase(name)
    if name == "cpu"
        return CPU()
    elseif name in ("gpu", "cuda")
        @assert CUDA.functional() "--backend gpu requested but CUDA.functional() is false"
        return CUDABackend()
    else
        error("Unknown backend $name (expected \"cpu\" or \"gpu\")")
    end
end

to_host(x::AbstractArray) = Array(x)

"""
    format_error(err)

Short, readable message for an exception, tagging out-of-memory errors explicitly so
they're easy to grep for in the result json / console log.
"""
function format_error(err)
    tag = if err isa CUDA.OutOfGPUMemoryError
        "GPU_OOM"
    elseif err isa OutOfMemoryError
        "OOM"
    else
        string(nameof(typeof(err)))
    end
    return "$tag: " * sprint(showerror, err)
end

"""
    is_oom_error(err)

True for an out-of-(GPU-)memory error. `case_order` (see `export_data_for_julia.py` /
`ls_injection.py`'s `all_file_names`) is laid out smallest-grid-first, so once a chunk OOMs
there is no reason to expect a bigger chunk of the same case, or any later (bigger) case in
the suite, to fare any better -- see `run_sweep` and `main`'s use of this to terminate early
instead of burning time re-attempting (and re-failing) on every larger grid.
"""
is_oom_error(err) = err isa CUDA.OutOfGPUMemoryError || err isa OutOfMemoryError

"""
    reclaim!(backend)

Best-effort cleanup after a failed solve so a subsequent chunk/case isn't starved by
memory the failed attempt never released (mainly matters for GPU allocations).
"""
function reclaim!(backend)
    if backend isa CUDABackend
        try
            CUDA.reclaim()
        catch
        end
    end
    GC.gc()
end

"""
    SolveTiming

Full timing breakdown of one `solve_case` call, split to mirror the phases
`ls_injection.py`/`pp_injection.py` report for lightsim2grid/pandapower:

- `build`: pure problem setup (parsing/`load_polar`, `NetworkStack`/(Batch)Jacobian
  construction, the initial structural `jacobian!` call needed to know the Jacobian's
  sparsity pattern, and this call's own load/gen/voltage overrides). Does NOT include the
  linear solver.
- `factorization`: the one-time linear-solver construction -- on CPU this is
  `default_linear_solver` -> `klu(J)`, i.e. KLU's symbolic (analysis/ordering) *and* first
  numeric factorization, done once per `PowerFlowProblem` (so once per case's whole
  scenario batch when `--batch-size 0`, the default -- matches lightsim2grid's own
  "pay Ybus/factor once, amortize over the sweep" convention, see `timer_factor` in
  `ls_injection.py`).
- `solve`: the Newton-Raphson loop itself (`ExaPF.solve!`), further broken down by
  `ConvergenceStatus` into `time_jacobian` (Jacobian (re)evaluation each iteration),
  `time_linear_solver_update` (KLU numeric *re*factorization each iteration, cheaper than
  the initial `factorization` since it reuses the symbolic ordering), and
  `time_linear_solver_ldiv` (the triangular solve itself) -- mirrors lightsim2grid's
  `timer_fillJ`/`timer_refactor`/`timer_solve`.

`build + factorization + solve` reconstructs the same total wall time the old single
`build_time`/`solve_time` split measured; only where each phase's cost is attributed differs.
"""
struct SolveTiming
    build::Float64
    factorization::Float64
    solve::Float64
    time_jacobian::Float64
    time_linear_solver_update::Float64
    time_linear_solver_ldiv::Float64
end

"""
    load_case(casefile, backend)

Parse `casefile` into an ExaPF `PolarForm` (`ExaPF.load_polar`) and nothing else -- pure grid
topology/parameter loading, independent of scenario count and of how the sweep is chunked
(`--batch-size`/`--nthreads`). Returns `(polar, load_time)`. Called exactly once per grid, by
`run_sweep`, before its wall clock starts and before any `build_problem_from_polar` call --
`polar` is then shared (read-only) by every worker's own `build_problem_from_polar`, which
never mutates it (only reads `polar.network`/`polar` itself to build its own private
`stack`/`jac`/`linear_solver`). This is the genuinely fixed, nthreads/batch-size-independent
one-time cost this backend excludes from `wall_time` -- the ExaPF equivalent of every other
backend's own one-time "load the grid" step (`init_from_matpower`, `pypn.load`,
`pp.from_matpower`, `PowerModels.parse_file`). Unlike that parse, Jacobian construction and
KLU factorization (`build_problem_from_polar`) are deliberately NOT excluded -- see the module
docstring for why.
"""
function load_case(casefile, backend)
    t0 = time_ns()
    polar = ExaPF.load_polar(casefile, backend)
    return polar, (time_ns() - t0) / 1e9
end

"""
    build_problem_from_polar(polar, backend, load_p_mw, load_q_mvar; rtol, max_iter)

Build a `:block_polar` (`nblocks>1`) or `:polar` (`nblocks==1`) ExaPF `PowerFlowProblem`
sized for `nblocks = size(load_p_mw, 2)` scenarios, from an already-loaded `polar` (see
`load_case` -- this function never parses a case file itself), using `load_p_mw`/`load_q_mvar`
(real scenario data, not a placeholder -- see below) as the loads for the initial structural
`jacobian!` call the linear solver is then built from. Returns `(prob, baseMVA, build_time,
factorization_time)`. A `prob` built once for a given `nblocks` can then be resolved for other,
different scenarios of that same size by calling `set_scenario_data!` (which overwrites
loads/gen/voltage again) followed by `ExaPF.solve!`, without rebuilding (see `run_sweep`) --
rebuilding would redo the linear solver's KLU symbolic+numeric factorization from scratch every
time, real, size-scaling analysis cost meant to be paid once per grid shape and shared across
every scenario solved against it (see the README's "Full timing breakdown" section), not once
per chunk.

Real loads (not e.g. all-zero placeholders, which was tried first) are required here: unlike
the symbolic-only ordering step, `default_linear_solver` -> `klu(J)` performs a full *numeric*
factorization immediately, at whatever values `J` currently holds -- an all-zero-load
operating point routinely makes that `J` singular (`SingularException`), since a bus with no
injection at all can make the linearization locally degenerate. Passing this chunk's own
already-realistic loads (as ExaPF's own constructor does, and ExaPF's own doc-strings default
argument does, `linear_solver=default_linear_solver(jac.J; nblocks=1)` in `Polar/newton.jl`
implicitly assumes a non-degenerate `J`) avoids that.

Manually replicates `ExaPF.PowerFlowProblem`'s own outer constructor (see ExaPF.jl's
`src/ExaPF.jl`) instead of calling it directly, solely so the linear-solver construction
(KLU factorization) can be timed separately from the rest of problem setup -- ExaPF's public
constructor does not expose that split itself. Reaches into a handful of non-exported ExaPF
internals (`mapping`, `Jacobian`/`BatchJacobian`, `NetworkStack`, `PowerFlowBalance`, `Basis`,
`set_params!`, `jacobian!`, `default_linear_solver`, `NewtonRaphson`, `ConvergenceStatus`) to
do so -- these aren't part of ExaPF's stable public API, so this is fragile against future
ExaPF internal refactors (as opposed to everything else in this file, which only uses
`ExaPF.PowerFlowProblem`/`ExaPF.solve!`/`ExaPF.load_polar` and the other genuinely public
names ExaPF documents/exports).
"""
function build_problem_from_polar(polar, backend, load_p_mw, load_q_mvar; rtol=TOL_PF, max_iter=MAX_ITER)
    nblocks = size(load_p_mw, 2)
    t0 = time_ns()
    baseMVA = polar.network.baseMVA
    mapx = ExaPF.mapping(polar, ExaPF.State())

    if nblocks == 1
        stack = ExaPF.NetworkStack(polar)
        powerflow = ExaPF.PowerFlowBalance(polar) ∘ ExaPF.Basis(polar)
        jac = ExaPF.Jacobian(polar, powerflow, mapx)
        ExaPF.set_params!(jac, stack)
        ExaPF.jacobian!(jac, stack)
    else
        blk_form = ExaPF.BlockPolarForm(polar, nblocks)
        stack = ExaPF.NetworkStack(blk_form)
        powerflow = ExaPF.PowerFlowBalance(blk_form) ∘ ExaPF.Basis(blk_form)
        jac = ExaPF.BatchJacobian(blk_form, powerflow, ExaPF.State())
        ExaPF.set_params!(stack, load_p_mw ./ baseMVA, load_q_mvar ./ baseMVA)
        ExaPF.set_params!(jac, stack)
        ExaPF.jacobian!(jac, stack)
    end
    t_build = time_ns()

    linear_solver = ExaPF.default_linear_solver(jac.J; nblocks=nblocks)
    t_factorization = time_ns()

    nlsolver = ExaPF.NewtonRaphson(tol=rtol, maxiter=max_iter, verbose=0)
    prob = ExaPF.PowerFlowProblem(
        polar, stack, powerflow, linear_solver, nlsolver, mapx, jac,
        ExaPF.ConvergenceStatus(false, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0), backend,
    )
    return prob, baseMVA, (t_build - t0) / 1e9, (t_factorization - t_build) / 1e9
end

"""
    build_problem(casefile, backend, load_p_mw, load_q_mvar; rtol, max_iter)

Convenience wrapper combining `load_case` + `build_problem_from_polar` into one call that also
parses `casefile` itself -- used by `solve_case` (one-off single-scenario solves:
`init_powerflow`/`following_powerflow`/`block_polar_warmup`/the JIT warmup, none of which reuse
a `polar` across multiple calls, so there's nothing to gain from splitting the parse out there).
Returns `(prob, polar, baseMVA, build_time, factorization_time)` -- `build_time` here times
only Jacobian/NetworkStack construction, same as `build_problem_from_polar`; the file parse
itself is not separately timed by this wrapper. `run_sweep`/`process_chunk_group` call
`load_case`/`build_problem_from_polar` directly instead, so every worker shares one parse of
the case file rather than each re-parsing it.
"""
function build_problem(casefile, backend, load_p_mw, load_q_mvar; rtol=TOL_PF, max_iter=MAX_ITER)
    polar, _ = load_case(casefile, backend)
    prob, baseMVA, build_time, factorization_time =
        build_problem_from_polar(polar, backend, load_p_mw, load_q_mvar; rtol=rtol, max_iter=max_iter)
    return prob, polar, baseMVA, build_time, factorization_time
end

"""
    set_scenario_data!(prob, polar, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init, baseMVA, nbus, nscen)

Point an already-built `prob` (from `build_problem`, `nblocks == nscen`) at a specific chunk
of scenario data -- loads, generator dispatch, and the initial voltage guess -- overwriting
whatever was there before (own scenario data, a previous chunk's, or `build_problem`'s zero
placeholder). Cheap (just a handful of `copyto!`s): safe and intended to be called repeatedly
on the same `prob` for successive same-sized chunks, see `run_sweep`. Returns the time spent
(seconds), folded into the caller's own `build`/setup total (not the "one-time per grid shape"
`factorization` cost, which is untouched by this function).
"""
function set_scenario_data!(prob, polar, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init, baseMVA, nbus, nscen)
    t0 = time_ns()
    copyto!(prob.stack.pload, vec(load_p_mw ./ baseMVA))
    copyto!(prob.stack.qload, vec(load_q_mvar ./ baseMVA))
    copyto!(prob.stack.pgen, vec(gen_p_mw ./ baseMVA))

    # Vmag at PV/slack buses is a fixed control setpoint (== matpower Vg), never touched by
    # Newton-Raphson: it is NOT part of the initial guess and must be left as init!() set it
    # from the base case. Overwriting it here (as a naive `stack.vmag .= repeat(vmag0, nscen)`
    # would do) silently corrupts the setpoint and converges to a physically wrong solution
    # (verified: this exact mistake made case14's slack bus solve to |V|=1.0 instead of the
    # correct Vg=1.06). Only PQ buses' vmag is actually a free/state variable, so only those
    # get the (flat/dc/file) initial guess; vang is a free variable everywhere and always safe
    # to overwrite.
    vmag0 = abs.(v_init)
    vang0 = angle.(v_init)
    pq = polar.network.pq
    # `stack.vmag` aliases `stack.input[1:k*nbus]`, so PQ-bus offsets double as `stack.input`
    # indices directly; use the stack-level copyto!(stack, map, vals) (same pattern ExaPF
    # itself uses internally) so the CUDA-specific dispatch (map/vals as CuArray) kicks in
    # instead of a slow/disallowed scalar-indexing fallback on the GPU.
    pq_blocks = vcat([pq .+ (s - 1) * nbus for s in 1:nscen]...)
    vmag_blocks = repeat(vmag0[pq], nscen)
    onGPU = backend isa CUDABackend
    map_arr = onGPU ? CuArray(pq_blocks) : pq_blocks
    val_arr = onGPU ? CuArray(vmag_blocks) : vmag_blocks
    copyto!(prob.stack, map_arr, val_arr)
    copyto!(prob.stack.vang, repeat(vang0, nscen))
    return (time_ns() - t0) / 1e9
end

"""
    solve_case(casefile, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init; rtol, max_iter)

Build a fresh `PowerFlowProblem` for exactly this chunk of scenarios (`build_problem` +
`set_scenario_data!`) and solve it. Returns `(prob, conv, timing::SolveTiming)`. May throw
(OOM, singular Jacobian, ...): callers are expected to catch. Used everywhere a single,
one-off solve is wanted (`init_powerflow`/`following_powerflow`/`block_polar_warmup`, and
`run_sweep` when `--batch-size 0`, the default, so the whole case is exactly one chunk);
`run_sweep` with `--batch-size` set calls `build_problem`/`set_scenario_data!` directly
instead, to reuse one `prob` across multiple same-sized chunks (see its own docstring).
"""
function solve_case(casefile, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init; rtol=TOL_PF, max_iter=MAX_ITER)
    nbus, nscen = size(load_p_mw)
    prob, polar, baseMVA, build_t, factorization_t = build_problem(casefile, backend, load_p_mw, load_q_mvar; rtol=rtol, max_iter=max_iter)
    override_t = set_scenario_data!(prob, polar, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init, baseMVA, nbus, nscen)

    t1 = time_ns()
    conv = ExaPF.solve!(prob)
    t2 = time_ns()

    timing = SolveTiming(
        build_t + override_t, factorization_t, (t2 - t1) / 1e9,
        conv.time_jacobian, conv.time_linear_solver_update, conv.time_linear_solver_ldiv,
    )
    return prob, conv, timing
end

"""
    SweepResult

Aggregated outcome of solve_case over all --batch-size chunks of one case's scenarios --
or, on GPU with --max-chunks > 0, over just the first `n_chunks` of the `n_chunks_total` that
would otherwise have run (see run_sweep). `n_chunks == n_chunks_total` whenever the sweep
wasn't capped (CPU, or GPU with --max-chunks 0 / >= the case's own chunk count).

`load_time` is the one-time, pre-sweep cost of parsing the case file (`load_case`), paid once
regardless of `--batch-size`/`--nthreads`, entirely *before* `wall_time` starts. `build_time`/
`factorization_time`/`modif_time`/`solve_time` are all summed across every chunk actually
solved (one or more per worker) and DO land inside `wall_time` -- see the module docstring for
why Jacobian construction/KLU factorization are treated differently from the file parse here.
"""
struct SweepResult
    V::Union{Nothing,Matrix{ComplexF64}}
    load_time::Float64
    build_time::Float64
    factorization_time::Float64
    modif_time::Float64
    solve_time::Float64
    time_jacobian::Float64
    time_linear_solver_update::Float64
    time_linear_solver_ldiv::Float64
    nb_solved::Int
    n_iterations::Int
    converged::Bool
    n_chunks::Int
    n_chunks_total::Int
    n_failed_chunks::Int
    errors::Vector{String}
    oom::Bool
    diverged_early::Bool
    wall_time::Float64
end

"""
    ChunkGroupResult

Accumulated outcome of solving one contiguous subset of a case's --batch-size chunks -- the
unit `process_chunk_group` returns and `run_sweep` merges (via `merge_chunk_groups`) across
worker tasks when --nthreads > 1. Field meanings mirror `SweepResult` (minus `V`, which workers
instead write into directly, and minus `load_time`, which `run_sweep` pays once, up front,
outside any group -- see `process_chunk_group`).
"""
struct ChunkGroupResult
    build::Float64
    factorization::Float64
    modif::Float64
    solve::Float64
    time_jacobian::Float64
    time_linear_solver_update::Float64
    time_linear_solver_ldiv::Float64
    nb_solved::Int
    n_iterations::Int
    n_attempted::Int
    n_failed::Int
    errors::Vector{String}
    oom::Bool
    diverged_early::Bool
end

"""
    process_chunk_group(polar, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init,
                         chunk_starts, chunk_size, nscen, nbus, V; rtol, max_iter, want_voltages)

Sequentially solve every chunk starting at each of `chunk_starts` (a contiguous subset of the
full chunk-start list -- the whole list when running single-threaded, one --nthreads-th of it
otherwise, see `run_sweep`), reusing one *local* `PowerFlowProblem` across consecutive
same-sized chunks exactly as the original single-threaded sweep did -- rebuilding would redo
KLU's symbolic+numeric factorization from scratch every chunk, i.e. pay real per-grid analysis
cost once per chunk instead of once per distinct chunk size (1, or exceptionally 2, regardless
of how many chunks this group has); see the README's "Full timing breakdown" section. This local
reuse is also *why* parallelizing needs one call per worker rather than sharing a `prob`: ExaPF's
`PowerFlowProblem` is mutated in place by `set_scenario_data!`/`solve!`, so two chunks --
even from different groups -- can never safely share one instance.

`polar` is the whole grid's *single* already-parsed `PolarForm` (`run_sweep`'s own `load_case`
call, done once, before any group is dispatched) -- shared read-only across every worker's own
`build_problem_from_polar` call (see that function's docstring: it only reads `polar`, never
mutates it), so no worker re-parses the case file. Each worker still builds its *own*
`PowerFlowProblem`/`Jacobian`/`linear_solver` from `polar` (once per distinct chunk size it
sees -- normally once, for its own uniform chunk size, per the invariant `split_into_groups`
documents: only the very last chunk overall can be a different, smaller size), and that
Jacobian-construction + KLU-factorization cost is timed into `build`/`factorization` below --
deliberately *inside* the timed dispatch (see the module docstring for why, unlike the one-time
`polar` parse, this is not excluded from `wall_time`).

A chunk that errors is logged and skipped -- its slice of `V` (if given) is left as NaN. A
non-OOM error moves on to the next chunk in this group (forcing a rebuild, in case the failure
left the reused `prob`'s internal state inconsistent); an OOM (`is_oom_error`) instead stops
*this group's* remaining chunks immediately (remaining chunks would only re-attempt the same
over-budget allocation) and sets the returned `oom` flag -- `run_sweep`/`main` still treat any
group hitting OOM as reason to stop the case (and, in `main`, the rest of the bigger-grid suite),
even though -- unlike the single-threaded case -- other groups already in flight are not
interrupted early; see `run_sweep`'s own docstring for that tradeoff.

GPU-only early-exit on repeated non-convergence (no exception, just `has_converged=false`): see
the README's "GPU batched-solve non-convergence at specific --batch-size values" section for the
full investigation, but in short -- on `backend isa CUDABackend`, ExaPF's `nblocks>1` linear
solve is routed through NVIDIA cuDSS's "uniform batch" LU path (still labeled "in preview" by
CUDSS.jl itself), and specific (`grid`, `--batch-size`) combinations have been observed to make
*every* chunk of a grid fail to converge on GPU only. Since this failure mode doesn't recover on
its own within a grid+batch-size combination, `CONSECUTIVE_DIVERGE_LIMIT` consecutive
non-converged chunks (reset to 0 by any converged chunk) trips `diverged_early` and stops the
rest of this group's chunks early. `run_sweep` never actually runs more than one group on GPU
(--nthreads is forced to 1 there, see `main`), so in practice this always means "the rest of
this grid", exactly as before this function existed.
"""
function process_chunk_group(polar, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init,
                              chunk_starts, chunk_size, nscen, nbus, V;
                              rtol=TOL_PF, max_iter=MAX_ITER, want_voltages::Bool=false)
    total_build = 0.0
    total_factorization = 0.0
    total_modif = 0.0
    total_solve = 0.0
    total_time_jacobian = 0.0
    total_time_linear_solver_update = 0.0
    total_time_linear_solver_ldiv = 0.0
    nb_solved = 0
    max_iters = 0
    n_failed = 0
    errors = String[]
    oom_hit = false
    diverged_early = false
    consecutive_diverged = 0
    n_attempted = 0

    prob = nothing
    baseMVA = nothing
    built_nblocks = -1

    for s0 in chunk_starts
        s1 = min(s0 + chunk_size - 1, nscen)
        rng = s0:s1
        this_nblocks = length(rng)
        n_attempted += 1
        try
            if prob === nothing || this_nblocks != built_nblocks
                prob, baseMVA, bt, ft = build_problem_from_polar(
                    polar, backend, load_p_mw[:, rng], load_q_mvar[:, rng]; rtol=rtol, max_iter=max_iter,
                )
                total_build += bt
                total_factorization += ft
                built_nblocks = this_nblocks
            end
            override_t = set_scenario_data!(
                prob, polar, backend,
                load_p_mw[:, rng], load_q_mvar[:, rng], gen_p_mw[:, rng], v_init,
                baseMVA, nbus, this_nblocks,
            )
            total_modif += override_t

            t1 = time_ns()
            conv = ExaPF.solve!(prob)
            total_solve += (time_ns() - t1) / 1e9
            total_time_jacobian += conv.time_jacobian
            total_time_linear_solver_update += conv.time_linear_solver_update
            total_time_linear_solver_ldiv += conv.time_linear_solver_ldiv
            max_iters = max(max_iters, conv.n_iterations)
            if conv.has_converged
                nb_solved += length(rng)
                consecutive_diverged = 0
            else
                n_failed += 1
                consecutive_diverged += 1
                push!(errors, "chunk $rng did not converge")
                println("    WARNING: chunk $rng did not converge")
                if backend isa CUDABackend && consecutive_diverged >= CONSECUTIVE_DIVERGE_LIMIT
                    diverged_early = true
                    println("    DIVERGED: $CONSECUTIVE_DIVERGE_LIMIT consecutive chunks failed to converge on GPU -- skipping the rest of this grid (other grids in the suite are unaffected)")
                    break
                end
            end
            if want_voltages
                vmag = reshape(to_host(ExaPF.get_voltage_magnitude(prob)), nbus, length(rng))
                vang = reshape(to_host(ExaPF.get_voltage_angle(prob)), nbus, length(rng))
                V[:, rng] .= vmag .* exp.(im .* vang)
            end
        catch err
            err isa InterruptException && rethrow()
            n_failed += 1
            msg = format_error(err)
            push!(errors, "chunk $rng: $msg")
            println("    ERROR in chunk $rng: $msg")
            reclaim!(backend)
            prob = nothing  # force a fresh build next chunk rather than reuse a possibly-corrupted one
            built_nblocks = -1
            if is_oom_error(err)
                oom_hit = true
                println("    OOM: stopping this case's sweep early (remaining chunks skipped)")
                break
            end
        end
    end

    return ChunkGroupResult(total_build, total_factorization, total_modif, total_solve,
                             total_time_jacobian, total_time_linear_solver_update, total_time_linear_solver_ldiv,
                             nb_solved, max_iters, n_attempted, n_failed, errors, oom_hit, diverged_early)
end

"""
    split_into_groups(starts, n_workers)

Partition the chunk-start list `starts` into `n_workers` contiguous, roughly-equal groups (any
remainder is spread one-per-group over the first groups) for `run_sweep`'s --nthreads > 1 path.
Contiguous so each worker's own chunks stay as close in size to each other as possible (only the
very last chunk overall can be a different, smaller size) -- keeping each worker's own
`build_problem` reuse (see `process_chunk_group`) paying the one-time factorization cost close to
once per worker, same as the single-threaded loop pays it once overall.
"""
function split_into_groups(starts::Vector{Int}, n_workers::Int)
    n = length(starts)
    n_workers = clamp(n_workers, 1, n)
    base, rem = divrem(n, n_workers)
    groups = Vector{Vector{Int}}(undef, n_workers)
    idx = 1
    for i in 1:n_workers
        len = base + (i <= rem ? 1 : 0)
        groups[i] = starts[idx:idx+len-1]
        idx += len
    end
    return groups
end

"""
    merge_chunk_groups(results::Vector{ChunkGroupResult})

Combine every worker's `ChunkGroupResult` (see `process_chunk_group`) into one, the way
`run_sweep` did inline when it was a single sequential loop: times/counts sum, `n_iterations`
takes the max across groups (a worst-case iteration count, matching what a single sequential
loop tracking `max_iters` across all its chunks would have reported), `errors` concatenate, and
`oom`/`diverged_early` are true if any group hit them.
"""
function merge_chunk_groups(results::Vector{ChunkGroupResult})
    return ChunkGroupResult(
        sum(r.build for r in results),
        sum(r.factorization for r in results),
        sum(r.modif for r in results),
        sum(r.solve for r in results),
        sum(r.time_jacobian for r in results),
        sum(r.time_linear_solver_update for r in results),
        sum(r.time_linear_solver_ldiv for r in results),
        sum(r.nb_solved for r in results),
        maximum(r.n_iterations for r in results),
        sum(r.n_attempted for r in results),
        sum(r.n_failed for r in results),
        reduce(vcat, (r.errors for r in results); init=String[]),
        any(r.oom for r in results),
        any(r.diverged_early for r in results),
    )
end

"""
    run_sweep(casefile, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init, batch_size; ...)

Solve all nscen scenarios, split into chunks of `batch_size` scenarios (or one single chunk if
batch_size <= 0) -- this is the "outer scenario loop" for one grid, as opposed to `main`'s
case_order loop over grids, which always stays sequential.

With the default `nthreads=1` (or on `backend isa CUDABackend`, where it is always forced to 1
-- see `main`), every chunk is solved sequentially by a single call to `process_chunk_group`
over the full chunk-start list, unchanged from before --nthreads existed. With `nthreads > 1` on
CPU, the chunk-start list is instead partitioned into up to `nthreads` contiguous groups
(`split_into_groups`), each solved by its own `Threads.@spawn` task calling
`process_chunk_group` independently (own local `PowerFlowProblem`, see that function's
docstring for why that's required), writing voltages (if requested) into disjoint columns of a
single shared `V` -- safe without locking since no two groups ever touch the same columns --
and the per-group results are combined with `merge_chunk_groups`. See `process_chunk_group` for
how OOM/GPU-divergence bailouts behave per group in the threaded path.

Default chunking (`batch_size <= 0`, i.e. not explicitly set): with `nthreads == 1` this stays
one single chunk covering all `nscen` scenarios, as before `--nthreads` existed. But with
`nthreads > 1`, sizing that one chunk to all of `nscen` regardless would silently defeat
`--nthreads` entirely -- a single chunk means `starts` has length 1, so
`split_into_groups`/`n_workers` below would have nothing to split, forcing every run back to 1
worker no matter what `--nthreads` requested -- and would build one monolithic block-diagonal
`PowerFlowProblem` sized for all `nscen` scenarios at once, whose memory footprint scales with
`nscen` regardless of `nthreads` (observed: OOM-killed by the OS, invisible to
`is_oom_error`/`OutOfMemoryError`, once `nscen` grew large enough on bigger grids). So when
`nthreads > 1` and no explicit `--batch-size` was given, the default chunk size is instead
derived as `ceil(nscen / nthreads)`: exactly `nthreads` (or fewer, for the ragged last chunk)
roughly-equal chunks, one per worker, bounding each worker's own block size to `nscen/nthreads`
instead of all of `nscen`, and making `--nthreads` actually parallelize rather than being a
no-op. An explicit `--batch-size` always takes precedence over this auto-derivation.

`max_chunks` (GPU only, see `--max-chunks`): after the chunk-start list is built, truncated to
its first `max_chunks` entries when `backend isa CUDABackend` and `max_chunks > 0` -- every
chunk of a grid has the same `nblocks` (`batch_size`, bar a ragged last chunk) so per-chunk cost
on GPU is essentially constant, making a prefix's timing average a good estimate of the full
sweep's at a fraction of the wall time. Never applied on CPU, where chunk cost also depends on
`--nthreads` contention and a short prefix is a worse proxy for the full sweep. The untruncated
chunk count is preserved as `n_chunks_total` in the returned `SweepResult` so callers can tell a
capped sweep from a genuinely short one.

Parse-once, then dispatch (wall_time coverage): before anything is timed, the case file is
parsed exactly once (`load_case`) -- the only genuinely fixed, batch-size/nthreads-independent
cost (see the module docstring for why Jacobian construction/KLU factorization are, unlike this
parse, deliberately kept INSIDE the timed dispatch below). The resulting `polar` is then shared
read-only across every worker: each still builds its own private `PowerFlowProblem` (Jacobian +
factorization) via `build_problem_from_polar` inside `process_chunk_group`, since ExaPF's
`PowerFlowProblem` can't be mutated concurrently by two workers (see that function's docstring).
"""
function run_sweep(casefile, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init, batch_size::Int;
                    rtol=TOL_PF, max_iter=MAX_ITER, want_voltages::Bool=false, nthreads::Int=1,
                    max_chunks::Int=0)
    nbus, nscen = size(load_p_mw)
    chunk_size = if batch_size > 0
        min(batch_size, nscen)
    elseif nthreads > 1
        max(1, cld(nscen, nthreads))
    else
        nscen
    end
    starts = collect(1:chunk_size:nscen)
    n_chunks_total = length(starts)
    V = want_voltages ? fill(ComplexF64(NaN, NaN), nbus, nscen) : nothing

    on_gpu = backend isa CUDABackend
    if on_gpu && max_chunks > 0 && length(starts) > max_chunks
        println("    capping sweep to the first $max_chunks/$n_chunks_total chunks (--max-chunks, GPU only)")
        starts = starts[1:max_chunks]
    end
    n_workers = on_gpu ? 1 : clamp(nthreads, 1, length(starts))
    println("    sweep: nscen=$nscen, chunk_size=$chunk_size, n_chunks=$(length(starts))/$n_chunks_total, n_workers=$n_workers")

    # the only cost excluded from wall_time -- see module docstring / this function's own
    polar, load_time = load_case(casefile, backend)

    t_sweep0 = time_ns()
    if n_workers <= 1
        merged = process_chunk_group(polar, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init,
                                      starts, chunk_size, nscen, nbus, V;
                                      rtol=rtol, max_iter=max_iter, want_voltages=want_voltages)
    else
        groups = split_into_groups(starts, n_workers)
        tasks = Vector{Task}(undef, length(groups))
        for (i, g) in enumerate(groups)
            tasks[i] = Threads.@spawn process_chunk_group(
                polar, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init,
                g, chunk_size, nscen, nbus, V;
                rtol=rtol, max_iter=max_iter, want_voltages=want_voltages,
            )
        end
        merged = merge_chunk_groups(fetch.(tasks))
    end
    wall_time = (time_ns() - t_sweep0) / 1e9

    return SweepResult(V, load_time, merged.build, merged.factorization, merged.modif, merged.solve,
                        merged.time_jacobian, merged.time_linear_solver_update, merged.time_linear_solver_ldiv,
                        merged.nb_solved, merged.n_iterations, merged.n_failed == 0,
                        merged.n_attempted, n_chunks_total, merged.n_failed, merged.errors,
                        merged.oom, merged.diverged_early, wall_time)
end

function main()
    println("Parsing command-line arguments...")
    parsed = parse_commandline()
    data_dir = parsed["data-dir"]
    if isempty(data_dir)
        data_dir = joinpath(@__DIR__, "..", "..", "matpower_injection_data",
                             "exapf_data_$(parsed["sample-data-meth"])_$(parsed["nb-pf"])")
        println("--data-dir not given, defaulting to $data_dir")
    end
    backend_name = parsed["backend"]
    results_dir = parsed["results-dir"]
    add_to_name = parsed["add-to-name"]
    save_voltages = parsed["save-voltages"]
    batch_size = parsed["batch-size"]
    max_iter = parsed["max-iter"]
    nthreads = parsed["nthreads"]
    max_chunks = parsed["max-chunks"]
    max_scen = parsed["max-scen"]
    println("Arguments parsed: data-dir=$data_dir backend=$backend_name nthreads=$nthreads batch-size=$batch_size max-chunks=$max_chunks max-scen=$max_scen save-voltages=$save_voltages")

    println("Loading data-dir metadata ($(joinpath(data_dir, "meta.json")))...")
    meta = JSON.parsefile(joinpath(data_dir, "meta.json"))
    ref_path = meta["ref_path"]
    case_order = meta["case_order"]
    sample_data_meth = meta["sample_data_meth"]
    nb_pf = meta["nb_pf"]
    println("Metadata loaded: $(length(case_order)) case(s), nb_pf=$nb_pf, sample_data_meth=$sample_data_meth")

    println("Initializing $backend_name backend...")
    backend = get_backend(backend_name)
    mkpath(results_dir)
    println("Backend ready, proceeding to warmup.")

    # Pin BLAS to 1 thread unconditionally (not just when nthreads > 1): each nthreads task
    # already does its own KLU factorization / BLAS calls, and even at nthreads == 1 leaving
    # BLAS at its own default (typically Sys.CPU_THREADS) would silently use more cores than the
    # nthreads figure being measured suggests, corrupting the single-thread baseline this sweep
    # depends on.
    LinearAlgebra.BLAS.set_num_threads(1)

    # --nthreads only parallelizes the per-grid scenario-chunk sweep (run_sweep), and only on
    # CPU -- see run_sweep/process_chunk_group. Forced to 1 on GPU (one device, and cuDSS's
    # batched solve already parallelizes across scenarios internally).
    if nthreads > 1 && backend isa CUDABackend
        println("--nthreads $nthreads requested but ignored: GPU backend always solves the scenario-chunk sweep sequentially (see run_sweep docstring)")
        nthreads = 1
    elseif nthreads > 1
        println("--nthreads $nthreads: parallelizing the per-case scenario-chunk sweep across $nthreads tasks (BLAS pinned to 1 thread/task); Julia itself is running with $(Threads.nthreads()) thread(s)")
        if Threads.nthreads() < nthreads
            println("  WARNING: Julia was started with only $(Threads.nthreads()) thread(s) -- start it with `julia -t $nthreads ...` (or JULIA_NUM_THREADS=$nthreads) for --nthreads $nthreads to run chunks truly in parallel")
        end
    end

    # --max-chunks only caps the GPU scenario-chunk sweep (run_sweep) -- see its docstring and
    # --max-chunks's help. Ignored on CPU, where per-chunk cost isn't as uniform across a grid
    # (depends on --nthreads contention too) so a short prefix is a worse proxy for the full sweep.
    if max_chunks > 0 && !(backend isa CUDABackend)
        println("--max-chunks $max_chunks requested but ignored: only applies to --backend gpu")
    end

    base_name = "exapf_injection_$(sample_data_meth)_$(nb_pf)_$(backend_name)"
    if !isempty(add_to_name)
        base_name *= add_to_name
    end
    complete_full_path = joinpath(results_dir, "$(base_name).json")
    full_path_Vs = joinpath(results_dir, "$(base_name)_{}_Vs.npy")

    # pay the JIT / GPU-kernel-compilation cost once, on the smallest available case,
    # before starting the timed loop (mirrors "warm up OLF" in olf_injection.py). A
    # warmup failure is not fatal (it only means the first real case pays the JIT cost).
    warmup_case = isempty(parsed["warmup-case"]) ? first(case_order) : parsed["warmup-case"]
    warmup_stem = first(splitext(warmup_case))
    println("Warming up on $warmup_case (paying JIT/GPU-compile cost before timing starts)...")
    try
        println("  loading warmup scenario data...")
        wl_p = npzread(joinpath(data_dir, "$(warmup_stem)_load_p_mw.npy"))
        wl_q = npzread(joinpath(data_dir, "$(warmup_stem)_load_q_mvar.npy"))
        wg_p = npzread(joinpath(data_dir, "$(warmup_stem)_gen_p_mw.npy"))
        wv_i = npzread(joinpath(data_dir, "$(warmup_stem)_v_init.npy"))
        println("  warmup data loaded, running :polar warmup solve...")
        solve_case(joinpath(ref_path, warmup_case), backend, wl_p[:, 1:1], wl_q[:, 1:1], wg_p[:, 1:1], wv_i; max_iter=max_iter)
        nscen_w = size(wl_p, 2)
        if nscen_w > 1
            println("  :polar warmup done, running :block_polar warmup solve...")
            solve_case(joinpath(ref_path, warmup_case), backend, wl_p, wl_q, wg_p, wv_i; max_iter=max_iter)

            # solve_case() above only JIT-compiles ExaPF's own :polar/:block_polar solve path
            # (build_problem/set_scenario_data!/ExaPF.solve!). run_sweep() and everything it
            # calls (process_chunk_group, split_into_groups, merge_chunk_groups, and the
            # Threads.@spawn dispatch itself when nthreads > 1) are never exercised by
            # solve_case() -- so without this, THEIR first-ever JIT compile lands on whichever
            # grid this process happens to solve first, inflating that grid's own wall_time by
            # up to ~100x versus its total_time (confirmed on case14.m: total_time=0.0072s but
            # wall_time=0.723s for the exact same 256-scenario sweep -- total_time only sums
            # time_ns() calls *inside* those functions' bodies, so compilation, which happens
            # before a function body ever starts, is invisible to it but fully counted in
            # wall_time). Every other grid in that same run showed wall_time == total_time,
            # confirming this is a one-shot compile tax, not a per-grid cost. Previously this
            # only ever hit case14.m (case_order's first, smallest grid) when many grids shared
            # one process; now that base_launch_exapf.sh runs one grid per Julia process
            # (--case), EVERY run's one-and-only grid would otherwise be "first" and pay this
            # tax -- hence warming up run_sweep here, unconditionally, for every backend/nthreads
            # combination, before any case's own sweep is timed.
            println("  :block_polar warmup done, running run_sweep warmup (JIT-compiles process_chunk_group/split_into_groups/merge_chunk_groups before any case is timed)...")
            run_sweep(joinpath(ref_path, warmup_case), backend, wl_p[:, 1:min(2, nscen_w)], wl_q[:, 1:min(2, nscen_w)],
                      wg_p[:, 1:min(2, nscen_w)], wv_i, 0; max_iter=max_iter, want_voltages=false,
                      nthreads=nthreads, max_chunks=max_chunks)
        end
        println("Warmup complete, proceeding to the case sweep.")
    catch err
        err isa InterruptException && rethrow()
        println("  WARNING: warmup failed (continuing anyway): ", format_error(err))
        reclaim!(backend)
    end

    benchmark_results = Dict{String,Any}()

    for (case_idx, case_fn) in enumerate(case_order)
        case_stem = first(splitext(case_fn))
        casefile = joinpath(ref_path, case_fn)
        println("=== [$case_idx/$(length(case_order))] $case_fn ($backend_name) ===")

        try
            println("  loading scenario data (load_p/load_q/gen_p/v_init)...")
            load_p_mw = npzread(joinpath(data_dir, "$(case_stem)_load_p_mw.npy"))
            load_q_mvar = npzread(joinpath(data_dir, "$(case_stem)_load_q_mvar.npy"))
            gen_p_mw = npzread(joinpath(data_dir, "$(case_stem)_gen_p_mw.npy"))
            v_init = npzread(joinpath(data_dir, "$(case_stem)_v_init.npy"))

            nbus_data, nscen_avail = size(load_p_mw)
            nscen = max_scen > 0 ? min(max_scen, nscen_avail) : nscen_avail
            if nscen < nscen_avail
                load_p_mw = load_p_mw[:, 1:nscen]
                load_q_mvar = load_q_mvar[:, 1:nscen]
                gen_p_mw = gen_p_mw[:, 1:nscen]
            end
            ngen_data = size(gen_p_mw, 1)
            println("  data loaded (nbus=$nbus_data, ngen=$ngen_data, nscen=$nscen/$nscen_avail), checking shapes against ExaPF's own parser...")

            polar_check = ExaPF.load_polar(casefile, backend)
            nbus = PS.get(polar_check.network, PS.NumberOfBuses())
            ngen = PS.get(polar_check.network, PS.NumberOfGenerators())
            if nbus != nbus_data || ngen != ngen_data
                println("  SKIP: shape mismatch (data: nbus=$nbus_data ngen=$ngen_data, ExaPF: nbus=$nbus ngen=$ngen)")
                benchmark_results[case_fn] = Dict("error" => "shape mismatch (data: nbus=$nbus_data ngen=$ngen_data, ExaPF: nbus=$nbus ngen=$ngen)")
                open(complete_full_path, "w") do f
                    JSON.print(f, benchmark_results, 2)
                end
                continue
            end
            println("  shapes OK, proceeding to init/following single-scenario solves...")

            # first + second single-scenario solve, mirroring ls_injection.py's warm-then-time pattern
            _, conv1, timing1 = solve_case(casefile, backend, load_p_mw[:, 1:1], load_q_mvar[:, 1:1], gen_p_mw[:, 1:1], v_init; max_iter=max_iter)
            _, conv2, timing2 = solve_case(casefile, backend, load_p_mw[:, 1:1], load_q_mvar[:, 1:1], gen_p_mw[:, 1:1], v_init; max_iter=max_iter)
            println("  init/following solves done (converged=$(conv1.has_converged)/$(conv2.has_converged)), proceeding to block_polar warmup + sweep...")

            timing_dict(t, conv) = Dict{String,Any}(
                "build_time" => t.build,
                "factorization_time" => t.factorization,
                "solve_time" => t.solve,
                "time_jacobian" => t.time_jacobian,
                "time_linear_solver_update" => t.time_linear_solver_update,
                "time_linear_solver_ldiv" => t.time_linear_solver_ldiv,
                "converged" => conv.has_converged, "n_iterations" => conv.n_iterations,
            )

            tmp_res = Dict{String,Any}(
                "grid_size" => Dict("total_bus" => nbus, "total_gen" => ngen),
                "init_powerflow" => timing_dict(timing1, conv1),
                "following_powerflow" => timing_dict(timing2, conv2),
            )

            # :block_polar-specific warmup: the single-scenario solves above only exercise the
            # :polar path (nscen==1). :block_polar (nscen>1, what the timed sweep below actually
            # uses) is a different ExaPF method, and its Jacobian coloring count is baked into a
            # Julia *type parameter* (ForwardDiff.Dual{Nothing,Float64,N}), so the first
            # :block_polar call for THIS grid's own coloring count N triggers a fresh, grid-specific
            # JIT compile of the whole seeding/Jacobian/KLU/Newton pipeline (confirmed via
            # --trace-compile: ~15-19 newly compiled methods per never-before-seen N, gone on a
            # repeat call with the same N) -- on top of, and roughly independent of, the real
            # per-grid analysis cost (KLU symbolic/numeric factorization, Jacobian coloring itself),
            # which genuinely scales with grid size and must stay attributed to the timed sweep's
            # own build_time/factorization_time below (time_series), not conflated with this
            # warmup's JIT-compile cost. 2 scenarios is enough to pay the compile tax (coloring/type
            # depend on the grid, not on nscen) without meaningfully pre-paying the real,
            # size-scaling build/factorization cost the timed sweep below is meant to capture.
            if nscen >= 2
                _, conv_w, timing_w = solve_case(casefile, backend, load_p_mw[:, 1:2], load_q_mvar[:, 1:2], gen_p_mw[:, 1:2], v_init; max_iter=max_iter)
                tmp_res["block_polar_warmup"] = timing_dict(timing_w, conv_w)
                println("  block_polar warmup done (converged=$(conv_w.has_converged)), starting scenario sweep...")
            else
                println("  starting scenario sweep...")
            end

            # batched injection sweep: nscen scenarios, solved --batch-size at a time.
            # run_sweep() parses the case file exactly once (load_case) before starting its own
            # internal wall clock -- the only genuinely fixed, batch-size/nthreads-independent
            # cost, reported separately below as grid_load.load_time. Jacobian construction and
            # KLU factorization, by contrast, are deliberately kept INSIDE sweep.wall_time/
            # total_time (see the module docstring for why: they scale with --nthreads, unlike a
            # one-time grid load). build_time/factorization_time/modif_time/solve_time (and their
            # sum, total_time) are, with --nthreads > 1, each summed across concurrently-running
            # worker tasks (see merge_chunk_groups), i.e. aggregate CPU-time, not elapsed time --
            # using that for pf/s throughput would understate a threaded run's real speedup; use
            # sweep.wall_time for that instead (see below).
            sweep = run_sweep(casefile, backend, load_p_mw, load_q_mvar, gen_p_mw, v_init, batch_size;
                               max_iter=max_iter, want_voltages=save_voltages, nthreads=nthreads, max_chunks=max_chunks)
            if !sweep.converged
                println("  WARNING: $(sweep.n_failed_chunks)/$(sweep.n_chunks) chunk(s) failed ($(sweep.nb_solved)/$nscen scenarios solved)")
            end
            if sweep.diverged_early
                println("  DIVERGED: $case_fn hit $CONSECUTIVE_DIVERGE_LIMIT consecutive non-converged chunks on GPU at --batch-size $(batch_size <= 0 ? nscen : batch_size) -- rest of this grid skipped, continuing with the rest of the suite")
            end
            chunks_str = sweep.n_chunks < sweep.n_chunks_total ? "$(sweep.n_chunks)/$(sweep.n_chunks_total) (capped)" : "$(sweep.n_chunks)"
            total_time = sweep.build_time + sweep.factorization_time + sweep.modif_time + sweep.solve_time
            @printf("  load: %.3fs  build: %.3fs  factorize: %.3fs  modif: %.3fs  solve: %.3fs  wall: %.3fs  (%.1f pf/s)  solved: %d/%d  chunks: %s (%d failed)\n",
                    sweep.load_time, sweep.build_time, sweep.factorization_time, sweep.modif_time, sweep.solve_time, sweep.wall_time,
                    sweep.nb_solved / max(sweep.wall_time, 1e-9),
                    sweep.nb_solved, nscen, chunks_str, sweep.n_failed_chunks)

            # one-time, pre-sweep cost of parsing the case file (load_case) -- the only thing
            # excluded from time_series.wall_time/total_time; kept as a sibling field, same role
            # as e.g. powermodels_injection.jl's "base_case_powerflow" or
            # rustpower_injection.py's "grid_size"/"v_init_method"
            tmp_res["grid_load"] = Dict{String,Any}("load_time" => sweep.load_time)

            tmp_res["time_series"] = Dict{String,Any}(
                # build_time + factorization_time + modif_time + solve_time -- unlike
                # olf_injection.py/pp_injection.py/rustpower_injection.py/
                # powermodels_injection.jl's own total_time, this includes Jacobian
                # construction/KLU factorization on purpose (see module docstring)
                "total_time" => total_time,
                "build_time" => sweep.build_time,
                "factorization_time" => sweep.factorization_time,
                "modif_time" => sweep.modif_time,
                "solver_time" => sweep.solve_time,
                "wall_time" => sweep.wall_time,
                "nb_solved" => sweep.nb_solved,
                "nb_total" => nscen,
                "time_jacobian" => sweep.time_jacobian,
                "time_linear_solver_update" => sweep.time_linear_solver_update,
                "time_linear_solver_ldiv" => sweep.time_linear_solver_ldiv,
                "n_iterations" => sweep.n_iterations,
                "batch_size" => batch_size <= 0 ? nscen : batch_size,
                "nthreads" => nthreads,
                "n_chunks" => sweep.n_chunks,
                "n_chunks_total" => sweep.n_chunks_total,
                "chunks_capped" => sweep.n_chunks < sweep.n_chunks_total,
                "n_failed_chunks" => sweep.n_failed_chunks,
                "converged" => sweep.converged,
                "diverged_early" => sweep.diverged_early,
            )
            if !isempty(sweep.errors)
                tmp_res["time_series"]["errors"] = sweep.errors
            end
            benchmark_results[case_fn] = tmp_res

            open(complete_full_path, "w") do f
                JSON.print(f, benchmark_results, 2)
            end

            if save_voltages && sweep.V !== nothing
                out_path = replace(full_path_Vs, "{}" => case_stem)
                npzwrite(out_path, permutedims(sweep.V, (2, 1)))  # (nbus, nscen) -> (nscen, nbus)
            end

            if sweep.oom
                println("  OOM on $case_fn -- stopping early, skipping the remaining (bigger) grids in the suite")
                break
            end
        catch err
            err isa InterruptException && rethrow()
            msg = format_error(err)
            println("  ERROR: failed to process $case_fn: $msg")
            benchmark_results[case_fn] = Dict("error" => msg)
            reclaim!(backend)
            open(complete_full_path, "w") do f
                JSON.print(f, benchmark_results, 2)
            end
            if is_oom_error(err)
                println("  OOM on $case_fn -- stopping early, skipping the remaining (bigger) grids in the suite")
                break
            end
        end
    end

    println("Done. Results written to $complete_full_path")
end

main()

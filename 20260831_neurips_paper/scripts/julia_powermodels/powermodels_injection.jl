
#!/usr/bin/env julia

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Julia/PowerModels.jl counterpart to ls_injection.py / olf_injection.py / pp_injection.py /
# julia_exapf/exapf_injection.jl: runs the same injection-sweep scenarios (exported by
# ../export_data_for_julia.py) through PowerModels.jl's native AC powerflow solver
# (PowerModels.compute_ac_pf, a pure NLsolve-based Newton solver) by default, and writes out
# benchmark timings (json) and, optionally, complex bus voltages (npy) in the same
# layout/convention as the other tools. Two options address the "known caveat" below:
#   warm-starting each scenario (opt-out via --no-warm-start-scenarios; warm start is on by default)
#     seeds every scenario's Newton start point from a single fixed anchor: the powerflow solve of
#     the case's *base case* (its own vm/vd/pg/qg, i.e. `data` exactly as parsed, before any
#     scenario's injections are applied -- computed once per case, right after loading, and
#     reused for every scenario in that case's sweep) instead of a flat 1pu/0rad guess.
#   --use-ipopt (opt-in; off by default) switches the solve itself to the JuMP+Ipopt
#     solve_ac_pf path -- an interior-point NLP solve of the same equations, not exposed to
#     NLsolve's basin-of-attraction behavior the same way, but far slower per scenario (builds
#     a fresh optimization model every solve).
#
# Base-case starting point: chosen automatically per case via get_init_method.jl (a Julia port
# of ../get_init_method.py, same per-case table -- kept as a separate script rather than a CLI
# flag for the same reason the python side does it that way: which of flat/dc/file makes the
# base-case powerflow converge is a property of the grid, not something to pick per benchmark
# run). "flat" solves from 1pu/0rad; "file" solves from the case file's own vm_start/va_start
# (see apply_voltage_start!); "dc" runs PowerModels.compute_dc_pf once first and seeds
# vm_start=1.0 (DC PF doesn't solve voltage magnitude)/va_start from its angles (see
# apply_dc_start!). Only the base-case solve and the init/following_powerflow repeat-solve
# pair are seeded this way -- once warm-starting is on (the default), every scenario in the
# sweep instead starts from the already-converged base-case solution regardless of init
# method, per the warm-starting paragraph above.
#
# Usage:
#   python ../export_data_for_julia.py --sample_data_meth sable --seed 0 --nb_pf 1000
#   julia --project=. powermodels_injection.jl \
#       --data-dir ../../matpower_injection_data/exapf_data_sable_1000 --save-voltages
#   julia --project=. -t 8 powermodels_injection.jl \
#       --data-dir ../../matpower_injection_data/exapf_data_sable_1000 --nthreads 8
#
# --nthreads: parallelizes the outer scenario loop -- the per-case `for scen in 1:nscen` sweep
# over one grid's own scenarios -- across Julia tasks (see solve_scenario_group). Requires
# launching Julia itself with at least as many threads (`julia -t N` / JULIA_NUM_THREADS=N); it
# never parallelizes the case_order loop over grids (each grid is still processed one at a
# time), and is forced to 1 whenever --use-ipopt is set (Ipopt's C library isn't safe to call
# concurrently from multiple threads).
#
# Element ordering / bus-ID alignment: unlike ExaPF (which, like lightsim2grid, indexes buses
# and generators by their raw matpower file row position -- see export_data_for_julia.py's module
# docstring), PowerModels keys buses (and everything derived from a bus id, i.e. loads) by the
# ORIGINAL matpower bus number (the BUS_I column), which is NOT always the row position (e.g.
# case6515rte.m has gaps: 6515 buses, max id 6519; case3375wp.m: 3374 buses, max id 10369. 
# Generators are,
# however, still indexed by matpower row order (PowerModels assigns gen "index" = row-order
# counter, 1-based, for every gen row including disconnected ones) -- exactly ExaPF's/
# lightsim2grid's own convention, so gen alignment is a straight positional lookup on the
# *active* (gen_status != 0) rows, same principle export_data_for_julia.py already documents for
# ExaPF. Bus/load alignment instead needs the row->bus_id map, which this script recovers
# directly from the raw .m file via PowerModels._IM.parse_matlab_file (the same unprocessed
# row list PowerModels' own matpower parser builds the bus dict from), rather than assuming
# row position and bus id coincide.
#
# Known caveat, confirmed by direct comparison against other solvers on the same scenarios: 
# on a handful of grids (confirmed so far on
# case2869pegase.m, case_ACTIVSg25k.m) a small number of degree-1 buses carrying a near-zero
# base load converge to a spurious near-zero-voltage root (|V| ~ 1e-40) instead of the
# physical ~1pu solution, while `termination_status` still reports converged (the ftol
# criterion is satisfied there too -- it is a genuine second root of the power-balance
# equations, not a numerical-tolerance artifact). This is a property of the
# reference tool, not of this script's element alignment (which the case14/case57/
# case1354pegase machine-precision matches above rule out as the cause): treat per-scenario
# voltage output as unreliable on a given grid unless spot-checked against another backend,
# and don't be surprised if `nb_solved` for a grid looks fully converged while a few of its
# buses are physically wrong.
#
# INFO / WARNING: this script has been mainly generated by Claude (sonnet 5), carefully
# reviewed by the authors of the paper. Results have been "sanity checked" when compared with 
# other methods (mainly lightsim2grid).

import Pkg
Pkg.activate(@__DIR__)

# Pin every implicit (non-Julia-Threads) thread pool to 1 before any package is loaded, so
# libraries that read these once at __init__ time (e.g. Ipopt's underlying MUMPS/Ipopt_jll,
# when --use-ipopt is set) still see them -- see main()'s own LinearAlgebra.BLAS.set_num_threads(1)
# for the same reasoning applied to BLAS (set at runtime there instead, since BLAS.set_num_threads
# is a real function call, not just an env read).
ENV["OMP_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"

using ArgParse
using PowerModels
using JuMP
using Ipopt
using NPZ
using JSON
using Printf
using Dates
using LinearAlgebra
using Base.Threads

include(joinpath(@__DIR__, "get_init_method.jl"))

const TOL_PF = 1e-8
const MAX_ITER = 100

PowerModels.silence()  # this script cares about the solve, not PowerModels' own INFO/WARN logs

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
            help = "solve only the first N scenarios of each case (0 = all available in --data-dir); use this for a quick smoke test without re-running export_data_for_julia.py with a smaller --nb_pf"
            arg_type = Int
            default = 0
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
            help = "case file (from the data dir's own case list) used to pay Julia's JIT compile cost before timing starts"
            arg_type = String
            default = ""
        "--max-iter"
            help = "maximum number of Newton-Raphson (NLsolve) iterations"
            arg_type = Int
            default = MAX_ITER
        "--tol"
            help = "convergence tolerance: NLsolve ftol by default, or Ipopt's \"tol\" option when --use-ipopt is set (matching TOL_PF elsewhere in this benchmark)"
            arg_type = Float64
            default = TOL_PF
        "--no-warm-start-scenarios"
            help = "opt out of warm-starting every scenario's solve from the case's base-case powerflow solution (the default; see module docstring). When set, every scenario is instead initialized the same way, per the case's automatic init method (see get_init_method.jl), matching this script's original behavior"
            action = :store_true
        "--use-ipopt"
            help = "solve with the JuMP+Ipopt-based PowerModels.solve_ac_pf instead of the default NLsolve-based compute_ac_pf; much more robust to the spurious near-zero-voltage root documented in the module docstring, but far slower per scenario (fresh optimization model every solve) -- meant for a correctness spot-check, not a full sweep"
            action = :store_true
        "--nthreads"
            help = "parallelize the outer scenario loop (the per-case `for scen in 1:nscen` sweep, i.e. the time-series/scenario sweep for one grid, NOT the case_order loop over grids, which always stays sequential) across this many Julia tasks. Requires Julia itself to have been started with at least this many threads (e.g. `julia -t N ...` or JULIA_NUM_THREADS=N). Forced to 1 when --use-ipopt is set (Ipopt's C library is not safe to call concurrently from multiple threads)"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end

"short, readable message for an exception"
function format_error(err)
    return string(nameof(typeof(err))) * ": " * sprint(showerror, err)
end

"""
    row_order_bus_ids(casefile)

Original matpower BUS_I values, in raw file row order (position i (1-based) -> bus id).
PowerModels itself only exposes bus data keyed by bus id (see module docstring above), so
this is recovered independently from the same file via the unprocessed row list
(`PowerModels._IM.parse_matlab_file`), matching the row order lightsim2grid / ExaPF use.
"""
function row_order_bus_ids(casefile)
    raw = PowerModels._IM.parse_matlab_file(casefile)
    return Int[Int(row[1]) for row in raw["mpc.bus"]]
end

"""
    active_gen_row_indices(data)

PowerModels gen keys ("1", "2", ...) whose gen_status != 0, in ascending (= matpower row)
order -- the same rows/order as export_data_for_julia.py's gen_p_mw columns (see its module
docstring: "ExaPF generator index i == matpower gen table row i restricted to gen_status >
0"; PowerModels uses the identical row-order convention for gen "index").
"""
function active_gen_row_indices(data)
    n = length(data["gen"])
    return [k for k in 1:n if data["gen"]["$k"]["gen_status"] != 0]
end

"""
    apply_voltage_start!(data, sol_bus)

Seed every bus's "vm_start"/"va_start" in `data` from a previously-solved scenario's solution
(`sol_bus == result["solution"]["bus"]`) -- in this script, the case's base-case powerflow,
computed once in `main` and reused as a fixed warm-start anchor for every scenario. Both
compute_ac_pf's own flat_start=false warm-start path and solve_ac_pf/solve_pf's JuMP variable
start values read these same "*_start" keys (see PowerModels.comp_start_value and
_compute_ac_pf's warm-start section in PowerModels/src/prob/pf.jl), so this one function
warm-starts either solver.
"""
function apply_voltage_start!(data, sol_bus)
    for (bid, bus_sol) in sol_bus
        bus = get(data["bus"], bid, nothing)
        bus === nothing && continue
        bus["vm_start"] = bus_sol["vm"]
        bus["va_start"] = bus_sol["va"]
    end
end

"""
    apply_dc_start!(data, dc_sol_bus)

Seed every bus's "vm_start"/"va_start" in `data` from a `PowerModels.compute_dc_pf` solution
(`dc_sol_bus == result["solution"]["bus"]`), for the "dc" branch of get_init_method.jl. DC
powerflow only solves bus angles -- it assumes a flat 1.0pu voltage-magnitude profile
throughout (see compute_dc_pf's docstring in PowerModels/src/prob/pf.jl) -- so vm_start is
hardcoded to 1.0 here rather than read from the DC solution.
"""
function apply_dc_start!(data, dc_sol_bus)
    for (bid, bus_sol) in dc_sol_bus
        bus = get(data["bus"], bid, nothing)
        bus === nothing && continue
        bus["vm_start"] = 1.0
        bus["va_start"] = bus_sol["va"]
    end
end

"true if a JuMP/MOI termination status counts as converged"
function ipopt_converged(status)
    return status == JuMP.OPTIMAL || status == JuMP.LOCALLY_SOLVED || status == JuMP.ALMOST_OPTIMAL || status == JuMP.ALMOST_LOCALLY_SOLVED
end

"""
    solve_scenario(data, baseMVA, bus_id_to_pos, active_gen_idx, load_p_bus, load_q_bus,
                   gen_p_active, scen; flat_start, max_iter, tol, warm_start_sol, ipopt_optimizer)

Overwrite `data`'s loads/gens in place for scenario `scen` (columns are 1-based, matching
Julia's npzread arrays), solve one AC powerflow, and return
`(converged::Bool, dt_modif, dt_solve, result)`. `dt_modif` times the load/gen dict mutation
above plus (when given) the warm-start voltage seeding -- this project's other backends call
this step "modif" (see e.g. olf_injection.py's `update_loads`/`update_generators` timing) --
and `dt_solve` times only the `compute_ac_pf`/`solve_ac_pf` call itself, mirroring
olf_injection.py's own `total_modif`/`total_pf` split.

If `warm_start_sol` (a fixed `result["solution"]`, e.g. the case's base-case solve) is given,
its bus voltages are used as the Newton/Ipopt start point instead of `flat_start`. If
`ipopt_optimizer` is given, the JuMP+Ipopt `solve_ac_pf` path is used instead of
`compute_ac_pf`.
"""
function solve_scenario(data, baseMVA, bus_id_to_pos, active_gen_idx,
                         load_p_bus, load_q_bus, gen_p_active, scen;
                         flat_start=true, max_iter=MAX_ITER, tol=TOL_PF,
                         warm_start_sol=nothing, ipopt_optimizer=nothing)
    t0 = time_ns()
    for (_, load) in data["load"]
        pos = bus_id_to_pos[load["load_bus"]]
        load["pd"] = load_p_bus[pos, scen] / baseMVA
        load["qd"] = load_q_bus[pos, scen] / baseMVA
    end
    for (j, k) in enumerate(active_gen_idx)
        data["gen"]["$k"]["pg"] = gen_p_active[j, scen] / baseMVA
    end

    if warm_start_sol !== nothing
        apply_voltage_start!(data, warm_start_sol["bus"])
        flat_start = false
    end
    t1 = time_ns()

    if ipopt_optimizer === nothing
        result = PowerModels.compute_ac_pf(data; flat_start=flat_start, iterations=max_iter, ftol=tol)
        converged = result["termination_status"]
    else
        result = PowerModels.solve_ac_pf(data, ipopt_optimizer)
        converged = ipopt_converged(result["termination_status"])
    end
    t2 = time_ns()
    dt_modif = (t1 - t0) / 1e9
    dt_solve = (t2 - t1) / 1e9
    return converged, dt_modif, dt_solve, result
end

"""
    split_range(n, n_workers)

Partition `1:n` into `n_workers` contiguous, roughly-equal `UnitRange`s (any remainder is spread
one-per-group over the first ranges) -- used by `main` to split one case's per-scenario loop
(`1:nscen`) across --nthreads worker tasks.
"""
function split_range(n::Int, n_workers::Int)
    n_workers = clamp(n_workers, 1, n)
    base, rem = divrem(n, n_workers)
    ranges = Vector{UnitRange{Int}}(undef, n_workers)
    idx = 1
    for i in 1:n_workers
        len = base + (i <= rem ? 1 : 0)
        ranges[i] = idx:(idx + len - 1)
        idx += len
    end
    return ranges
end

"""
    solve_scenario_group(data, baseMVA, bus_id_to_pos, active_gen_idx, load_p_bus, load_q_bus,
                         gen_p_active, scens, bus_ids, V; flat_start, max_iter, tol,
                         warm_start_sol, ipopt_optimizer)

Sequentially solve every scenario in `scens` (a contiguous subset of one case's `1:nscen` -- the
whole range when running single-threaded, one --nthreads-th of it otherwise, see `main`) against
its own `data`. `data` must already be private to this call -- either the case's own live `data`
(single-threaded: no other task touches it concurrently) or a `deepcopy` the caller made just for
this group (--nthreads > 1: required because `solve_scenario` mutates `data`'s "load"/"gen"
values in place every call, and PowerModels' dict-of-dicts case representation is not otherwise
safe to share/mutate across concurrent tasks). Writes solved voltages into `V`'s rows for this
group's own scenarios when given (disjoint from every other group's rows, so this needs no
locking). Returns `(total_modif, total_solver_time, nb_solved, errors::Vector{String})` for
`main` to merge across groups -- `total_modif` sums each scenario's `dt_modif`, `total_solver_time`
sums each scenario's `dt_solve` (see `solve_scenario`), matching olf_injection.py's own
`total_modif`/`total_pf` split.
"""
function solve_scenario_group(data, baseMVA, bus_id_to_pos, active_gen_idx,
                               load_p_bus, load_q_bus, gen_p_active, scens, bus_ids, V;
                               flat_start=true, max_iter=MAX_ITER, tol=TOL_PF,
                               warm_start_sol=nothing, ipopt_optimizer=nothing)
    total_modif = 0.0
    total_solver_time = 0.0
    nb_solved = 0
    errors = String[]
    for scen in scens
        conv, dt_modif, dt_solve, result = solve_scenario(data, baseMVA, bus_id_to_pos, active_gen_idx,
                                           load_p_bus, load_q_bus, gen_p_active, scen;
                                           flat_start=flat_start, max_iter=max_iter, tol=tol,
                                           warm_start_sol=warm_start_sol, ipopt_optimizer=ipopt_optimizer)
        total_modif += dt_modif
        total_solver_time += dt_solve
        if conv
            nb_solved += 1
            if V !== nothing
                sol_bus = result["solution"]["bus"]
                for (i, bid) in enumerate(bus_ids)
                    bus_sol = get(sol_bus, "$bid", nothing)
                    if bus_sol !== nothing
                        V[scen, i] = bus_sol["vm"] * exp(im * bus_sol["va"])
                    end
                end
            end
        else
            push!(errors, "scenario $scen did not converge")
        end
    end
    return total_modif, total_solver_time, nb_solved, errors
end

function main()
    parsed = parse_commandline()
    data_dir = parsed["data-dir"]
    if isempty(data_dir)
        data_dir = joinpath(@__DIR__, "..", "..", "matpower_injection_data",
                             "exapf_data_$(parsed["sample-data-meth"])_$(parsed["nb-pf"])")
        println("--data-dir not given, defaulting to $data_dir")
    end
    results_dir = parsed["results-dir"]
    add_to_name = parsed["add-to-name"]
    save_voltages = parsed["save-voltages"]
    max_scen = parsed["max-scen"]
    max_iter = parsed["max-iter"]
    tol = parsed["tol"]
    warm_start_scenarios = !parsed["no-warm-start-scenarios"]
    use_ipopt = parsed["use-ipopt"]
    ipopt_optimizer = use_ipopt ? JuMP.optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes", "max_iter" => max_iter, "tol" => tol) : nothing
    nthreads = parsed["nthreads"]

    # Pin BLAS to 1 thread unconditionally (not just when nthreads > 1): each nthreads task
    # already does its own linear algebra, and even at nthreads == 1 leaving BLAS at its own
    # default (typically Sys.CPU_THREADS) would silently use more cores than the nthreads figure
    # being measured suggests, corrupting the single-thread baseline this sweep depends on.
    LinearAlgebra.BLAS.set_num_threads(1)

    # --nthreads only parallelizes the per-grid scenario loop (see solve_scenario_group), never
    # the case_order loop over grids. Forced to 1 with --use-ipopt: Ipopt's underlying C library
    # is not safe to call concurrently from multiple threads.
    if nthreads > 1 && use_ipopt
        println("--nthreads $nthreads requested with --use-ipopt: forcing --nthreads 1 (Ipopt is not safe to call concurrently from multiple threads)")
        nthreads = 1
    elseif nthreads > 1
        println("--nthreads $nthreads: parallelizing the per-case scenario loop across $nthreads tasks (BLAS pinned to 1 thread/task); Julia itself is running with $(Threads.nthreads()) thread(s)")
        if Threads.nthreads() < nthreads
            println("  WARNING: Julia was started with only $(Threads.nthreads()) thread(s) -- start it with `julia -t $nthreads ...` (or JULIA_NUM_THREADS=$nthreads) for --nthreads $nthreads to run scenarios truly in parallel")
        end
    end

    println("Starting powermodels_injection.jl (", Dates.now(), ") -- warm_start_scenarios=$warm_start_scenarios use_ipopt=$use_ipopt nthreads=$nthreads")

    meta = JSON.parsefile(joinpath(data_dir, "meta.json"))
    ref_path = meta["ref_path"]
    case_order = meta["case_order"]
    sample_data_meth = meta["sample_data_meth"]
    nb_pf = meta["nb_pf"]

    mkpath(results_dir)

    # nb_pf (the exported scenario count) is only folded into the output filename when
    # --add-to-name is empty. A bare/manual invocation (no --add-to-name) still needs nb_pf to
    # tell different exports apart -- that's the original behavior, preserved here. But once
    # --add-to-name is given, it's expected to already carry whatever identifies a run (e.g.
    # base_launch_powermodels.sh's own "_<case>_nbthreads<N>" suffix, on top of the machine
    # suffix its own launch_powermodels_*.sh wrapper adds), so nb_pf on top of that is
    # redundant and, worse, can be actively misleading: it's the *exported* count, not
    # necessarily the number of scenarios actually run (--max-scen can be smaller).
    base_name = "powermodels_injection_$(sample_data_meth)"
    if isempty(add_to_name)
        base_name *= "_$(nb_pf)"
    end
    if !warm_start_scenarios
        base_name *= "_nowarmstart"
    end
    if use_ipopt
        base_name *= "_ipopt"
    end
    if !isempty(add_to_name)
        base_name *= add_to_name
    end
    complete_full_path = joinpath(results_dir, "$(base_name).json")
    full_path_Vs = joinpath(results_dir, "$(base_name)_{}_Vs.npy")

    # pay the JIT compile cost once, on the smallest available case, before starting the
    # timed loop (mirrors "warm up OLF" in olf_injection.py / the warmup in exapf_injection.jl)
    warmup_case = isempty(parsed["warmup-case"]) ? first(case_order) : parsed["warmup-case"]
    println("Warming up on $warmup_case ...")
    try
        wdata = PowerModels.parse_file(joinpath(ref_path, warmup_case))
        if use_ipopt
            PowerModels.solve_ac_pf(wdata, ipopt_optimizer)
        else
            wresult = PowerModels.compute_ac_pf(wdata; flat_start=true, iterations=max_iter, ftol=tol)
            # also pay the JIT cost of compute_ac_pf's flat_start=false (warm-start) branch:
            # since get_init_method.jl now picks flat/dc/file automatically per case, which
            # branch a given case's base-case solve actually takes is no longer fixed for the
            # whole run, so both need warming here rather than just the one the CLI used to select
            if wresult["termination_status"]
                apply_voltage_start!(wdata, wresult["solution"]["bus"])
                PowerModels.compute_ac_pf(wdata; flat_start=false, iterations=max_iter, ftol=tol)
            end
        end
    catch err
        err isa InterruptException && rethrow()
        println("  WARNING: warmup failed (continuing anyway): ", format_error(err))
    end

    benchmark_results = Dict{String,Any}()

    for case_fn in case_order
        case_stem = first(splitext(case_fn))
        casefile = joinpath(ref_path, case_fn)
        println("=== $case_fn === (", Dates.now(), ")")

        try
            load_p_bus = npzread(joinpath(data_dir, "$(case_stem)_load_p_mw.npy"))
            load_q_bus = npzread(joinpath(data_dir, "$(case_stem)_load_q_mvar.npy"))
            gen_p_active = npzread(joinpath(data_dir, "$(case_stem)_gen_p_mw.npy"))
            nbus_data, nscen_avail = size(load_p_bus)
            nscen = max_scen > 0 ? min(max_scen, nscen_avail) : nscen_avail

            data = PowerModels.parse_file(casefile)
            baseMVA = data["baseMVA"]
            nbus = length(data["bus"])
            if nbus != nbus_data
                msg = "shape mismatch (data: nbus=$nbus_data, PowerModels: nbus=$nbus)"
                println("  SKIP: $msg")
                benchmark_results[case_fn] = Dict("error" => msg)
                open(complete_full_path, "w") do f
                    JSON.print(f, benchmark_results, 2)
                end
                continue
            end

            # which starting point to solve the base case from -- automatic per case, same
            # table as ../get_init_method.py (see get_init_method.jl and module docstring)
            init_method = get_init_method(case_fn)
            flat_start_case = true
            if init_method == "dc"
                dc_result = PowerModels.compute_dc_pf(data)
                apply_dc_start!(data, dc_result["solution"]["bus"])
                flat_start_case = false
            elseif init_method == "file"
                flat_start_case = false
            elseif init_method != "flat"
                error("unknown init method \"$init_method\" for $case_fn")
            end

            bus_ids = row_order_bus_ids(casefile)  # position (1-based) -> matpower bus id
            bus_id_to_pos = Dict(bid => i for (i, bid) in enumerate(bus_ids))
            active_gen_idx = active_gen_row_indices(data)
            n_active_gen = length(active_gen_idx)
            if n_active_gen != size(gen_p_active, 1)
                msg = "gen count mismatch (data: $(size(gen_p_active, 1)), PowerModels active gens: $n_active_gen)"
                println("  SKIP: $msg")
                benchmark_results[case_fn] = Dict("error" => msg)
                open(complete_full_path, "w") do f
                    JSON.print(f, benchmark_results, 2)
                end
                continue
            end
            # loose sanity check on the bus/load alignment: every PowerModels load's bus
            # should have picked up a nonzero export value somewhere in scenario 1 unless the
            # base case itself has a zero-P/zero-Q load there (rare, but not impossible)
            n_pm_loads = length(data["load"])
            n_export_loads = count(!=(0.0), load_p_bus[:, 1]) + count(!=(0.0), load_q_bus[:, 1])
            if n_pm_loads > 0 && n_export_loads == 0
                println("  WARNING: PowerModels has $n_pm_loads loads but exported scenario 1 has no nonzero load_p/load_q -- check Pd/Qd alignment")
            end

            # base-case powerflow: `data` exactly as parsed, before any scenario overwrites its
            # loads/gens. Solved once here and, when warm_start_scenarios is on, reused as a
            # fixed warm-start anchor for every scenario below instead of chaining scenario to
            # scenario (see module docstring for why: a fixed, well-conditioned anchor can't be
            # corrupted by an earlier scenario's spurious root the way a chain can)
            base_case_sol = nothing
            base_case_conv = false
            base_case_time = 0.0
            if warm_start_scenarios
                t0 = time_ns()
                if ipopt_optimizer === nothing
                    base_result = PowerModels.compute_ac_pf(data; flat_start=flat_start_case, iterations=max_iter, ftol=tol)
                    base_case_conv = base_result["termination_status"]
                else
                    base_result = PowerModels.solve_ac_pf(data, ipopt_optimizer)
                    base_case_conv = ipopt_converged(base_result["termination_status"])
                end
                base_case_time = (time_ns() - t0) / 1e9
                if base_case_conv
                    base_case_sol = base_result["solution"]
                else
                    println("  WARNING: base-case powerflow did not converge -- falling back to flat/file start for every scenario")
                end
            end

            # first + second single-scenario solve, mirroring ls_injection.py's warm-then-time
            # pattern -- both solve the *same* scenario from the same (flat/file) start, so
            # scenario-to-scenario warm starting (warm_start_scenarios) intentionally does not
            # apply here, keeping "following_powerflow" a measure of repeat-solve cost rather
            # than of the warm-start benefit
            conv1, _, t1, _ = solve_scenario(data, baseMVA, bus_id_to_pos, active_gen_idx,
                                           load_p_bus, load_q_bus, gen_p_active, 1;
                                           flat_start=flat_start_case, max_iter=max_iter, tol=tol,
                                           ipopt_optimizer=ipopt_optimizer)
            conv2, _, t2, _ = solve_scenario(data, baseMVA, bus_id_to_pos, active_gen_idx,
                                           load_p_bus, load_q_bus, gen_p_active, 1;
                                           flat_start=flat_start_case, max_iter=max_iter, tol=tol,
                                           ipopt_optimizer=ipopt_optimizer)

            tmp_res = Dict{String,Any}(
                "grid_size" => Dict("total_bus" => nbus, "total_gen_active" => n_active_gen),
                "init_method" => init_method,
                "init_powerflow" => Dict("total_time" => t1, "converged" => conv1),
                "following_powerflow" => Dict("total_time" => t2, "converged" => conv2),
            )
            if warm_start_scenarios
                tmp_res["base_case_powerflow"] = Dict("total_time" => base_case_time, "converged" => base_case_conv)
            end

            V = save_voltages ? fill(ComplexF64(NaN, NaN), nscen, nbus) : nothing
            n_workers = clamp(nthreads, 1, nscen)
            # start the wall clock time measurment
            t_loop0 = time_ns()
            if n_workers <= 1
                # single-threaded: solve directly against this case's own live `data`,
                # unchanged from before --nthreads existed (no deepcopy needed/wanted here)
                total_modif, total_solver_time, nb_solved, errors = solve_scenario_group(
                    data, baseMVA, bus_id_to_pos, active_gen_idx, load_p_bus, load_q_bus, gen_p_active,
                    1:nscen, bus_ids, V; flat_start=flat_start_case, max_iter=max_iter, tol=tol,
                    warm_start_sol=base_case_sol, ipopt_optimizer=ipopt_optimizer,
                )
            else
                # each worker task gets its own deepcopy of `data` (made here, before
                # dispatching) since solve_scenario mutates it in place every call -- see
                # solve_scenario_group's docstring
                groups = split_range(nscen, n_workers)
                tasks = Vector{Task}(undef, length(groups))
                for (i, g) in enumerate(groups)
                    # unlike pandapower, olf and rustpower, this is included in the wall time
                    # The threadings possibility julia offers by default (the one used here)
                    # cannot easily (as far as we know) exclude that from wall clock
                    data_copy = deepcopy(data)
                    tasks[i] = Threads.@spawn solve_scenario_group(
                        data_copy, baseMVA, bus_id_to_pos, active_gen_idx, load_p_bus, load_q_bus, gen_p_active,
                        g, bus_ids, V; flat_start=flat_start_case, max_iter=max_iter, tol=tol,
                        warm_start_sol=base_case_sol, ipopt_optimizer=ipopt_optimizer,
                    )
                end
                total_modif = 0.0
                total_solver_time = 0.0
                nb_solved = 0
                errors = String[]
                for t in tasks
                    gmodif, gsolver, gsolved, gerrors = fetch(t)
                    total_modif += gmodif
                    total_solver_time += gsolver
                    nb_solved += gsolved
                    append!(errors, gerrors)
                end
            end
            # wall-clock, not `total_time`: with --nthreads > 1, total_time sums each worker's
            # own solve durations (aggregate CPU-time across concurrently-running tasks), which
            # would understate a threaded run's real speedup if used for pf/s throughput
            wall_time = (time_ns() - t_loop0) / 1e9
            total_time = total_modif + total_solver_time
            @printf("  total: %.3fs  wall: %.3fs  (%.1f pf/s)  solved: %d/%d\n",
                    total_time, wall_time, nb_solved / max(wall_time, 1e-9), nb_solved, nscen)

            tmp_res["time_series"] = Dict{String,Any}(
                # sum of total_modif + total_solver_time (aggregate CPU time across workers
                # when nthreads > 1, NOT elapsed time -- see wall_time), matching
                # olf_injection.py's own total_time = total_modif + total_pf
                "total_time" => total_time,
                "total_modif" => total_modif,
                "total_solver_time" => total_solver_time,
                "wall_time" => wall_time,
                "nb_solved" => nb_solved,
                "nb_total" => nscen,
                "nthreads" => nthreads,
            )
            if !isempty(errors)
                tmp_res["time_series"]["errors"] = errors
            end
            benchmark_results[case_fn] = tmp_res
            open(complete_full_path, "w") do f
                JSON.print(f, benchmark_results, 2)
            end

            if save_voltages && V !== nothing
                npzwrite(replace(full_path_Vs, "{}" => case_stem), V)
            end
        catch err
            err isa InterruptException && rethrow()
            msg = format_error(err)
            println("  ERROR: failed to process $case_fn: $msg")
            benchmark_results[case_fn] = Dict("error" => msg)
            open(complete_full_path, "w") do f
                JSON.print(f, benchmark_results, 2)
            end
        end
    end

    println("Done (", Dates.now(), "). Results written to $complete_full_path")
end

main()

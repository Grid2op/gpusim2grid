# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""rustpower's low-level `NewtonSolver` as an injection-sweep backend.

Same benchmark shape as `ls_injection.py`/`olf_injection.py`: fixed topology, one JSON per
run under `../../results_matpower`, each scenario resamples every load's P/Q and every gen's
P around the grid's own base case (`get_injection_from_base()`, shared with every other
backend here). Unlike `ls_injection.py`'s `InjectionSweepCPP`, rustpower exposes no batch API
from Python, so this is a plain per-scenario Python loop, structurally closer to
`olf_injection.py`/ExaPF's own loop -- including the same `--nb_threads` multiprocessing
dispatch as `olf_injection.py`/`pp_injection.py` (see `run_multiprocess`/`_worker_init`/
`_mp_worker` below).

rustpower has no grid model of its own that this project's matpower files could be loaded
into in a directly comparable way (`PowerGrid.from_pandapower`'s own conversion was checked
and differs from pandapower's by ~1e-2 pu on case118/case300 -- not voltage-comparable). So
lightsim2grid still owns the modelling of the grid -- but only to build the *fixed* per-grid system this
sweep reuses for every scenario: one `ac_pf()` in
`solve_base_case()` (a) picks `v_init` and (b) is the only way to make lightsim2grid populate
its own solver-order arrays (`get_Ybus_solver`/`get_pv_solver`/`get_pq_solver`/
`get_slack_ids_solver` -- all empty before any `ac_pf` has run).

Since this experiment never changes topology, gen status, or
v_gen setpoints -- only load P/Q and gen P vary scenario to scenario -- those arrays stay
valid for the rest of the sweep unchanged once populated. `build_static_context` captures
them, right after `solve_base_case`, exactly once per grid; `run_scenario_range`'s
per-scenario work is then exactly two things: build that scenario's `Sbus` and hand it to
rustpower's `NewtonSolver`.

`Sbus` can be built knowing only load_p, load_q and gen_p because no elements change 
buses on the benchmarked sweep. Only the information about which load / gen sits on which bus is 
relevant here.

`build_static_context` captures the load/gen -> solver-bus index maps once; each scenario
then only needs a vectorized scatter-add over `load_p[s]`/`load_q[s]`/`gen_p[s]` -- see
`sbus_build_time` in the JSON output, the "prepare the data" half of a fair wall-time number
(the other half is `rustpower_setup_time` + `rustpower_solve_time`).

lightsim2grid's runs exactly once per grid (inside
`solve_base_case`, to seed `v_init` and populate the solver arrays above).

Four things found when doing this benchmark:

1. `setup_context` documents its `y_*` arguments as CSC but reads them as CSR -- it solves
   with Y transposed. Pass `Ybus.T.tocsc()`. Without it, every grid with phase-shifting
   transformers (all PEGASE/RTE cases) converges happily and reports a real power mismatch
   of 0.4-200 pu. Symmetric-Ybus grids (case14/case118/case300-shaped) are unaffected, so
   this is invisible in small-case testing. The author of rustpower will be notified 
   (probably an documentation issue)
2. `NewtonSolver` hardcodes `max_it=10`/`tol=1e-8` (2-norm of the mismatch) with no Python
   setter, so every scenario here solves under that same 10-iteration budget
3. rustpower's `v_init` is not just a Newton starting guess: `setup_context` takes no
   generator-voltage-setpoint input of its own, so whatever magnitude `v_init` carries at
   PV/ref buses is exactly what stays fixed there through the whole solve. 
   Feeding rustpower a `np.ones(...)` flat start
   silently solves a *different* problem (every PV/ref bus pinned to 1.0 pu instead of its
   real setpoint) and converges to a self-consistent but wrong answer `solve_base_case()` 
   below always derives `v_init` from a real powerflow solve (DC, or
   the case file's own bus voltages -- see point 5), never from a flat guess, specifically so
   PV/ref magnitude is correct going in, and `build_static_context` permutes that same
   `v_init` into solver order once, for every scenario to reuse.
5. Rustpower's low-level solver has no distributed-slack
   concept. None of this project's 15 canonical grids actually have more
   than one slack (checked directly), so this makes no observed difference here but need to
   be kept in mind.

Known open finding, *not* one of the five fixes above: on `case_ACTIVSg70k.m` specifically,
`base_case_regression` (the one-time check above) has shown `max_abs_dV` around 0.09 pu even
though both sides' own power-mismatch self-checks are ~1e-11 -- i.e. both lightsim2grid and
rustpower converge to a genuinely power-balanced solution of the *same* (Ybus, Sbus, pv, pq)
system, just two different ones (rustpower's uniformly lower voltage across nearly every bus,
not a handful of spurious ones).
"""

import os

# Must happen before numpy/scipy/rustpower are imported below -- each loads its
# own BLAS/OpenMP (and, for rustpower, its own Rust thread pool if it uses one) backend at
# import time, defaulting to as many threads as this machine has cores. --nb_threads > 1 adds
# *OS processes* on top of that (run_multiprocess/_worker_init below), so left unpinned, N
# worker processes x each its own full-core-count thread pool massively oversubscribes the
# machine and erases (or reverses) the intended scaling -- this is what "no throughput
# improvement across --nb_threads" looks like. Same fix this project's own Julia backends
# already apply for the identical reason (see exapf_injection.jl/powermodels_injection.jl's
# own "--nthreads: Pins BLAS to 1 thread/task to avoid oversubscription when N > 1").
#
# Set here, unconditionally, forcing (not merely defaulting) every one of these -- not inside
# _worker_init: under multiprocessing's "spawn" start method (see run_multiprocess) a worker
# reimports this entire module from scratch, including every import below, before
# _worker_init ever runs, so anything worker-specific set inside it would already be too
# late. Applying it to the single-process (--nb_threads <= 1) run too costs essentially
# nothing here: this workload is a sequence of small, mostly-sparse per-scenario operations,
# not large dense matrix multiplies that would actually benefit from BLAS-level threading.
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_env] = "1"

import json
import multiprocessing as mp
import time
import warnings

import numpy as np
from matpowercaseframes import CaseFrames
from tqdm import tqdm

from lightsim2grid.network import init_from_matpower
import rustpower
from rustpower.solver import NewtonSolver

from kcl_checker import (
    KCLChecker,
    summarize_kcl_mismatch,
    summarize_kcl_mismatch_checkpoints,
    kcl_mismatch_per_scenario,
)
from scenario_utils import (
    all_file_names,
    REF_PATH,
    TOL_PF,
    get_injection_from_base,
    get_default_args,
    PATH_RESULTS,
    get_final_name,
)

BASE_EXPE_NAME = "rustpower_injection_{}_{}"

# rustpower's low-level NewtonSolver hardcodes these -- no python setter exists (see module
# docstring, point 2). Reusing lightsim2grid's own TOL_PF (1e-8) keeps both sides on the same
# tolerance; the iteration cap is set to match rustpower's own hardcoded 10 explicitly.
RUSTPOWER_MAX_IT = 10
RUSTPOWER_TOL = TOL_PF

# see module docstring, point 5.
LS_ALGO_FOR_RUSTPOWER = "NRSing_KLU"


def get_args():
    parser = get_default_args()
    parser.add_argument('--nb_threads', type=int, default=1,
                        help="Number of concurrent worker processes used for the scenario "
                             "sweep (each its own OS process, own lightsim2grid grid load -- "
                             "see run_multiprocess; rustpower/lightsim2grid expose no batch "
                             "or thread-parallel API of their own, so this is plain "
                             "multiprocessing, same shape as olf_injection.py/pp_injection.py), "
                             "default to 1")
    parser.add_argument("--grids", type=str, default=",".join(all_file_names),
                         help="comma-separated matpower filenames to benchmark (default: all "
                              f"{len(all_file_names)} grids in all_file_names)")
    return parser


def check_rustpower_klu():
    """Warn (do not raise) if the installed rustpower wheel was not built with a KLU
    backend -- the ~1e-12 voltage agreement this backend regression-checks against was only
    validated KLU-vs-KLU (see module docstring)."""
    feats = rustpower.features()
    if not any("klu" in f.lower() for f in feats):
        warnings.warn(
            f"rustpower.features() = {feats!r} does not report a KLU backend; "
            "NewtonSolver may be using a different linear solver than lightsim2grid's own "
            "KLU, which would invalidate this backend's regression check against it.")


def solver_ids(lsgrid, n_bus):
    """model bus id -> AC-solver bus id (identity if lightsim2grid reports no remapping).

    `.copy()`d: `id_me_to_ac_solver()` (like every other `get_*_solver()` getter used in
    this file) hands back a view into lightsim2grid's own buffer (`OWNDATA` False,
    confirmed directly) -- fine to use while `lsgrid` is still alive, but every caller here
    (`build_static_context`) stores the result past that point, in particular past
    `_worker_init` returning and its local `lsgrid` being freed, at which point an
    uncopied view reads freed memory (silently, on Linux/glibc)."""
    me2s = np.asarray(lsgrid.id_me_to_ac_solver())
    return np.arange(n_bus) if me2s.size == 0 else me2s.copy()


def power_mismatch(Ybus, Sbus, V, pv, pq):
    """max |P| mismatch over pv+pq and |Q| mismatch over pq, in pu."""
    mis = V * np.conj(Ybus @ V) - Sbus
    idx = np.concatenate([pv, pq])
    p_mis = np.abs(mis[idx].real).max() if idx.size else 0.0
    q_mis = np.abs(mis[pq].imag).max() if pq.size else 0.0
    return max(p_mis, q_mis)


def solve_base_case(lsgrid, path, n_bus):
    """Pick the `v_init` this whole sweep (lightsim2grid's own per-scenario `ac_pf` *and*
    every scenario's rustpower `v_init`, see module docstring point 3) will be anchored on,
    and confirm the base case actually converges under it. Tries DC-derived init first (bakes
    in the true PV/ref magnitude *and* usually gives NR a much better angle guess on meshed
    grids than flat -- see module docstring point 2/3); falls back to the case file's own bus
    voltages for the couple of grids where DC-derived init does not converge even for
    lightsim2grid's own `ac_pf` (matches `get_init_method.py`'s own "file"-tagged grids:
    case3375wp.m, case_ACTIVSg70k.m -- confirmed empirically while wiring this in, not just
    copied from that table).

    Returns (v_init, method, V_base) with V_base empty (`.shape[0] == 0`) if neither method
    converges.
    """
    v_dc = lsgrid.dc_pf(np.ones(n_bus, dtype=complex), RUSTPOWER_MAX_IT, 1e-6)
    if v_dc.shape[0] > 0:
        # no unset_changes() between dc_pf and this first ac_pf -- see module docstring point 4
        V_base = lsgrid.ac_pf(v_dc.copy(), RUSTPOWER_MAX_IT, RUSTPOWER_TOL)
        lsgrid.unset_changes()
        if V_base.shape[0] > 0:
            return v_dc, "dc", V_base
    cf = CaseFrames(path)
    v_file = cf.bus["VM"].values * np.exp(1j * np.deg2rad(cf.bus["VA"].values))
    V_base = lsgrid.ac_pf(v_file.copy(), RUSTPOWER_MAX_IT, RUSTPOWER_TOL)
    lsgrid.unset_changes()
    return v_file, "file", V_base


def build_static_context(lsgrid, v_init, n_bus):
    """Everything about this grid that is fixed across the whole sweep -- topology, gen
    status and v_gen setpoints never change here, only load P/Q and gen P do -- captured
    exactly once, right after `solve_base_case`'s own `ac_pf` populated lightsim2grid's
    solver-order arrays (module docstring). Nothing in here, or in the returned dict, ever
    touches `lsgrid` again after this call returns.

    Includes the load/gen -> solver-bus index maps `build_sbus` scatters into, so every
    scenario's `Sbus` can be built directly from that scenario's own `load_p`/`load_q`/
    `gen_p` row -- see module docstring for why this reproduces lightsim2grid's own
    `get_Sbus_solver()` exactly (shunts and everything else fixed land in `Ybus`, not here).

    Every `get_*_solver()` read below is `.copy()`'d (`Ybus` via `.tocsc().copy()` since
    `.tocsc()` on an already-CSC matrix is a no-op, not a copy; `pv`/`pq`/`ref` explicitly)
    -- see `solver_ids`'s own docstring for why: this dict is built to outlive `lsgrid`
    (`_worker_init` frees its `lsgrid` local right after calling this), and every one of
    these getters hands back a view into lightsim2grid's own buffer, not an owned array.
    """
    Ybus = lsgrid.get_Ybus_solver().tocsc().copy()
    Yt = Ybus.T.tocsc()  # rustpower reads y_* as CSR, not CSC -- module docstring point 1
    pv = np.asarray(lsgrid.get_pv_solver()).copy()
    pq = np.asarray(lsgrid.get_pq_solver()).copy()
    ref = np.asarray(lsgrid.get_slack_ids_solver()).copy()
    n = Ybus.shape[0]

    ids = solver_ids(lsgrid, n_bus)
    live = (ids >= 0) & (ids < n)
    v0_s = np.zeros(n, dtype=complex)
    v0_s[ids[live]] = v_init[np.flatnonzero(live)]

    p_vec = np.concatenate([pq, pv, ref]).astype(np.int64)
    p_inv = np.zeros(n, dtype=np.int64)
    p_inv[p_vec] = np.arange(n)

    n_load = len(lsgrid.get_loads_res_full()[0])
    n_gen = len(lsgrid.get_gen_res_full()[0])
    bus_load = np.array([lsgrid.get_bus_load(i) for i in range(n_load)])
    bus_gen = np.array([lsgrid.get_bus_gen(i) for i in range(n_gen)])

    return dict(
        Ybus=Ybus, pv=pv, pq=pq, n=n, n_bus=n_bus,
        y_indptr=Yt.indptr.astype(np.int32), y_indices=Yt.indices.astype(np.int32),
        y_data=Yt.data.astype(complex),
        v0_s=v0_s, p_vec_l=p_vec.tolist(), p_inv_l=p_inv.tolist(),
        npv=len(pv), npq=len(pq),
        ids=ids, live=live,
        bus_load_solver=ids[bus_load], bus_gen_solver=ids[bus_gen],
        sn_mva=lsgrid.get_sn_mva(),
    )


def build_sbus(ctx_static, load_p_row, load_q_row, gen_p_row):
    """This scenario's `Sbus`, in solver-bus order, straight from its own load/gen values.
    """
    Sbus = np.zeros(ctx_static["n"], dtype=complex)
    np.add.at(Sbus, ctx_static["bus_gen_solver"], gen_p_row / ctx_static["sn_mva"])
    np.add.at(Sbus, ctx_static["bus_load_solver"],
              -(load_p_row + 1j * load_q_row) / ctx_static["sn_mva"])
    return Sbus


def base_case_regression(ctx_static, V_base_solver, load_p0, load_q0, gen_p0):
    """One-time sanity check, run once per grid outside the timed sweep (module docstring):
    solve rustpower on the exact base-case `Sbus` and compare against lightsim2grid's own
    base-case voltage (`V_base_solver`, already in solver order).
    """
    Sbus = build_sbus(ctx_static, load_p0, load_q0, gen_p0)
    ctx = NewtonSolver()
    ctx.setup_context(y_indptr=ctx_static["y_indptr"], y_indices=ctx_static["y_indices"],
                       y_data=ctx_static["y_data"], s_bus=Sbus, v_init=ctx_static["v0_s"],
                       p_vec=ctx_static["p_vec_l"], p_inv=ctx_static["p_inv_l"],
                       npv=ctx_static["npv"], npq=ctx_static["npq"])
    converged = ctx.solve()
    if not converged:
        return {"converged": False}
    V_rp = ctx.get_voltage()
    return {
        "converged": True,
        "max_abs_dV": float(np.abs(V_rp - V_base_solver).max()),
        "mismatch_ls2g": power_mismatch(ctx_static["Ybus"], Sbus, V_base_solver,
                                         ctx_static["pv"], ctx_static["pq"]),
        "mismatch_rustpower": power_mismatch(ctx_static["Ybus"], Sbus, V_rp,
                                              ctx_static["pv"], ctx_static["pq"]),
    }


def run_scenario_range(ctx_static, load_p, load_q, gen_p, start, end, voltages):
    """Run scenarios [start, end) against rustpower only, using the fixed `ctx_static`
    (module docstring/`build_static_context`). Used both
    directly (nb_threads <= 1) and inside each multiprocessing worker (see _mp_worker), where
    `load_p`/`load_q`/`gen_p` are already that worker's own local [0, end-start) chunk and
    `voltages` (if any) is that worker's own local array -- so `start`/`end` are always the
    *local* array's own bounds, never scenario ids into some larger array a worker only holds
    a slice of.

    Per scenario, timed separately: `sbus_build` (build_sbus, pure numpy) then
    `rp_setup`/`rp_solve` (rustpower's own setup_context/solve) -- nothing else. Returns
    (acc, unsolved_rp) with `unsolved_rp` holding indices relative to `start` (ie directly
    usable as positions into this same [start, end) range -- callers needing global scenario
    ids, see run_multiprocess, add the chunk's own offset back in themselves).
    """
    n_bus = ctx_static["n_bus"]
    ids, live = ctx_static["ids"], ctx_static["live"]
    acc = dict(sbus_build=0.0, rp_setup=0.0, rp_solve=0.0, conv_rp=0, mism_rp=0.0)
    unsolved_rp = []

    for s in range(start, end):
        t0 = time.perf_counter()
        Sbus = build_sbus(ctx_static, load_p[s], load_q[s], gen_p[s])
        t1 = time.perf_counter()

        ctx = NewtonSolver()
        ctx.setup_context(y_indptr=ctx_static["y_indptr"], y_indices=ctx_static["y_indices"],
                           y_data=ctx_static["y_data"], s_bus=Sbus, v_init=ctx_static["v0_s"],
                           p_vec=ctx_static["p_vec_l"], p_inv=ctx_static["p_inv_l"],
                           npv=ctx_static["npv"], npq=ctx_static["npq"])
        t2 = time.perf_counter()
        converged = ctx.solve()
        t3 = time.perf_counter()
        acc["sbus_build"] += t1 - t0
        acc["rp_setup"] += t2 - t1
        acc["rp_solve"] += t3 - t2

        if not converged:
            unsolved_rp.append(s)
            continue
        acc["conv_rp"] += 1

        V_rp = ctx.get_voltage()
        acc["mism_rp"] = max(acc["mism_rp"],
                              power_mismatch(ctx_static["Ybus"], Sbus, V_rp,
                                              ctx_static["pv"], ctx_static["pq"]))

        if voltages is not None:
            # back to original (model) bus ordering, comparable with the other backends'
            # saved voltages -- undo the solver permutation built once in ctx_static.
            v_model = np.full(n_bus, np.nan, dtype=complex)
            v_model[np.flatnonzero(live)] = V_rp[ids[live]]
            voltages[s] = v_model

    return acc, unsolved_rp


# Set once per worker process by _worker_init, read by _mp_worker -- process-local global,
# never shared across workers (each is its own OS process).
_worker_ctx_static = None


def _worker_init(fn, barrier):
    """Pool(initializer=...) hook: runs exactly once in each worker process, before that
    worker ever pulls a real task off the queue -- this is where the (expensive, one-time)
    grid load + base-case solve + build_static_context belongs, kept out of the timed
    _mp_worker task function entirely. Mirrors olf_injection.py's _worker_init.

    Runs with the "spawn" start method (see run_multiprocess), so this whole module is
    freshly re-imported in the child before this function ever runs -- lightsim2grid/
    rustpower's native (C++/Rust) extensions get a brand new process here, never one forked
    from the parent (which, by the time workers are created, has already loaded a grid and
    driven both extensions through at least one solve -- forking a process with live native
    state is the same risk olf_injection.py's own _worker_init docstring flags for
    pypowsybl's GraalVM isolate).

    `barrier.wait()` at the end (parties = n_workers + 1, the parent included -- see
    run_multiprocess) is what lets the parent exclude this setup time from its wall_time
    measurement. `lsgrid` itself is discarded once `build_static_context` returns -- nothing
    from _mp_worker onward ever touches it again (module docstring).
    """
    global _worker_ctx_static
    path = os.path.join(REF_PATH, fn)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lsgrid = init_from_matpower(path)
    lsgrid.change_algorithm(LS_ALGO_FOR_RUSTPOWER)
    n_bus = lsgrid.total_bus()
    v_init, _, V_base = solve_base_case(lsgrid, path, n_bus)
    if V_base.shape[0] == 0:
        raise RuntimeError(f"{fn}: base powerflow did not converge in worker "
                            "(tried dc and file init)")
    _worker_ctx_static = build_static_context(lsgrid, v_init, n_bus)
    barrier.wait()


def _mp_worker(load_p_chunk, load_q_chunk, gen_p_chunk, need_voltages):
    """The timed part only: solve this worker's own [0, n_local) chunk against the
    ctx_static _worker_init already built for this process."""
    n_local = load_p_chunk.shape[0]
    voltages = (np.full((n_local, _worker_ctx_static["n_bus"]), np.nan, dtype=complex)
                if need_voltages else None)
    acc, unsolved_rp = run_scenario_range(
        _worker_ctx_static, load_p_chunk, load_q_chunk, gen_p_chunk, 0, n_local, voltages)
    return acc, unsolved_rp, voltages


def run_multiprocess(nb_threads, fn, load_p, load_q, gen_p, need_voltages):
    """Replaces the single-process run_scenario_range() call for --nb_threads > 1: spawns
    nb_threads processes (multiprocessing.Pool, "spawn" context), each loading its own grid
    once via _worker_init, then solving its own contiguous [start, end) chunk of
    load_p/load_q/gen_p. Mirrors olf_injection.py's run_multiprocess (same Barrier trick to
    exclude worker setup/warm-up time from wall_time) -- unlike pp_injection.py's own
    multiprocess dispatcher, which instead deepcopy's an already-built pandapower net into
    each worker: lightsim2grid's LSGrid and rustpower's NewtonSolver are compiled-extension
    objects with no such support, so every worker here builds its own from scratch instead.
    """
    ctx = mp.get_context("spawn")
    nb_scen = load_p.shape[0]
    n_workers = min(nb_threads, nb_scen)
    bounds = np.linspace(0, nb_scen, n_workers + 1).astype(int)
    barrier = ctx.Barrier(n_workers + 1)

    with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(fn, barrier)) as pool:
        args = [
            (load_p[bounds[i]:bounds[i + 1]], load_q[bounds[i]:bounds[i + 1]],
             gen_p[bounds[i]:bounds[i + 1]], need_voltages)
            for i in range(n_workers)
        ]
        barrier.wait()  # blocks until every worker has finished _worker_init
        beg_wall = time.perf_counter()
        results = pool.starmap(_mp_worker, args)
        wall_time = time.perf_counter() - beg_wall

    acc = dict(sbus_build=0.0, rp_setup=0.0, rp_solve=0.0, conv_rp=0, mism_rp=0.0)
    unsolved_rp = []
    for i, (a, u_rp, _) in enumerate(results):
        for k in ("sbus_build", "rp_setup", "rp_solve", "conv_rp"):
            acc[k] += a[k]
        acc["mism_rp"] = max(acc["mism_rp"], a["mism_rp"])
        unsolved_rp.extend(int(bounds[i]) + x for x in u_rp)
    voltages = np.concatenate([r[2] for r in results], axis=0) if need_voltages else None
    return acc, unsolved_rp, wall_time, voltages


if __name__ == "__main__":
    check_rustpower_klu()

    args = get_args().parse_args()
    nb_pf_total = int(args.nb_pf)
    seed = int(args.seed)
    sample_data_meth = str(args.sample_data_meth)
    add_to_name = str(args.add_to_name)
    save_voltages = args.save_voltages
    evaluate_kcl = args.evaluate_kcl
    need_voltages = save_voltages or evaluate_kcl
    nb_threads = int(args.nb_threads)
    grids = [g.strip() for g in args.grids.split(",") if g.strip()]

    if args.save_flows:
        print("rustpower_injection.py: --save_flows is not supported (rustpower's low-level "
              "solver exposes no branch-flow computation) -- ignored")

    benchmark_results = {}
    nm_results = BASE_EXPE_NAME.format(sample_data_meth, nb_pf_total)
    complete_full_path = get_final_name(PATH_RESULTS, nm_results, add_to_name)
    nm_tmp, _ = os.path.splitext(complete_full_path)
    full_path_Vs = f"{nm_tmp}_{{}}_Vs.npy"

    for fn in tqdm(grids):
        tmp_res_dict = {}
        path = os.path.join(REF_PATH, fn)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lsgrid = init_from_matpower(path)
        lsgrid.change_algorithm(LS_ALGO_FOR_RUSTPOWER)

        n_bus = lsgrid.total_bus()
        v_init, init_method, V_base = solve_base_case(lsgrid, path, n_bus)
        if V_base.shape[0] == 0:
            print(f"Error for {fn}: base powerflow did not converge (tried dc and file init), skipping")
            benchmark_results[fn] = {"error": "base powerflow did not converge"}
            with open(complete_full_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_results, fp=f, indent=2)
            continue

        tmp_res_dict["grid_size"] = {
            "total_bus": int(n_bus),
            "total_branch": len(lsgrid.get_lines()) + len(lsgrid.get_trafos()),
            "Ybus_nnz": int(lsgrid.get_Ybus_solver().nnz),
        }
        tmp_res_dict["v_init_method"] = init_method

        # build_static_context reads lsgrid's solver-order arrays as populated by
        # solve_base_case's own ac_pf above -- must run before anything else touches lsgrid
        # again (it doesn't, past this point: module docstring). V_base_solver/base
        # load-gen values are grabbed here too, for the one-time base_case_regression check
        # below, while they're still valid to read off lsgrid.
        ctx_static = build_static_context(lsgrid, v_init, n_bus)
        V_base_solver = np.asarray(lsgrid.get_V_solver()).ravel()
        load_p0, load_q0 = lsgrid.get_loads_res_full()[:2]
        gen_p0 = lsgrid.get_gen_target_p()
        tmp_res_dict["base_case_regression"] = base_case_regression(
            ctx_static, V_base_solver, load_p0, load_q0, gen_p0)

        # get_injection_from_base() reads lsgrid.get_loads_res_full()/get_gen_res_full(),
        # which are only populated after a powerflow has been solved -- solve_base_case()
        # above already did that (same prerequisite as ls_injection.py/olf_injection.py).
        load_p, load_q, gen_p = get_injection_from_base(
            fn, lsgrid, sample_meth=sample_data_meth, seed=seed, nb_pf_total=nb_pf_total)

        if nb_threads <= 1:
            all_voltages = (np.full((nb_pf_total, n_bus), np.nan, dtype=complex)
                             if need_voltages else None)
            acc, unsolved_rp = run_scenario_range(
                ctx_static, load_p, load_q, gen_p, 0, nb_pf_total, all_voltages)
            wall_time = acc["sbus_build"] + acc["rp_setup"] + acc["rp_solve"]
        else:
            # each worker builds its own grid + ctx_static from scratch (see
            # run_multiprocess) -- the `lsgrid`/`ctx_static` built above are not reused for
            # the sweep itself in this branch, only for grid_size/v_init_method/
            # base_case_regression/get_injection_from_base() above.
            acc, unsolved_rp, wall_time, all_voltages = run_multiprocess(
                nb_threads, fn, load_p, load_q, gen_p, need_voltages)

        rp_total = acc["rp_setup"] + acc["rp_solve"]
        total_time = acc["sbus_build"] + rp_total
        tmp_res_dict["time_series"] = {
            "nb_scenarios": nb_pf_total,
            "nb_solved": acc["conv_rp"],
            # aggregate time summed across workers when nb_threads > 1 -- NOT elapsed time,
            # see wall_time for that (mirrors olf_injection.py's own total_time/wall_time split)
            "total_time": total_time,
            "wall_time": wall_time,
            "solver_time": rp_total,
            "sbus_build_time": acc["sbus_build"],
            "rustpower_setup_time": acc["rp_setup"],
            "rustpower_solve_time": acc["rp_solve"],
            "nb_threads": nb_threads,
        }
        tmp_res_dict["mismatch_rustpower"] = acc["mism_rp"]

        # a scenario counts as solved (for voltages/KCL purposes) only if rustpower converged
        # on it -- exactly the scenarios run_scenario_range() wrote a real (non-NaN) row of
        # `all_voltages` for.
        solved_mask = np.ones(nb_pf_total, dtype=bool)
        if unsolved_rp:
            solved_mask[unsolved_rp] = False
            tmp_res_dict["unsolved_scenarios_rustpower"] = unsolved_rp

        if evaluate_kcl:
            kcl_checker = KCLChecker(path)
            kcl_mismatch = kcl_checker.check_kcl_injection(
                load_p[solved_mask], load_q[solved_mask], gen_p[solved_mask],
                all_voltages[solved_mask])
            tmp_res_dict.update(summarize_kcl_mismatch(kcl_mismatch))
            tmp_res_dict["kcl_mismatch_checkpoints"] = summarize_kcl_mismatch_checkpoints(kcl_mismatch)
            tmp_res_dict.update(kcl_mismatch_per_scenario(kcl_mismatch, solved_mask))

        benchmark_results[fn] = tmp_res_dict
        with open(complete_full_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, fp=f, indent=2)

        if save_voltages:
            case_nm, _ = os.path.splitext(fn)
            np.save(file=full_path_Vs.format(case_nm), arr=all_voltages)

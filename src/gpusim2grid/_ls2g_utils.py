# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Helpers bridging lightsim2grid ``GridModel`` / ``LSGrid`` objects to the
plain arrays the gpusim2grid GPU solvers consume.

lightsim2grid owns the physics model and the base-case (N) power flow on the
CPU.  These helpers extract — *without re-parsing pandapower or recomputing
admittances* — the Ybus (CSR), Sbus, pv/pq/slack partition, the converged
voltage, and the π-model branch admittances that gpusim2grid needs.

This is the Python-side extraction path.  It works against any installed
lightsim2grid and keeps gpusim2grid usable on the already-compiled extension
without a C++ rebuild.  A future zero-copy C++ bridge (reaching directly into
the solved ``LSGrid`` object) can replace it transparently behind the same
``from_lightsim2grid`` constructors.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def _relabel_to_solver(me_to_solver, global_ids):
    """Relabel grid-MODEL bus ids to AC-solver bus ids.

    ``me_to_solver`` is ``grid.id_me_to_ac_solver()``: empty means identity
    numbering, ``-1`` marks a bus the solver dropped (isolated / Kron-reduced).
    Ids outside the map (already ``-1``, or past its end) are passed through
    unchanged, exactly like ``ls2g_bridge.cpp``'s ``concat_busids_to_solver``.
    """
    global_ids = np.asarray(global_ids)
    if me_to_solver.size == 0:
        return global_ids   # identity numbering
    out = global_ids.copy()
    valid = (global_ids >= 0) & (global_ids < me_to_solver.shape[0])
    out[valid] = me_to_solver[global_ids[valid]]
    return out


def _validate_precision(precision):
    """Raise ``ValueError`` if *precision* disagrees with the compiled binary.

    Precision is a compile-time choice in gpusim2grid (FP32 vs FP64).  This
    surfaces a mismatch as a clear, early error instead of silently ignoring
    the request.  ``precision=None`` accepts whatever was compiled.
    """
    if precision is None:
        return
    from . import _gpusim2grid

    precision = str(precision).lower()
    if precision not in ("fp32", "fp64"):
        raise ValueError(
            f"Unrecognized precision {precision!r}. Expected 'fp32' or 'fp64'.")

    is_fp32 = bool(_gpusim2grid.is_fp32)
    if precision == "fp32" and not is_fp32:
        raise ValueError(
            "precision='fp32' requested but the extension was compiled FP64. "
            "Rebuild with CUDA_REAL_FLOAT=1, or pass precision='fp64'.")
    if precision == "fp64" and is_fp32:
        raise ValueError(
            "precision='fp64' requested but the extension was compiled FP32. "
            "Rebuild without CUDA_REAL_FLOAT, or pass precision='fp32'.")


def _ensure_solved(grid, v_init, max_iter, tol):
    """Run a CPU AC power flow so Ybus/Sbus/V are populated and consistent.

    Returns ``(v_init, v_converged)``.  ``v_init`` is a DC warm-start;
    ``v_converged`` is the AC solution lightsim2grid produced on the CPU.
    """
    n_bus = grid.get_bus_vn_kv().shape[0]
    if v_init is None:
        v_init_dc = np.ones(n_bus, dtype=complex)
        v_init = grid.dc_pf(v_init_dc, 1, 1e-6)
    if v_init.shape[0] != n_bus:
        raise ValueError(
            f"v_init has {v_init.shape[0]} buses, grid has {n_bus}.")
    v_converged = grid.ac_pf(v_init.copy(), int(max_iter), float(tol))
    if v_converged.shape[0] == 0:
        raise RuntimeError(
            "lightsim2grid CPU AC power flow did not converge; cannot seed the "
            "GPU solver. Check the grid and solver settings.")
    return v_init, v_converged


def extract_grid_arrays(grid, v_init=None, max_iter=10, tol=1e-8):
    """Extract the AC solver inputs gpusim2grid needs from a solved grid.

    Parameters
    ----------
    grid : lightsim2grid GridModel / LSGrid
        Already configured with a solver (e.g. KLU).
    v_init : complex ndarray, optional
        DC warm-start. Computed via ``grid.dc_pf`` when omitted.
    max_iter, tol : int, float
        Passed to the CPU ``grid.ac_pf`` reference solve.

    Returns
    -------
    dict with keys: ``n_bus``, ``Ybus`` (scipy CSR complex), ``Sbus``
    (complex128), ``pv``, ``pq``, ``slack`` (int32), ``slack_weights``
    (float64, sums to 1), ``v_init`` (DC), ``v_converged`` (AC).
    """
    v_init, v_converged = _ensure_solved(grid, v_init, max_iter, tol)

    Ybus = grid.get_Ybus_solver().copy()
    Sbus = grid.get_Sbus_solver().copy()
    pv = grid.get_pv_solver().copy()
    pq = grid.get_pq_solver().copy()
    slack = grid.get_slack_ids_solver().copy()

    n_bus = Ybus.shape[0]
    slack_weights = np.zeros(n_bus, dtype=float)
    slack_weights[slack] = 1.0 / slack.shape[0]

    return {
        "n_bus": n_bus,
        "Ybus": Ybus,
        "Sbus": Sbus,
        "pv": pv,
        "pq": pq,
        "slack": slack,
        "slack_weights": slack_weights,
        "v_init": v_init,
        "v_converged": v_converged,
    }


def extract_branch_data(grid):
    """Effective π-model branch admittances for ``set_branch_data``.

    Branches are ordered lines-then-trafos: branch index ``c < n_lines`` is
    line ``c``; ``c >= n_lines`` is trafo ``c - n_lines``.  This matches
    lightsim2grid's ``add_all_n1()`` and gpusim2grid's branch-flow output.

    Values come straight from lightsim2grid (``get_yac_eff_*``); they are never
    recomputed — tap reference side, asymmetric shunts, etc. are error-prone.

    Returns
    -------
    (args, n_lines, n_trafos) where ``args`` is the tuple expected by
    ``set_branch_data``: (branch_from, branch_to, yff, yft, ytf, ytt,
    bus_vn_kv, sn_mva).
    """
    lines = grid.get_lines()
    trafos = grid.get_trafos()

    # branch_from/branch_to/vn_kv must all be in AC-solver bus numbering (like
    # bus_vmin_kv/bus_vmax_kv in gpu_facade.py's _extract_limits_arrays), but
    # lines.get_bus_id_side_*() and get_bus_vn_kv() both return grid-MODEL
    # numbering -- relabel via id_me_to_ac_solver() (-1 = isolated bus),
    # mirroring ls2g_bridge.cpp's concat_busids_to_solver()/relabel() exactly.
    # Left unrelabeled: branch endpoints index past V's solver-sized length
    # (IndexError in compute_branch_flows_cpu, confirmed on a real
    # ~47k-bus/7270-solver-bus grid) and/or vn_kv mismatches V in size (crash
    # in get_violations_n()) or silently pairs the wrong nominal voltage with
    # the wrong bus -- whenever the grid's model->solver bus map isn't the
    # identity (e.g. isolated buses excluded under KLU/the augmented
    # multi-slack system, or half-open lines).
    # Branch endpoints come in as arbitrary model-bus ids, so relabeling them
    # needs the forward map (model id -> solver id, or -1): id_me_to_ac_solver().
    # The inverse map id_ac_solver_to_me() (used below for vn_kv, where every
    # solver slot needs filling) doesn't help here -- it would have to be
    # inverted back into a forward map first, which is more work, not less.
    me_to_solver = np.asarray(grid.id_me_to_ac_solver())

    branch_from = _relabel_to_solver(me_to_solver, np.concatenate(
        (lines.get_bus_id_side_1(), trafos.get_bus_id_side_1())))
    branch_to = _relabel_to_solver(me_to_solver, np.concatenate(
        (lines.get_bus_id_side_2(), trafos.get_bus_id_side_2())))
    yff = np.concatenate(
        (lines.get_yac_eff_11().copy(), trafos.get_yac_eff_11().copy()))
    yft = np.concatenate(
        (lines.get_yac_eff_12().copy(), trafos.get_yac_eff_12().copy()))
    ytf = np.concatenate(
        (lines.get_yac_eff_21().copy(), trafos.get_yac_eff_21().copy()))
    ytt = np.concatenate(
        (lines.get_yac_eff_22().copy(), trafos.get_yac_eff_22().copy()))
    # Unlike branch endpoints above, every solver slot needs a vn_kv value, so
    # this is a straight gather through the inverse map (solver id -> model
    # id) instead of a scatter through the forward one.
    vn_kv_model = grid.get_bus_vn_kv().copy()
    solver_to_me = np.asarray(grid.id_ac_solver_to_me())
    if solver_to_me.size == 0:
        vn_kv = vn_kv_model   # identity numbering
    else:
        vn_kv = vn_kv_model[solver_to_me]
    sn_mva = grid.get_sn_mva()
    args = (branch_from, branch_to, yff, yft, ytf, ytt, vn_kv, sn_mva)
    return args, len(lines), len(trafos)


@dataclass(frozen=True)
class InjectionElements:
    """Snapshot letting per-element (load_p, load_q, gen_p) become a per-bus Sbus.

    Built once by :func:`extract_injection_elements` off a solved grid and
    consumed by :func:`build_bus_injections`.  It is a *snapshot*: later
    ``grid.change_p_load(...)`` / topology changes do not propagate, matching
    the GPU session's own Ybus/Sbus snapshot semantics.
    """

    n_load: int
    n_gen: int
    n_bus: int
    sn_mva: float
    # Indices of the elements that actually contribute (status on). Disconnected
    # elements are dropped entirely, like ``TimeSeries::fill_SBus_real``.
    load_sel: np.ndarray
    gen_sel: np.ndarray
    # (n_bus, n_selected) 0/1 scipy CSR incidence: element -> its AC-solver bus.
    scatter_load: Any
    scatter_gen: Any
    # Everything the three arrays do NOT vary, in MW/MVAr, solver numbering:
    # HVDC converter stations, static gens, storages, REACTIVE_POWER-mode SVCs,
    # and the Q of non-voltage-regulating ("PQ") generators.
    const_mw: np.ndarray
    # (n_gen,) AC-solver bus id of every generator, -1 for a disconnected one
    # (deliberately, not its stale model bus id -- see set_gen_v()'s doc on
    # why a disconnected generator's column must never alias a still-connected
    # one's bus). Used by ScenarioSweepGPU.set_gen_v() / InjectionSweepGPU.set_gen_v().
    gen_bus: np.ndarray


def extract_injection_elements(grid, n_bus):
    """Snapshot the load/generator -> solver-bus wiring of a *solved* grid.

    Reproduces the accumulation ``LSGrid::fillSbus_me`` performs -- loads
    subtract ``P + jQ``, generators add ``P`` -- but only for the loads and
    generators, keeping every other contributor in a constant term derived from
    the grid's own ``get_Sbus_solver()``.  Feeding the grid's own base values
    back through :func:`build_bus_injections` therefore reproduces the base-case
    Sbus exactly.

    Parameters
    ----------
    grid : lightsim2grid GridModel / LSGrid, already AC-solved.
    n_bus : int — the AC-solver bus count (``session.n_bus``).

    Raises
    ------
    ValueError
        If the grid has no computed results, if its solver Sbus does not have
        ``n_bus`` entries, or if a *connected* load/generator sits on a bus the
        solver dropped (lightsim2grid throws in the same situation).
    """
    from scipy.sparse import csr_matrix

    loads = grid.get_loads()
    gens = grid.get_generators()
    n_load = len(loads)
    n_gen = len(gens)

    sbus_base = np.asarray(grid.get_Sbus_solver(), dtype=np.complex128)
    if sbus_base.shape[0] != n_bus:
        raise ValueError(
            f"grid.get_Sbus_solver() has {sbus_base.shape[0]} entries but the "
            f"solver has {n_bus} buses.")

    me_to_solver = np.asarray(grid.id_me_to_ac_solver())
    load_status = np.asarray(grid.get_loads_status(), dtype=bool)
    gen_status = np.asarray(grid.get_gen_status(), dtype=bool)
    load_bus = _relabel_to_solver(me_to_solver, np.asarray(loads.get_bus_id()))
    gen_bus = _relabel_to_solver(me_to_solver, np.asarray(gens.get_bus_id()))

    def _check_alive(name, status, bus):
        bad = np.flatnonzero(status & ((bus < 0) | (bus >= n_bus)))
        if bad.size:
            raise ValueError(
                f"{name} {bad[:10].tolist()} are connected but sit on a bus the "
                f"AC solver dropped (isolated / Kron-reduced). lightsim2grid "
                f"raises here too -- reconnect them or deactivate them.")

    _check_alive("loads", load_status, load_bus)
    _check_alive("generators", gen_status, gen_bus)

    load_sel = np.flatnonzero(load_status)
    gen_sel = np.flatnonzero(gen_status)

    # Loads are a pure-PQ container, so lightsim2grid's *results* are literally
    # the targets it stamped into Sbus (OneSideContainer_PQ::set_osc_pq_res_p /
    # _res_q), which gives us both P and Q vectorized in one call. Generators
    # are NOT symmetric here: GeneratorContainer::set_p_slack folds the absorbed
    # slack mismatch into a slack gen's res_p, so gen base P must come from the
    # targets instead.
    if n_load and not loads[0].has_res:
        raise ValueError(
            "grid has no computed results: call grid.ac_pf(...) before building "
            "an element-level injection sweep.")
    load_res = grid.get_loads_res_full()
    load_p_base = np.asarray(load_res[0], dtype=np.float64)
    load_q_base = np.asarray(load_res[1], dtype=np.float64)
    gen_p_base = np.asarray(grid.get_gen_target_p(), dtype=np.float64)

    def _scatter(bus, sel):
        return csr_matrix(
            (np.ones(sel.size), (bus[sel], np.arange(sel.size))),
            shape=(n_bus, sel.size))

    scatter_load = _scatter(load_bus, load_sel)
    scatter_gen = _scatter(gen_bus, gen_sel)

    const_mw = sbus_base * float(grid.get_sn_mva())
    const_mw -= scatter_gen @ gen_p_base[gen_sel]
    const_mw += scatter_load @ (load_p_base[load_sel]
                                + 1j * load_q_base[load_sel])

    gen_bus_out = np.where(gen_status, gen_bus, -1)

    return InjectionElements(
        n_load=n_load, n_gen=n_gen, n_bus=int(n_bus),
        sn_mva=float(grid.get_sn_mva()),
        load_sel=load_sel, gen_sel=gen_sel,
        scatter_load=scatter_load, scatter_gen=scatter_gen,
        const_mw=const_mw, gen_bus=gen_bus_out)


def build_bus_injections(elements, load_p, load_q, gen_p):
    """Assemble per-bus (p_mw, q_mvar) from per-element injection matrices.

    Parameters
    ----------
    elements : InjectionElements — from :func:`extract_injection_elements`.
    load_p, load_q : (n_steps, n_load) float arrays, MW / MVAr.
    gen_p : (n_steps, n_gen) float array, MW.

    Returns
    -------
    (p_mw, q_mvar), each ``(n_steps, n_bus)`` float64 in AC-solver bus
    numbering — the arrays ``set_injections`` expects.
    """
    def _check(mat, name, n_expected_cols):
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim != 2:
            raise ValueError(
                f"'{name}' must be 2-D (n_steps, n_elements), got shape "
                f"{mat.shape}.")
        if mat.shape[1] != n_expected_cols:
            raise ValueError(
                f"'{name}' has {mat.shape[1]} columns while the grid counts "
                f"{n_expected_cols} such elements.")
        return mat

    load_p = _check(load_p, "load_p", elements.n_load)
    load_q = _check(load_q, "load_q", elements.n_load)
    gen_p = _check(gen_p, "gen_p", elements.n_gen)

    n_steps = gen_p.shape[0]
    if n_steps == 0:
        raise ValueError("injection matrices must have at least one row (step).")
    for mat, name in ((load_p, "load_p"), (load_q, "load_q")):
        if mat.shape[0] != n_steps:
            raise ValueError(
                f"'{name}' has {mat.shape[0]} rows (time steps) while 'gen_p' "
                f"has {n_steps}. All injection matrices must share the same "
                f"number of rows.")

    sbus_mw = np.empty((n_steps, elements.n_bus), dtype=np.complex128)
    sbus_mw[:] = elements.const_mw
    if elements.gen_sel.size:
        sbus_mw += (elements.scatter_gen @ gen_p[:, elements.gen_sel].T).T
    if elements.load_sel.size:
        load_s = (load_p[:, elements.load_sel]
                  + 1j * load_q[:, elements.load_sel])
        sbus_mw -= (elements.scatter_load @ load_s.T).T

    return (np.ascontiguousarray(sbus_mw.real),
            np.ascontiguousarray(sbus_mw.imag))


def grid_from_pandapower(net, solver_type="NR_KLU"):
    """Convert a pandapower network to a configured lightsim2grid grid.

    ``solver_type`` is a registered algorithm name (see the grid's
    ``available_solver_names()``), e.g. ``"NR_KLU"`` or ``"NR_SparseLU"``.
    """
    try:
        from lightsim2grid.network import init_from_pandapower
    except ImportError:
        # Older lightsim2grid keeps the converter under .gridmodel.
        from lightsim2grid.gridmodel import init_from_pandapower

    grid = init_from_pandapower(net)
    try:
        grid.change_algorithm(solver_type)
    except (AttributeError, RuntimeError):
        # Older lightsim2grid: select via SolverType enum on change_solver.
        from lightsim2grid.solver import SolverType

        grid.change_solver(getattr(SolverType, solver_type.replace("NR_", "")))
    return grid

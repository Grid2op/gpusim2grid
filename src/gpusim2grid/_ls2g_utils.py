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

import numpy as np


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

    # TODO(bug, non-bridge/array path only): Ybus/Sbus are read in AC-SOLVER
    # numbering (get_*_solver()) but pv/pq/slack below are read in
    # GRID-MODEL numbering (get_pv()/get_pq()/get_slack_ids()), unlike the
    # bridge path (ls2g_bridge.cpp) which consistently uses the
    # *_solver_numpy() accessors for all of these. When model numbering !=
    # solver numbering this mismatches the bus partition against Ybus/Sbus,
    # which can produce a large/garbage NR residual. Needs get_pv_solver_numpy()
    # / get_pq_solver_numpy() / get_slack_ids_solver_numpy() here instead, if
    # those exist on lightsim2grid versions this fallback path supports.
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
    branch_from = np.concatenate(
        (lines.get_bus_id_side_1(), trafos.get_bus_id_side_1()))
    branch_to = np.concatenate(
        (lines.get_bus_id_side_2(), trafos.get_bus_id_side_2()))
    yff = np.concatenate(
        (lines.get_yac_eff_11().copy(), trafos.get_yac_eff_11().copy()))
    yft = np.concatenate(
        (lines.get_yac_eff_12().copy(), trafos.get_yac_eff_12().copy()))
    ytf = np.concatenate(
        (lines.get_yac_eff_21().copy(), trafos.get_yac_eff_21().copy()))
    ytt = np.concatenate(
        (lines.get_yac_eff_22().copy(), trafos.get_yac_eff_22().copy()))
    # TODO(bug): vn_kv is left in grid-MODEL bus numbering, unlike
    # branch_from/branch_to above (whatever numbering lines.get_bus_id_side_*
    # already returns -- solver numbering per the ls2g_bridge.cpp analog) and
    # unlike bus_vmin_kv/bus_vmax_kv in gpu_facade.py's _extract_limits_arrays
    # (explicitly relabeled via id_me_to_ac_solver()). Whenever the grid's
    # model->solver bus map isn't the identity (e.g. isolated buses excluded
    # under KLU/the augmented multi-slack system), this mismatches V in size
    # (crash in get_violations_n) or silently pairs the wrong nominal voltage
    # with the wrong bus. Needs the same id_me_to_ac_solver() remap as
    # _extract_limits_arrays.
    vn_kv = grid.get_bus_vn_kv().copy()
    sn_mva = grid.get_sn_mva()
    args = (branch_from, branch_to, yff, yft, ytf, ytt, vn_kv, sn_mva)
    return args, len(lines), len(trafos)


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

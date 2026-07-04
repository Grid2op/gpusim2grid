"""
Shared fixtures for gpusim2grid pytest suite.

Grid: IEEE 14-bus (pandapower case14 → lightsim2grid).
Scope: session — the grid solve is expensive; all tests share the same objects.
"""
import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# GPU availability — skip entire test module if the extension isn't installed
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    try:
        import gpusim2grid._gpusim2grid  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="gpusim2grid C extension not installed / no CUDA GPU",
)


def _is_fp32() -> bool:
    """True when the extension was compiled with GPUSIM2GRID_REAL_FLOAT=ON."""
    try:
        from gpusim2grid._gpusim2grid import is_fp32 as _compiled_fp32
        return bool(_compiled_fp32)
    except (ImportError, AttributeError):
        return os.environ.get("CUDA_REAL_FLOAT", "0") not in ("", "0")


# ---------------------------------------------------------------------------
# Voltage / residual tolerances that depend on compiled precision
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def solver_atol() -> float:
    """Absolute tolerance for voltage magnitude / angle comparisons."""
    return 1e-4 if _is_fp32() else 1e-6


@pytest.fixture(scope="session")
def residual_atol() -> float:
    """Absolute tolerance for power-balance residual |F(V)|∞."""
    return 1e-4 if _is_fp32() else 1e-6


# ---------------------------------------------------------------------------
# Core lightsim2grid / pandapower grid
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ieee14_grid():
    """lightsim2grid GridModel for the IEEE 14-bus network.

    Solver is set to KLU (same as the production pipeline) so that the
    reference voltages are as accurate as possible.
    """
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower
    from lightsim2grid.solver import SolverType

    grid = init_from_pandapower(pn.case14())
    grid.change_solver(SolverType.KLU)
    return grid


@pytest.fixture(scope="session")
def ieee14_base_case(ieee14_grid):
    """Fully solved base-case for the IEEE 14-bus network.

    Returns a dict with:
      Ybus, dcYbus, Sbus, dcSbus, pv, pq, slack, slack_weights,
      v_init (DC-warm-started), v_ref (AC reference from lightsim2grid),
      v_init_dc (flat-start for DC), n_bus.
    """
    grid = ieee14_grid
    n_bus = grid.get_bus_vn_kv().shape[0]

    # --- DC warm-start ---
    v_init_dc = np.ones(n_bus, dtype=complex)
    v_init = grid.dc_pf(v_init_dc.copy(), 1, 1e-6)
    assert v_init.shape[0] == n_bus, "DC PF returned unexpected size"

    # --- AC reference (tight tolerance, many iterations) ---
    v_ref = grid.ac_pf(v_init.copy(), 20, 1e-8)
    assert v_ref.shape[0] == n_bus, "AC PF returned unexpected size"
    assert np.all(np.abs(v_ref) > 1e-6), "Reference AC solve produced zero voltages"

    # --- Matrices and index arrays ---
    Ybus: csr_matrix = grid.get_Ybus_solver().copy()
    Sbus: np.ndarray = grid.get_Sbus_solver().copy()
    dcYbus: csr_matrix = grid.get_dcYbus_solver().copy()
    dcSbus: np.ndarray = grid.get_dcSbus_solver().copy()

    pv = grid.get_pv()
    pq = grid.get_pq()
    slack = grid.get_slack_ids()
    slack_weights = np.zeros(n_bus, dtype=float)
    slack_weights[slack] = 1.0 / slack.shape[0]

    return {
        "grid": grid,
        "n_bus": n_bus,
        "Ybus": Ybus,
        "Sbus": Sbus,
        "dcYbus": dcYbus,
        "dcSbus": dcSbus,
        "pv": pv,
        "pq": pq,
        "slack": slack,
        "slack_weights": slack_weights,
        "v_init_dc": v_init_dc,
        "v_init": v_init,
        "v_ref": v_ref,
    }


# ---------------------------------------------------------------------------
# Contingency helpers
# ---------------------------------------------------------------------------

def branch_triplets(gridmodel, branch_id: int, *, is_powerline: bool = True):
    """Return the four (row, col, yr, yi) Ybus entries for a branch.

    These entries represent the admittance contribution of the branch.
    Subtracting them from Ybus simulates an N-1 disconnection.
    """
    branch = (gridmodel.get_lines() if is_powerline else gridmodel.get_trafos())[branch_id]
    i, j = branch.bus1_id, branch.bus2_id
    return [
        (i, i, branch.yac_11.real, branch.yac_11.imag),
        (j, j, branch.yac_22.real, branch.yac_22.imag),
        (i, j, branch.yac_12.real, branch.yac_12.imag),
        (j, i, branch.yac_21.real, branch.yac_21.imag),
    ]


def branch_data_arrays(grid):
    """ContingencyAnalysisSolver.set_branch_data args (effective π-model
    admittances) plus (n_lines, n_trafos).

    Branch index c < n_lines is line c; c >= n_lines is trafo (c - n_lines) —
    the lines-then-trafos order used by build_contingencies([[c], ...]) and by
    lightsim2grid's add_all_n1().
    """
    lines  = grid.get_lines()
    trafos = grid.get_trafos()
    branch_from = np.concatenate((lines.get_bus_id_side_1(),  trafos.get_bus_id_side_1()))
    branch_to   = np.concatenate((lines.get_bus_id_side_2(),  trafos.get_bus_id_side_2()))
    yff = np.concatenate((lines.get_yac_eff_11().copy(), trafos.get_yac_eff_11().copy()))
    yft = np.concatenate((lines.get_yac_eff_12().copy(), trafos.get_yac_eff_12().copy()))
    ytf = np.concatenate((lines.get_yac_eff_21().copy(), trafos.get_yac_eff_21().copy()))
    ytt = np.concatenate((lines.get_yac_eff_22().copy(), trafos.get_yac_eff_22().copy()))
    vn_kv  = grid.get_bus_vn_kv().copy()
    sn_mva = grid.get_sn_mva()
    branch_data = (branch_from, branch_to, yff, yft, ytf, ytt, vn_kv, sn_mva)
    return branch_data, len(lines), len(trafos)


def branch_eff_triplet(branch_data, c):
    """The four (row, col, yr, yi) Ybus entries for branch c, built from the
    effective admittances in branch_data.  Subtracting them from Ybus removes
    branch c — matching the patch ContingencyAnalysisSolver applies internally.
    """
    branch_from, branch_to, yff, yft, ytf, ytt = branch_data[:6]
    i, j = int(branch_from[c]), int(branch_to[c])
    return [
        (i, i, yff[c].real, yff[c].imag),
        (j, j, ytt[c].real, ytt[c].imag),
        (i, j, yft[c].real, yft[c].imag),
        (j, i, ytf[c].real, ytf[c].imag),
    ]


@pytest.fixture(scope="session")
def ieee14_contingencies(ieee14_grid):
    """All N-1 contingencies for IEEE 14: one per line, then one per trafo.

    Returns (contingencies, n_lines, n_trafos).
    """
    grid = ieee14_grid
    lines = grid.get_lines()
    trafos = grid.get_trafos()
    n_lines = len(lines)
    n_trafos = len(trafos)

    contingencies = (
        [branch_triplets(grid, lid, is_powerline=True)  for lid in range(n_lines)] +
        [branch_triplets(grid, lid, is_powerline=False) for lid in range(n_trafos)]
    )
    return contingencies, n_lines, n_trafos


@pytest.fixture(scope="session")
def ieee14_cpu_n1_reference(ieee14_grid, ieee14_base_case):
    """Run ContingencyAnalysisCPP (lightsim2grid CPU reference) on IEEE 14.

    Returns (V_ref_2d, n_lines, n_trafos) where V_ref_2d has shape
    (n_contingencies, 2*n_bus) — lightsim2grid uses double-bus topology.
    """
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    grid = ieee14_grid
    n_bus = ieee14_base_case["n_bus"]
    v_init = ieee14_base_case["v_init"]
    v_init_ca = 1. * v_init

    ca = ContingencyAnalysisCPP(grid)
    ca.add_all_n1()
    ca.compute(v_init_ca, 20, 1e-7)
    V_ref = ca.get_voltages()          # shape: (n_ctg, 2*n_bus)

    lines = grid.get_lines()
    trafos = grid.get_trafos()
    return V_ref, len(lines), len(trafos)


# ---------------------------------------------------------------------------
# ContingencyAnalysisSolver with branch flows (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ieee14_solver_with_flows(ieee14_grid, ieee14_base_case):
    """Run ContingencyAnalysisSolver on IEEE 14 and compute branch flows.

    Returns a dict with:
      or_amps       — shape (n_ctg, n_branches), ampere flows at origin side
      residuals     — shape (n_ctg,)
      cont_branch_ids — list of [branch_id] per contingency (branch index = line
                        for lid < n_lines, trafo for lid >= n_lines)
      n_lines, n_trafos, n_branches
    """
    from gpusim2grid.contingency_analysis import ContingencyAnalysisSolver

    grid = ieee14_grid
    d = ieee14_base_case

    lines  = grid.get_lines()
    trafos = grid.get_trafos()
    n_lines  = len(lines)
    n_trafos = len(trafos)
    n_branches = n_lines + n_trafos

    branch_from = np.concatenate((lines.get_bus_id_side_1(),  trafos.get_bus_id_side_1()))
    branch_to   = np.concatenate((lines.get_bus_id_side_2(),  trafos.get_bus_id_side_2()))
    yff = np.concatenate((lines.get_yac_eff_11().copy(), trafos.get_yac_eff_11()))
    yft = np.concatenate((lines.get_yac_eff_12().copy(), trafos.get_yac_eff_12()))
    ytf = np.concatenate((lines.get_yac_eff_21().copy(), trafos.get_yac_eff_21()))
    ytt = np.concatenate((lines.get_yac_eff_22().copy(), trafos.get_yac_eff_22()))
    vn_kv = grid.get_bus_vn_kv().copy()
    sn_mva = grid.get_sn_mva()

    cont_branch_ids = [[l] for l in range(n_lines)] + [[n_lines + t] for t in range(n_trafos)]

    solver = ContingencyAnalysisSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=5, nb_iter=10, max_iter_base=10, tol_base=1e-6,
    )
    solver.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt, vn_kv, sn_mva)
    solver.build_contingencies(cont_branch_ids)
    solver.run()
    solver.compute_flows()

    or_amps   = solver.or_amps.to_numpy().reshape(n_branches, n_branches)
    residuals = solver.residuals.to_numpy()

    return {
        "or_amps":          or_amps,
        "residuals":        residuals,
        "cont_branch_ids":  cont_branch_ids,
        "n_lines":          n_lines,
        "n_trafos":         n_trafos,
        "n_branches":       n_branches,
    }


# ---------------------------------------------------------------------------
# Power-balance mismatch (pure-NumPy, no GPU)
# ---------------------------------------------------------------------------

def compute_pq_mismatch(
    Ybus: csr_matrix,
    V: np.ndarray,
    Sbus: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
) -> np.ndarray:
    """Return the NR mismatch vector F = [ΔP_pvpq | ΔQ_pq] (pu)."""
    Ibus = Ybus @ V
    Scalc = V * np.conj(Ibus)
    mis = Scalc - Sbus
    pvpq = np.r_[pv, pq]
    return np.r_[mis[pvpq].real, mis[pq].imag]

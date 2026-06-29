"""
Shared helpers for the gpusim2grid examples.

These wrap lightsim2grid / pandapower to produce the plain NumPy / SciPy arrays
that gpusim2grid consumes (Ybus, Sbus, pv/pq/slack indices, branch admittances).
gpusim2grid itself never depends on pandapower — it only consumes these arrays.
"""
import numpy as np
from scipy.sparse import csr_matrix


def load_case(name="case14"):
    """Load a pandapower network and solve its AC base case on the CPU.

    Parameters
    ----------
    name : str
        Any ``pandapower.networks`` factory name, e.g. ``"case14"`` or
        ``"case6515rte"``.

    Returns
    -------
    dict with keys: grid, n_bus, Ybus, Sbus, pv, pq, slack, slack_weights,
    v_init (DC warm-start), v_ref (AC reference from lightsim2grid).
    """
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower
    from lightsim2grid.solver import SolverType

    if not hasattr(pn, name):
        raise ValueError(f"{name!r} is not a pandapower.networks factory")

    grid = init_from_pandapower(getattr(pn, name)())
    grid.change_solver(SolverType.KLU)

    n_bus = grid.get_bus_vn_kv().shape[0]

    # DC warm-start, then a tight AC solve as the reference oracle.
    v_init_dc = np.ones(n_bus, dtype=complex)
    v_init = grid.dc_pf(v_init_dc.copy(), 1, 1e-6)
    v_ref = grid.ac_pf(v_init.copy(), 20, 1e-8)

    Ybus: csr_matrix = grid.get_Ybus_solver().copy()
    Sbus: np.ndarray = grid.get_Sbus_solver().copy()
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
        "pv": pv,
        "pq": pq,
        "slack": slack,
        "slack_weights": slack_weights,
        "v_init": v_init,
        "v_ref": v_ref,
    }


def branch_data(grid):
    """Effective pi-model branch admittances for ``set_branch_data``.

    Branches are ordered lines-then-trafos: branch index ``c < n_lines`` is
    line ``c``; ``c >= n_lines`` is trafo ``c - n_lines``. This matches
    lightsim2grid's ``add_all_n1()``.

    Returns
    -------
    (args, n_lines, n_trafos) where ``args`` is the tuple expected by
    ``set_branch_data``: (branch_from, branch_to, yff, yft, ytf, ytt,
    bus_vn_kv, sn_mva).
    """
    lines = grid.get_lines()
    trafos = grid.get_trafos()
    branch_from = np.concatenate((lines.get_bus_id_side_1(), trafos.get_bus_id_side_1()))
    branch_to = np.concatenate((lines.get_bus_id_side_2(), trafos.get_bus_id_side_2()))
    yff = np.concatenate((lines.get_yac_eff_11().copy(), trafos.get_yac_eff_11().copy()))
    yft = np.concatenate((lines.get_yac_eff_12().copy(), trafos.get_yac_eff_12().copy()))
    ytf = np.concatenate((lines.get_yac_eff_21().copy(), trafos.get_yac_eff_21().copy()))
    ytt = np.concatenate((lines.get_yac_eff_22().copy(), trafos.get_yac_eff_22().copy()))
    vn_kv = grid.get_bus_vn_kv().copy()
    sn_mva = grid.get_sn_mva()
    args = (branch_from, branch_to, yff, yft, ytf, ytt, vn_kv, sn_mva)
    return args, len(lines), len(trafos)

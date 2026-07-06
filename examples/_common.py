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
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    if not hasattr(pn, name):
        raise ValueError(f"{name!r} is not a pandapower.networks factory")

    grid = init_from_pandapower(getattr(pn, name)())
    grid.change_algorithm(AlgorithmType.NR_KLU)

    n_bus = grid.get_bus_vn_kv().shape[0]

    # DC warm-start, then a tight AC solve as the reference oracle.
    v_init_dc = np.ones(n_bus, dtype=complex)
    v_init = grid.dc_pf(v_init_dc.copy(), 1, 1e-6)
    v_ref = grid.ac_pf(v_init.copy(), 20, 1e-8)

    # Solver-numbered arrays, consistent with get_Ybus_solver().
    Ybus: csr_matrix = grid.get_Ybus_solver().copy()
    Sbus: np.ndarray = grid.get_Sbus_solver().copy()
    pv = grid.get_pv_solver()
    pq = grid.get_pq_solver()
    slack = grid.get_slack_ids_solver()
    # (n_bus,) distributed-slack weights, summing to 1 over the slack buses.
    slack_weights = grid.get_slack_weights_solver().copy()

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


def print_acpf_timings(timings, label="Timings"):
    """Hierarchical breakdown of an ``AcPfTimings`` (single power flow):

    total, then the 4 coarse buckets it's made of, then each bucket's own
    sub-steps. CPU preprocessing has no further breakdown (it's a single
    CPU-only measurement), so it has no detail section.
    """
    print(f"{label} (ms):")
    print(f"  TOTAL                 : {timings.t_grand_total_ms:9.3f}")
    print(f"    CPU preprocessing   : {timings.t_cpu_preprocess_ms:9.3f}")
    print(f"    Host -> Device      : {timings.t_host_to_device_ms:9.3f}")
    print(f"    GPU compute         : {timings.t_gpu_compute_ms:9.3f}")
    print(f"    Device -> Host      : {timings.t_device_to_host_ms:9.3f}")

    print("  Host -> Device detail:")
    _cuda_setup_ms = timings.t_host_to_device_ms - timings.t_upload_ms
    print(f"    H->D data transfers               : {timings.t_upload_ms:9.3f}")
    print(f"    cuSPARSE/cuDSS setup (remainder)   : {_cuda_setup_ms:9.3f}")

    print("  GPU compute detail:")
    print(f"    cuDSS analysis        : {timings.t_analyze_ms:9.3f}")
    print(f"    Adjoint (Jt) setup    : {timings.t_prepare_jt_ms:9.3f}")
    print(f"    SpMV                  : {timings.t_spmv.wall_ms:9.3f}")
    print(f"    Fill F                : {timings.t_fill_F.wall_ms:9.3f}")
    print(f"    Fill J                : {timings.t_fill_J.wall_ms:9.3f}")
    print(f"    Factorize (1st)       : {timings.t_first_factorize.wall_ms:9.3f}")
    print(f"    Refactorize ({timings.n_refactorize}x)     : {timings.t_refactorize.wall_ms:9.3f}")
    print(f"    Solve                 : {timings.t_solve.wall_ms:9.3f}")
    print(f"    Update V              : {timings.t_update_V.wall_ms:9.3f}")
    print(f"    Mismatch check        : {timings.t_mismatch.wall_ms:9.3f}")

    print("  Device -> Host detail:")
    print(f"    Copy V to host        : {timings.t_copy_v_to_host_ms:9.3f}")


def print_batch_timings(timings, label="Timings", unit="item", unit_plural=None):
    """Hierarchical breakdown of a ``BatchTimings`` (contingency / injection
    sweep): total, then the 4 coarse buckets, then each bucket's sub-steps.

    ``unit`` names what's being batched ("contingency" / "scenario") for the
    per-item throughput line; ``unit_plural`` overrides the default "+s" for
    irregular plurals ("contingency" -> "contingencies"). CPU preprocessing
    has no further breakdown exposed (it's one accumulated CPU-only
    measurement).
    """
    if unit_plural is None:
        unit_plural = unit + "s"
    print(f"{label} (ms):")
    print(f"  TOTAL                  : {timings.t_grand_total_ms:10.3f}")
    print(f"    CPU preprocessing    : {timings.t_cpu_preprocess_ms:10.3f}")
    print(f"    Host -> Device       : {timings.t_host_to_device_ms:10.3f}")
    print(f"    GPU compute          : {timings.t_gpu_compute_ms:10.3f}")
    print(f"    Device -> Host       : {timings.t_device_to_host_ms:10.3f}")

    print("  Host -> Device detail:")
    print(f"    Base-case + batch alloc/upload : {timings.t_alloc_ms:10.3f}")
    print(f"    Source-specific init           : {timings.t_source_init_ms:10.3f}")
    print(f"    Branch-data upload             : {timings.t_branch_data_upload_ms:10.3f}")

    print(f"  GPU compute detail ({timings.n_chunks} chunk(s)):")
    print(f"    Base-case solve (analyze + NR) : {timings.t_base_case_solve_only_ms:10.3f}")
    print(f"    cuDSS analysis (batch)         : {timings.t_analysis_ms:10.3f}")
    print(f"    Tile V                         : {timings.t_tile_V.wall_ms:10.3f}")
    print(f"    Tile Ybus                      : {timings.t_tile_Ybus.wall_ms:10.3f}")
    print(f"    Tile Sbus                      : {timings.t_tile_Sbus.wall_ms:10.3f}")
    print(f"    Patch Ybus                     : {timings.t_patch_Ybus.wall_ms:10.3f}")
    print(f"    SpMV                           : {timings.t_spmv.wall_ms:10.3f}")
    print(f"    Fill F                         : {timings.t_fill_F.wall_ms:10.3f}")
    print(f"    Fill J                         : {timings.t_fill_J.wall_ms:10.3f}")
    print(f"    Factorize (1st)                : {timings.t_first_factorize.wall_ms:10.3f}")
    print(f"    Refactorize ({timings.n_refactorize}x)              : {timings.t_refactorize.wall_ms:10.3f}")
    print(f"    Solve                           : {timings.t_solve.wall_ms:10.3f}")
    print(f"    Update V                        : {timings.t_update_V.wall_ms:10.3f}")
    print(f"    Residual eval                   : {timings.t_residual.wall_ms:10.3f}")
    print(f"    Store V                         : {timings.t_store_V.wall_ms:10.3f}")
    print(f"    Branch flows                    : {timings.t_flow_computation.wall_ms:10.3f}")

    print("  Device -> Host detail:")
    print(f"    Copy flows to host              : {timings.t_copy_flows_to_host_ms:10.3f}")
    print(f"    Copy V to host                   : {timings.t_copy_V_to_host_ms:10.3f}")
    print(f"    Copy residuals to host           : {timings.t_copy_residuals_to_host_ms:10.3f}")

    print(f"  Per {unit}: {timings.t_per_contingency_ms:.4f} ms "
          f"({1000. / timings.t_per_contingency_ms:.0f} {unit_plural}/s)")


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

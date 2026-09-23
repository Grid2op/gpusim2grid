"""N-1 security analysis on real RTE snapshots: per grid, how many branch
contingencies trigger an outer loop (PHYSICAL violation: bus reactive capability
LOW_Q/HIGH_Q or hvdc droop P saturation) and how many end with ||F||_inf > KCL_TOL."""
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pypowsybl as pp
import pypowsybl.loadflow as lf

sys.path.insert(0, "/home/donnotben/Documents/gpusim2grid")
from test_real_data import _discover_pf_bug_files, _enable_max_voltage_change, PF_BUGS_ROOT  # noqa: E402
from lightsim2grid.network.from_pypowsybl import init, bake_outer_loops  # noqa: E402
from lightsim2grid.network.from_pypowsybl._olf_compare import iidm_bus_voltages  # noqa: E402

import gpusim2grid  # noqa: E402
from gpusim2grid import ContingencyAnalysisGPU  # noqa: E402
from gpusim2grid.contingency_analysis import LimitViolationType  # noqa: E402

KCL_TOL = 1e-4
NB_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 10
BAKE = not (len(sys.argv) > 4 and sys.argv[4] == "nobake")
BATCH = 256


def run_one(path):
    out = {"file": Path(path).name}
    t0 = time.perf_counter()
    net = pp.network.load(path, {'iidm.die.with-extensions': "all"})
    res = lf.run_ac(net, parameters=None)
    out["olf_status"] = str(res[0].status)
    out["baked"] = BAKE
    if BAKE:
        bake_outer_loops(net)
    v_olf = iidm_bus_voltages(net)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init(net, gen_slack_id=None, sort_index=False, buses_for_sub=False,
                    keep_half_open_lines=True, fuse_zero_impedance_branches=True)
    _enable_max_voltage_change(grid)
    v = v_olf["vm_pu"] * np.exp(1j * np.deg2rad(v_olf["va_deg"]))
    v[~np.isfinite(v)] = 1.
    v0 = np.ones(grid.get_bus_vn_kv().shape[0], dtype=complex)
    v0[grid._orig_to_ls] = v
    V = grid.ac_pf(v0, 300, 1e-6)
    out["t_prepare_s"] = time.perf_counter() - t0
    if V.shape[0] == 0:
        out["error"] = "lightsim2grid base case diverges"
        return out

    n_lines, n_trafos = len(grid.get_lines()), len(grid.get_trafos())
    out.update(n_bus_solver=int(grid.get_Ybus_solver().shape[0]), n_lines=n_lines, n_trafos=n_trafos,
               sn_mva=float(grid.get_sn_mva()))
    t0 = time.perf_counter()
    ca = ContingencyAnalysisGPU(grid, nb_iter=NB_ITER, handle_disconnected_grid=True,
                                compute_physical_violations=True)
    # default 16 records/row truncates the un-baked grids (base case alone has >= 16)
    ca.physical_violation_capacity = 512
    ctgs = [[b] for b in range(n_lines + n_trafos)]
    ca.add_contingencies_by_branch_id(ctgs)
    ca.compute(batch_size=BATCH)
    res = ca.last_residuals()
    out["t_gpu_s"] = time.perf_counter() - t0

    phys = ca.get_physical_violations()
    phys_n = ca.get_physical_violations_n()
    finite = np.isfinite(res)
    conv = finite & (res <= KCL_TOL)
    n_q = np.array([any(int(x.violation_type) in (LimitViolationType.LOW_Q, LimitViolationType.HIGH_Q)
                        for x in row) for row in phys])
    n_h = np.array([any(int(x.violation_type) == LimitViolationType.HVDC_P_SATURATION for x in row)
                    for row in phys])
    # violations already present in the base case do not count as "triggered by the contingency"
    base_keys = {(int(x.element_type), int(x.element_id), int(x.side), int(x.violation_type)) for x in phys_n}
    new = np.array([any((int(x.element_type), int(x.element_id), int(x.side), int(x.violation_type))
                        not in base_keys for x in row) for row in phys])
    out.update(
        n_ctg=len(ctgs),
        n_not_simulated_nan=int((~finite).sum()),
        n_kcl_above_tol=int((finite & (res > KCL_TOL)).sum()),
        n_converged=int(conv.sum()),
        n_outer_loop_any=int((n_q | n_h).sum()),
        n_outer_loop_bus_q=int(n_q.sum()),
        n_outer_loop_hvdc=int(n_h.sum()),
        n_outer_loop_new_vs_base=int(new.sum()),
        n_truncated=int(ca.get_physical_violations_truncated().sum()),
        base_physical_violations=len(phys_n),
        base_physical_types=dict(Counter(LimitViolationType(int(x.violation_type)).name for x in phys_n)),
        residual_p50=float(np.nanmedian(res)) if finite.any() else None,
        residual_max=float(np.nanmax(res)) if finite.any() else None,
        worst_ctg=[int(i) for i in np.argsort(np.where(finite, res, -1))[::-1][:5]],
    )
    return out


if __name__ == "__main__":
    files = list(_discover_pf_bug_files(PF_BUGS_ROOT).values())
    idx = [int(i) for i in sys.argv[1].split(",")]
    out_json = sys.argv[2]
    gpusim2grid.warmup()
    results = []
    for i in idx:
        path = files[i][1]
        print(f"=== [{i}] {path}", flush=True)
        try:
            r = run_one(path)
        except Exception as exc:  # keep going on the other grids
            r = {"file": Path(path).name, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(r), flush=True)
        results.append(r)
        Path(out_json).write_text(json.dumps(results, indent=1))

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Inputs the batch sessions must refuse, or normalise, before they reach the GPU.

- ``violation_capacity`` below 1: the operational check would write its DIVERGENCE
  record into an empty output buffer (an illegal address that poisons the CUDA context);
- ``benchmark_cudss_batch_raw`` on a malformed CSR matrix (indices or ``indptr`` out of
  range): host reads past the arrays, then cuDSS on a broken pattern;
- a repeated id in a contingency (a branch named twice, in ``branch_ids`` and a side list,
  or a shunt / load / generator named twice): tripped once, as lightsim2grid's
  ``std::set``-based contingencies, not subtracted twice;
- ``ScenarioSweepGPU.set_contingency_shunts`` after the row count changed: the branch
  trips it keeps are the implicit "none" ones, not a stale explicit topology;
- ``set_contingency_loads`` with a wrong column count, or one that no longer matches the
  rows: refused, not silently dropped.
"""
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge, is_fp32

pytestmark = requires_gpu
needs_bridge = pytest.mark.skipif(not have_ls2g_bridge, reason="needs the lightsim2grid C++ bridge")

MAX_IT, TOL = 20, 1e-10
# the GPU base case's tolerance: 1e-10 cannot be reached in FP32, which keeps the default
GPU_TOL = {} if is_fp32 else {"tol_base": TOL}


def _make_grid():
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(pn.case14())
    grid.change_algorithm("NR_KLU")
    v_ref = grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL)
    assert v_ref.shape[0] > 0
    return grid, v_ref


@pytest.fixture(scope="module")
def case():
    grid, v_ref = _make_grid()
    return {"grid": grid, "v_ref": v_ref, "n_bus": grid.get_Ybus_solver().shape[0],
            "buses": np.asarray(grid.id_ac_solver_to_me(), dtype=int)}


def _V(solver, n_rows, n_bus):
    return solver.V_results.to_numpy().reshape(n_rows, n_bus)


def _rep(a, n):
    return np.repeat(np.asarray(a, dtype=np.float64)[None, :], n, axis=0)


def _elements(grid, n):
    load_p, load_q = grid.get_loads_res_full()[:2]
    return _rep(load_p, n), _rep(load_q, n), _rep(grid.get_gen_target_p(), n)


# ---------------------------------------------------------------- violation_capacity
def _sessions(grid):
    from gpusim2grid import ContingencyAnalysisGPU, ScenarioSweepGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, **GPU_TOL)
    sw = ScenarioSweepGPU(grid, nb_iter=10, **GPU_TOL)
    return {"ca": ca, "ss": sw}


@needs_bridge
@pytest.mark.parametrize("kind", ["ca", "ss"])
@pytest.mark.parametrize("value", [0, -1])
def test_violation_capacity_below_one_is_refused(case, kind, value):
    """The engine property and the raw session attribute both refuse K < 1 at once, and
    the session keeps its previous capacity."""
    obj = _sessions(case["grid"])[kind]
    engine = obj._inner
    with pytest.raises(ValueError, match="violation_capacity"):
        engine.violation_capacity = value
    with pytest.raises(ValueError, match="violation_capacity"):
        engine._s.violation_capacity = value
    assert engine.violation_capacity == 16


@needs_bridge
@pytest.mark.parametrize("kind", ["ca", "ss"])
def test_violation_settings_set_on_the_facade_take_effect(case, kind):
    """violation_capacity and violation_tol set on the ContingencyAnalysisGPU /
    ScenarioSweepGPU facade reach the session (they were plain Python attributes, silently
    ignored): with K = 1 and every bus above its limit, each row keeps one record and is
    truncated, its total still exact; with a violation_tol no Newton step can reach, every
    row is one DIVERGENCE record."""
    from gpusim2grid import ContingencyAnalysisGPU, ScenarioSweepGPU
    from gpusim2grid.contingency_analysis import LimitViolationType

    grid = case["grid"]
    n_bus, n_rows = case["n_bus"], 2
    n_lines = len(grid.get_lines())
    n_br = n_lines + len(grid.get_trafos())
    vn = np.asarray(grid.get_bus_vn_kv(), dtype=float)[case["buses"]]
    if kind == "ca":
        obj = ContingencyAnalysisGPU(grid, nb_iter=10, compute_limit_violations=True, **GPU_TOL)
        obj.add_contingencies_by_branch_id([[0], [1]])
    else:
        obj = ScenarioSweepGPU(grid, nb_iter=10, compute_limit_violations=True, **GPU_TOL)
        obj.set_injections_from_elements(*_elements(grid, n_rows))
    obj.set_limits(0.1 * vn, 0.5 * vn, np.full(n_br, np.nan), np.full(n_br, np.nan), n_lines) \
        if kind == "ca" else obj._inner.set_limits(0.1 * vn, 0.5 * vn, np.full(n_br, np.nan),
                                                   np.full(n_br, np.nan), n_lines)
    obj.violation_capacity = 1
    assert obj._inner.violation_capacity == 1
    obj.compute(batch_size=4)
    viol = obj.get_violations()
    assert [len(v) for v in viol] == [1] * n_rows
    assert all(v[0].violation_type == LimitViolationType.HIGH_VOLTAGE for v in viol)
    assert obj.get_violations_truncated().all()
    np.testing.assert_array_equal(obj.get_violation_counts()["high_voltage"], n_bus)

    obj.violation_tol = 1e-30           # below rounding: nothing can converge to it
    assert obj._inner.violation_tol == pytest.approx(1e-30)
    obj.compute(batch_size=4)
    viol = obj.get_violations()
    assert all(len(v) == 1 and v[0].violation_type == LimitViolationType.DIVERGENCE for v in viol)
    with pytest.raises(ValueError, match="violation_capacity"):
        obj.violation_capacity = 0


# ------------------------------------------------------- benchmark_cudss_batch_raw
_BAD_CSR = {
    # (indptr, indices) of a 3x3 matrix with 5 entries, each broken one way
    "index_past_dim":  ([0, 2, 3, 5], [0, 1, 1, 0, 3]),
    "negative_index":  ([0, 2, 3, 5], [0, 1, 1, -1, 2]),
    "indptr_decreasing": ([0, 3, 2, 5], [0, 1, 2, 1, 2]),
    "indptr_not_from_zero": ([1, 2, 3, 5], [0, 1, 1, 0, 2]),
}


@pytest.mark.parametrize("which", sorted(_BAD_CSR))
def test_benchmark_raw_refuses_a_malformed_csr(which):
    """Run in a child process: before the check, the bad pattern reads past host arrays
    and hands cuDSS a broken matrix, which can take the CUDA context down with it."""
    indptr, indices = _BAD_CSR[which]
    code = textwrap.dedent(f"""
        from gpusim2grid import _gpusim2grid as c
        try:
            c.benchmark_cudss_batch_raw(3, {indptr}, {indices}, [4., 1., 4., 1., 4.], 2, 1)
        except RuntimeError as e:
            print("REFUSED:", e)
        else:
            print("ACCEPTED")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "REFUSED:" in out.stdout, (out.returncode, out.stdout, out.stderr[-2000:])
    assert "benchmark_cudss_batch_raw" in out.stdout


# --------------------------------------------------------------- repeated ids
@needs_bridge
def test_branch_named_twice_is_tripped_once(case, solver_atol):
    """add_contingencies_by_branch_id([[3, 3]]) is the trip of branch 3, as lightsim2grid
    (whose contingency is a set) has it."""
    from gpusim2grid import ContingencyAnalysisGPU
    grid = case["grid"]
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, **GPU_TOL)
    ca.add_contingencies_by_branch_id([[3], [3, 3], [5, 3, 5]])
    ca.compute(batch_size=4)
    V = _V(ca, 3, case["n_bus"])
    assert np.all(ca.last_residuals() < solver_atol)
    np.testing.assert_allclose(V[1], V[0], rtol=0., atol=solver_atol)
    ref = ContingencyAnalysisGPU(grid, nb_iter=10, **GPU_TOL)
    ref.add_contingencies_by_branch_id([[3, 5]])
    ref.compute(batch_size=1)
    np.testing.assert_allclose(V[2], _V(ref, 1, case["n_bus"])[0], rtol=0., atol=solver_atol)


@needs_bridge
def test_engine_normalises_overlapping_lists(case, solver_atol):
    """The engine's build_contingencies_full (and the session under it) take the lists
    as given: a branch also named at one end, or at both ends, is opened at both; a
    shunt, load or generator named twice is taken out once. Same rows as the facade's
    normalised ones."""
    from gpusim2grid import ContingencyAnalysisGPU
    grid = case["grid"]
    n_bus = case["n_bus"]
    raw = ContingencyAnalysisGPU(grid, nb_iter=10, **GPU_TOL)
    #       branches    shunts  loads   gens    side1   side2
    rows = [([3],       [],     [],     [],     [3],    []),
            ([],        [],     [],     [],     [4],    [4]),
            ([],        [0, 0], [],     [],     [],     []),
            ([],        [],     [2, 2], [],     [],     []),
            ([],        [],     [],     [1, 1], [],     [])]
    raw._inner.build_contingencies_full(*[[list(r[k]) for r in rows] for k in range(6)])
    raw.compute(batch_size=8)
    clean = ContingencyAnalysisGPU(grid, nb_iter=10, **GPU_TOL)
    clean.add_contingencies([((3,), (), (), ()), ((4,), (), (), ()), ((), (0,), (), ()),
                             ((), (), (2,), ()), ((), (), (), (1,))])
    clean.compute(batch_size=8)
    assert np.all(clean.last_residuals() < solver_atol)
    np.testing.assert_allclose(_V(raw, 5, n_bus), _V(clean, 5, n_bus), rtol=0., atol=solver_atol)
    # and the branch opened at both ends is reported tripped once (no flow, counted once)
    raw.compute_flows()
    assert raw.or_amps.to_numpy().reshape(5, -1)[0, 3] == 0.


# ------------------------------------------- ScenarioSweepGPU shunt / load masks
def _shunt_mask(grid, n):
    m = np.zeros((n, len(grid.get_shunts())), dtype=bool)
    m[0, 0] = True
    return m


@needs_bridge
def test_sweep_shunt_mask_follows_a_new_row_count(case, solver_atol):
    """set_contingency_shunts on 2 rows, then on 4: the branch trips it keeps are the
    implicit "none", resized with the rows (the user never called set_topology)."""
    from gpusim2grid import ScenarioSweepGPU
    grid = case["grid"]
    sw = ScenarioSweepGPU(grid, nb_iter=10, **GPU_TOL)
    shunt_off_ref = grid.copy()
    shunt_off_ref.deactivate_shunt(0)
    shunt_off_ref.tell_solver_need_reset()
    V_shunt = shunt_off_ref.ac_pf(case["v_ref"].copy(), MAX_IT, TOL)
    assert V_shunt.shape[0] > 0
    for n in (2, 4):
        sw.set_injections_from_elements(*_elements(grid, n))
        sw.set_contingency_shunts(_shunt_mask(grid, n))
        sw.compute(batch_size=4)
        V = _V(sw.solver, n, case["n_bus"])
        np.testing.assert_allclose(V[0], V_shunt[case["buses"]], rtol=0., atol=10 * solver_atol)
        np.testing.assert_allclose(V[1:], np.repeat(case["v_ref"][case["buses"]][None, :], n - 1, axis=0),
                                   rtol=0., atol=10 * solver_atol)


@needs_bridge
def test_sweep_shunt_mask_keeps_refusing_a_stale_explicit_topology(case):
    """An explicit set_topology on 2 rows is not silently replaced by "no trip" when the
    rows become 4: the shunt mask (or the run) asks for set_topology again."""
    from gpusim2grid import ScenarioSweepGPU
    grid = case["grid"]
    sw = ScenarioSweepGPU(grid, nb_iter=10, **GPU_TOL)
    sw.set_injections_from_elements(*_elements(grid, 2))
    sw.set_topology([[3], []])
    sw.set_injections_from_elements(*_elements(grid, 4))
    with pytest.raises(RuntimeError, match="set_topology"):
        sw.set_contingency_shunts(_shunt_mask(grid, 4))
        sw.compute(batch_size=4)


@needs_bridge
def test_batch_power_flow_shunt_status_then_another_row_count(case, solver_atol):
    """A forward with shunt_status, then one with another batch size and no status: the
    second forward runs with every shunt connected."""
    torch = pytest.importorskip("torch")
    from gpusim2grid.differentiable import BatchPowerFlow
    grid = case["grid"]
    pf = BatchPowerFlow.from_lsgrid(grid, nb_iter=10)
    load_p0 = np.asarray(grid.get_loads_res_full()[0], dtype=np.float64)
    status = np.ones((3, len(grid.get_shunts())), dtype=bool)
    status[0, 0] = False
    pf(load_p=torch.tensor(_rep(load_p0, 3), device="cuda"), shunt_status=torch.as_tensor(status, device="cuda"))
    V = pf(load_p=torch.tensor(_rep(load_p0, 5), device="cuda"))
    np.testing.assert_allclose(V.cpu().numpy(), np.repeat(case["v_ref"][case["buses"]][None, :], 5, axis=0),
                               rtol=0., atol=10 * solver_atol)


@needs_bridge
def test_load_mask_with_wrong_columns_is_refused_by_the_session(case):
    from gpusim2grid import ScenarioSweepGPU
    grid = case["grid"]
    sw = ScenarioSweepGPU(grid, nb_iter=10, **GPU_TOL)
    with pytest.raises(RuntimeError, match="load"):
        sw._inner._s.set_contingency_loads(np.zeros((2, len(grid.get_loads()) + 1), dtype=bool))


@needs_bridge
def test_stale_load_mask_is_refused_not_dropped(case):
    """set_contingency_loads on 2 rows, then injections on 4: compute refuses (as for
    set_contingency_gens) instead of solving with every load connected."""
    from gpusim2grid import ScenarioSweepGPU
    grid = case["grid"]
    sw = ScenarioSweepGPU(grid, nb_iter=10, **GPU_TOL)
    mask = np.zeros((2, len(grid.get_loads())), dtype=bool)
    mask[0, 3] = True
    sw.set_contingency_loads(mask)
    sw.set_injections_from_elements(*_elements(grid, 4))
    with pytest.raises(RuntimeError, match="set_contingency_loads"):
        sw.compute(batch_size=4)

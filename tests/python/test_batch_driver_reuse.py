# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Driver persistence of the contingency-analysis and injection-sweep
sessions, and their pinned result downloads.

``compute()`` keeps the batch driver (allocations, cuDSS ANALYSIS,
factorization state) while nothing that shapes it changes: a repeated call
pays no new ANALYSIS, and every reused run must give what a freshly built
session on the same inputs gives. ``V_results`` / residuals / flows come
back through a pinned, chunked staging buffer and must equal the device
buffers exactly.
"""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu

torch = pytest.importorskip("torch")


def _elements(grid):
    load_p = np.asarray([l.target_p_mw for l in grid.get_loads()])
    load_q = np.asarray([l.target_q_mvar for l in grid.get_loads()])
    gen_p = np.asarray([g.target_p_mw for g in grid.get_generators()])
    return load_p, load_q, gen_p


def _injections(grid, factors):
    load_p, load_q, gen_p = _elements(grid)
    f = np.asarray(factors, dtype=float)[:, None]
    return load_p[None] * f, load_q[None] * f, gen_p[None] * f


def _session(facade):
    return facade.solver._s


def _device_V(facade):
    return torch.from_dlpack(_session(facade).v_results_dlpack()).cpu().numpy().ravel()


def _assert_same(a, b, atol):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    m = np.isfinite(a)
    np.testing.assert_allclose(a[m], b[m], rtol=0, atol=atol)


# ---------------------------------------------------------------------------
# Contingency analysis
# ---------------------------------------------------------------------------

def _ca(grid, branches, **kw):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, **kw)
    ca.add_contingencies_by_branch_id([[b] for b in branches])
    return ca


def test_ca_repeated_compute_reuses_the_driver(ieee14_base_case, solver_atol):
    grid = ieee14_base_case["grid"]
    ca = _ca(grid, range(10))
    ca.compute()
    V1 = ca.V_results.to_numpy()
    assert _session(ca).driver_build_counter == 1
    assert ca.timings.t_analysis_ms > 0

    ca.compute()
    V2 = ca.V_results.to_numpy()
    assert _session(ca).driver_build_counter == 1      # no new driver
    assert ca.timings.t_analysis_ms == 0.0              # no new ANALYSIS
    assert ca.timings.t_alloc_ms >= 0.0
    _assert_same(V2, V1, 1e-3 * solver_atol)
    _assert_same(V2, _ca_V(grid, range(10)), 1e-3 * solver_atol)   # = a fresh session


def _ca_V(grid, branches, **kw):
    ca = _ca(grid, branches, **kw)
    ca.compute()
    return ca.V_results.to_numpy()


def test_ca_rebuilds_when_shape_changes(ieee14_base_case, solver_atol):
    grid = ieee14_base_case["grid"]
    ca = _ca(grid, range(10))
    ca.compute(batch_size=512)
    ca.compute(batch_size=4)                            # new chunking
    assert _session(ca).driver_build_counter == 2
    ca.add_contingencies_by_branch_id([[b] for b in range(3, 15)])   # new list
    ca.compute(batch_size=4)
    assert _session(ca).driver_build_counter == 3
    assert ca.timings.t_analysis_ms > 0
    _assert_same(ca.V_results.to_numpy(), _ca_V(grid, range(3, 15)), 1e-3 * solver_atol)

    # same-size list, other branches: still a new source, so a new driver
    ca.add_contingencies_by_branch_id([[b] for b in range(4, 16)])
    ca.compute(batch_size=4)
    assert _session(ca).driver_build_counter == 4
    _assert_same(ca.V_results.to_numpy(), _ca_V(grid, range(4, 16)), 1e-3 * solver_atol)


def test_ca_nb_iter_change_keeps_the_driver(ieee14_base_case, solver_atol):
    grid = ieee14_base_case["grid"]
    ca = _ca(grid, range(10), nb_iter=2)
    ca.compute()
    ca.solver.nb_iter = 5
    ca.compute()
    assert _session(ca).driver_build_counter == 1
    _assert_same(ca.V_results.to_numpy(), _ca_V(grid, range(10), nb_iter=5), 1e-3 * solver_atol)


def test_ca_hot_run_keeps_disconnected_rows(ieee14_base_case):
    """The per-row islanding flags describe the live source: a hot run must
    report the same disconnected rows (it skips the connectivity pass)."""
    grid = ieee14_base_case["grid"]
    n_br = len(grid.get_lines()) + len(grid.get_trafos())
    ca = _ca(grid, range(n_br))
    ca.compute()
    n_disc, res1 = ca.timings.n_disconnected, ca.last_residuals()
    assert n_disc > 0, "ieee14 should have at least one islanding single outage"
    ca.compute()
    assert _session(ca).driver_build_counter == 1
    assert ca.timings.n_disconnected == n_disc
    np.testing.assert_array_equal(np.isnan(ca.last_residuals()), np.isnan(res1))


def test_ca_downloads_are_exact(ieee14_base_case):
    grid = ieee14_base_case["grid"]
    ca = _ca(grid, range(10))
    ca.compute()
    np.testing.assert_array_equal(ca.V_results.to_numpy(), _device_V(ca))
    ca.compute_flows()
    ref = _ca(grid, range(10))
    ref.compute()
    ref.compute_flows()
    # two sessions: equal to rounding only (batched GPU reductions)
    np.testing.assert_allclose(ca.or_amps.to_numpy(), ref.or_amps.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(ca.ex_amps.to_numpy(), ref.ex_amps.to_numpy(), rtol=1e-9)


# ---------------------------------------------------------------------------
# Injection sweep
# ---------------------------------------------------------------------------

def _is(grid, factors, **kw):
    from gpusim2grid import InjectionSweepGPU
    sw = InjectionSweepGPU(grid, **kw)
    sw.set_injections_from_elements(*_injections(grid, factors))
    return sw


def _is_V(grid, factors, gen_v=None, **kw):
    sw = _is(grid, factors, **kw)
    if gen_v is not None:
        sw.set_gen_v(gen_v)
    sw.compute()
    return sw.V_results.to_numpy()


F1 = np.linspace(0.95, 1.05, 7)
F2 = np.linspace(1.04, 0.96, 7)


def test_is_repeated_compute_reuses_the_driver(ieee14_base_case, solver_atol):
    grid = ieee14_base_case["grid"]
    sw = _is(grid, F1)
    sw.compute()
    V1 = sw.V_results.to_numpy()
    sw.compute()
    assert _session(sw).driver_build_counter == 1
    assert sw.timings.t_analysis_ms == 0.0
    _assert_same(sw.V_results.to_numpy(), V1, 1e-3 * solver_atol)


def test_is_new_injections_same_count_keep_the_driver(ieee14_base_case, solver_atol):
    """The common loop: new injections every call, same scenario count."""
    grid = ieee14_base_case["grid"]
    sw = _is(grid, F1)
    sw.compute()
    sw.set_injections_from_elements(*_injections(grid, F2))
    sw.compute()
    assert _session(sw).driver_build_counter == 1
    assert sw.timings.t_analysis_ms == 0.0
    V = sw.V_results.to_numpy()
    _assert_same(V, _is_V(grid, F2), 1e-3 * solver_atol)
    assert np.abs(V - _is_V(grid, F1)).max() > 1e3 * solver_atol   # really moved


def test_is_new_gen_v_keeps_the_driver(ieee14_base_case, solver_atol):
    grid = ieee14_base_case["grid"]
    n_gen = len(grid.get_generators())
    gen_v = np.full((len(F1), n_gen), np.nan)
    gen_v[:, 0] = np.linspace(1.0, 1.06, len(F1))
    sw = _is(grid, F1)
    sw.compute()
    sw.set_gen_v(gen_v)
    sw.compute()
    assert _session(sw).driver_build_counter == 1
    _assert_same(sw.V_results.to_numpy(), _is_V(grid, F1, gen_v=gen_v), 1e-3 * solver_atol)


def test_is_row_count_change_rebuilds(ieee14_base_case, solver_atol):
    grid = ieee14_base_case["grid"]
    sw = _is(grid, F1)
    sw.compute()
    F3 = np.linspace(0.98, 1.02, 11)
    sw.set_injections_from_elements(*_injections(grid, F3))
    sw.compute()
    assert _session(sw).driver_build_counter == 2
    _assert_same(sw.V_results.to_numpy(), _is_V(grid, F3), 1e-3 * solver_atol)


def test_is_downloads_are_exact(ieee14_base_case):
    grid = ieee14_base_case["grid"]
    sw = _is(grid, F1)
    sw.compute()
    np.testing.assert_array_equal(sw.V_results.to_numpy(), _device_V(sw))


def test_download_spans_several_staging_chunks():
    """> 2 MB of voltages: several chunks through both staging halves."""
    import warnings
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    from gpusim2grid import InjectionSweepGPU
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(pn.case1354pegase())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n = grid.get_bus_vn_kv().shape[0]
    assert grid.ac_pf(grid.dc_pf(np.ones(n, dtype=complex), 1, 1e-6), 30, 1e-8).shape[0] == n
    sw = InjectionSweepGPU(grid)
    f = np.linspace(0.99, 1.01, 300)
    sw.set_injections_from_elements(*_injections(grid, f))
    sw.compute()
    V = sw.V_results.to_numpy()
    assert V.nbytes > 3 * (2 << 20)
    np.testing.assert_array_equal(V, _device_V(sw))

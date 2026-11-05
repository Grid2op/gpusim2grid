# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The batched LU of our own (``GPUSIM2GRID_OWN_LU=1``, ``src/_cpp/own_lu/``,
docs/dev_notes/batched_lu_and_constrained_ordering.md).

It replaces cuDSS's numeric factorization and solve in every batch linear solve
(forward, continuous tail, adjoint), with cuDSS's ANALYSIS kept for its order:
static diagonal pivots, tiny ones clamped and counted, iterative refinement on
``ir_n_steps``. Another factorization changes the rounding only, so every batch
must end as with cuDSS: same statuses and iterations, same V to the solver
tolerance; with lightsim2grid's constrained order no pivot is clamped.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
import gpusim2grid._gpusim2grid as _cpp

pytestmark = requires_gpu

PREC = "fp32" if _cpp.is_fp32 else "fp64"
CONVERGED = 0


def _grid(case):
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(getattr(pn, case)())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n = grid.get_bus_vn_kv().shape[0]
    assert grid.ac_pf(np.ones(n, dtype=complex), 30, 1e-10).shape[0] > 0
    return grid


@pytest.fixture(scope="module")
def case118():
    return _grid("case118")


@pytest.fixture
def own_lu(monkeypatch):
    """GPUSIM2GRID_OWN_LU is read when a batch solver is built: set it before
    building a session, unset it to build the cuDSS reference."""
    def use(on):
        if on:
            monkeypatch.setenv("GPUSIM2GRID_OWN_LU", "1")
        else:
            monkeypatch.delenv("GPUSIM2GRID_OWN_LU", raising=False)
    yield use
    monkeypatch.delenv("GPUSIM2GRID_OWN_LU", raising=False)


def _outcome(obj, n_rows):
    return (np.asarray(obj.get_row_status()), np.asarray(obj.get_row_iterations()),
            obj.V_results.to_numpy().reshape(n_rows, -1))


def _same_rows(run, ref, solver_atol):
    st0, it0, V0 = ref
    st, it, V = run
    np.testing.assert_array_equal(st, st0)
    np.testing.assert_array_equal(it, it0)
    conv = st0 == CONVERGED
    np.testing.assert_array_equal(np.isnan(V[conv]), np.isnan(V0[conv]))
    m = conv[:, None] & ~np.isnan(V0)
    np.testing.assert_allclose(V[m], V0[m], rtol=0, atol=solver_atol)


# =============================================================================
# The raw solver
# =============================================================================
def test_raw_benchmark_solves_the_jacobian(case118, own_lu, residual_atol):
    """benchmark_cudss_batch_raw goes through CudssBatchSolver: the own LU
    factorizes, solves and refines the base-case Jacobian of case118."""
    from gpusim2grid import AcPfGPU
    pf = AcPfGPU(case118, precision=PREC)
    indptr, indices, data = (np.asarray(a) for a in pf._s.get_J())
    n = len(indptr) - 1
    res = {}
    for on in (False, True):
        own_lu(on)
        for ir in (0, 1):
            r = _cpp.benchmark_cudss_batch_raw(n, indptr.tolist(), indices.tolist(),
                                               data.astype(float).tolist(), 20, n_refactorize=2, ir_n_steps=ir)
            assert r.n_nonfinite_slots == 0
            res[(on, ir)] = r
    assert res[(True, 0)].max_rel_residual < residual_atol
    assert res[(True, 1)].max_rel_residual <= max(res[(True, 0)].max_rel_residual, 1e-13)
    assert res[(True, 0)].lu_nnz > 0


# =============================================================================
# The batch facades, own LU against cuDSS
# =============================================================================
@pytest.mark.parametrize("scheduling", ["chunked", "continuous"])
def test_contingency_analysis(case118, own_lu, scheduling, solver_atol):
    """N-1 of case118 with handle_disconnected_grid (masked islands); the
    continuous run with a tail of groups of 3 (bind_slice on its own handle)."""
    from gpusim2grid import ContingencyAnalysisGPU
    nbr = len(case118.get_lines()) + len(case118.get_trafos())
    runs = {}
    for on in (False, True):
        own_lu(on)
        ca = ContingencyAnalysisGPU(case118, precision=PREC, nb_iter=10, scheduling=scheduling,
                                    handle_disconnected_grid=True)
        if scheduling == "continuous":
            ca.tail_max_groups, ca.tail_group_size = 10 ** 6, 3
        ca.add_contingencies_by_branch_id([[i] for i in range(nbr)])
        ca.compute(batch_size=64)
        runs[on] = _outcome(ca, nbr)
        if scheduling == "continuous":
            assert ca.timings.n_tail_rounds > 0
    assert np.any(np.isnan(runs[False][2]))
    _same_rows(runs[True], runs[False], solver_atol)


def test_injection_sweep(case118, own_lu, solver_atol):
    from gpusim2grid import InjectionSweepGPU
    n_rows = 40
    n_bus = case118.get_Ybus_solver().shape[0]
    sn = case118.get_sn_mva()
    Sbus = case118.get_Sbus_solver()
    rng = np.random.default_rng(0)
    scale = rng.uniform(0.6, 1.4, size=(n_rows, 1)) * rng.uniform(0.9, 1.1, size=(n_rows, n_bus))
    runs = {}
    for on in (False, True):
        own_lu(on)
        sw = InjectionSweepGPU(case118, precision=PREC, nb_iter=10)
        sw.set_injections(Sbus.real * sn * scale, Sbus.imag * sn * scale, sn)
        sw.compute(batch_size=16)
        runs[on] = _outcome(sw, n_rows)
    _same_rows(runs[True], runs[False], solver_atol)


# =============================================================================
# The outer loops, with lightsim2grid's constrained order: nothing clamped
# =============================================================================
@pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_loops", False),
                    reason="gpusim2grid built against a lightsim2grid without NROuter_*")
@pytest.mark.skipif(bool(_cpp.is_fp32), reason="parity with lightsim2grid's double-precision NR")
def test_outer_loops_constrained_order(own_lu, monkeypatch, solver_atol):
    """case14 with a remote VoltageControl group (rows the loops switch by
    value): with cuDSS's own order the own LU clamps the pivots those rows zero
    and still ends as cuDSS; with lightsim2grid's constrained order it clamps
    none. Same statuses, outer iterations and V everywhere."""
    from test_ls2g_constrained_ordering import _case14_remote_group, _outer_ca
    monkeypatch.setenv("GPUSIM2GRID_COUNT_PIVOTS", "1")
    grid = _case14_remote_group()
    own_lu(False)
    ref, n = _outer_ca(grid, "default")
    st0 = np.asarray(ref.get_row_status())
    assert (st0 == CONVERGED).any()
    V0 = ref.V_results.to_numpy().reshape(n, -1)
    outer0 = [r.nb_outer_iterations for r in ref.get_outer_results()]
    own_lu(True)
    for alg in ("default", "ls2g_constrained"):
        ca, _ = _outer_ca(grid, alg)
        np.testing.assert_array_equal(np.asarray(ca.get_row_status()), st0, err_msg=alg)
        assert [r.nb_outer_iterations for r in ca.get_outer_results()] == outer0, alg
        ok = st0 == CONVERGED
        np.testing.assert_allclose(ca.V_results.to_numpy().reshape(n, -1)[ok], V0[ok],
                                   atol=solver_atol, rtol=0, err_msg=alg)
        t = ca.timings
        assert t.cudss_npivots_reads > 0, alg
        if alg == "ls2g_constrained":
            assert t.cudss_user_perm_honoured == 1
            assert t.cudss_npivots_total == 0


# =============================================================================
# The adjoint (a second batch solver, over J^T)
# =============================================================================
@pytest.mark.skipif(bool(_cpp.is_fp32), reason="finite-difference gradient checks need the FP64 build")
def test_batch_power_flow_gradient(ieee14_base_case, own_lu):
    """BatchPowerFlow's backward solves J^T on the own LU: central differences
    on every input kind, a row with a tripped line included."""
    torch = pytest.importorskip("torch")
    from test_batch_power_flow import _pf, _base_inputs, _all_connected, _fd_check, RDT
    own_lu(True)
    pf = _pf(ieee14_base_case["grid"])
    n = 3
    load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1, 0.9])
    line_status, _ = _all_connected(pf, n)
    line_status[1, 3] = False
    gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
    gen_v[:, 1] = 1.04
    coords = [("load_p", (0, 2)), ("load_p", (1, 5)), ("load_q", (2, 4)),
              ("gen_p", (1, 1)), ("gen_v", (0, 1)), ("gen_v", (2, 1))]
    _fd_check(pf, load_p, load_q, gen_p, gen_v, line_status, coords)

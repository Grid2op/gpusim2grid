# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The batched LU of our own (``linear_solver='own_lu'``, ``src/_cpp/own_lu/``,
docs/dev_notes/batched_lu_and_constrained_ordering.md).

It replaces cuDSS in every batch linear solve (forward, continuous tail,
adjoint), with no cuDSS call at all: static diagonal pivots along an order
chosen on the host from ``reordering_alg`` (lightsim2grid's constrained order
by default when the grid declares constraints, AMD otherwise), tiny pivots
clamped and counted, iterative refinement on ``ir_n_steps``. Another
factorization changes the rounding only, so every batch must end as with
cuDSS: same statuses and iterations, same V to the solver tolerance; with
lightsim2grid's constrained order no pivot is clamped.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
import gpusim2grid._gpusim2grid as _cpp

pytestmark = requires_gpu

PREC = "fp32" if _cpp.is_fp32 else "fp64"
CONVERGED = 0
SOLVERS = ("cudss", "own_lu")
needs_outer = [
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_loops", False),
                       reason="gpusim2grid built against a lightsim2grid without NROuter_*"),
    pytest.mark.skipif(bool(_cpp.is_fp32), reason="parity with lightsim2grid's double-precision NR"),
]


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


def _n1(grid, linear_solver, scheduling="chunked", **kw):
    from gpusim2grid import ContingencyAnalysisGPU
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=10, scheduling=scheduling,
                                handle_disconnected_grid=True, linear_solver=linear_solver, **kw)
    ca.add_contingencies_by_branch_id([[i] for i in range(nbr)])
    return ca, nbr


# =============================================================================
# The raw solver
# =============================================================================
def test_raw_benchmark_solves_the_jacobian(case118, residual_atol):
    """benchmark_cudss_batch_raw(linear_solver=OwnLu): the own LU factorizes,
    solves and refines case118's Jacobian, on AMD (no grid, no constraints)
    and on the identity order."""
    from gpusim2grid import AcPfGPU
    pf = AcPfGPU(case118, precision=PREC)
    indptr, indices, data = (np.asarray(a) for a in pf._s.get_J())
    n = len(indptr) - 1
    args = (n, indptr.tolist(), indices.tolist(), data.astype(float).tolist(), 20)
    ref = _cpp.benchmark_cudss_batch_raw(*args, n_refactorize=2)
    for alg in (_cpp.ReorderingAlg.Default, _cpp.ReorderingAlg.Amd, _cpp.ReorderingAlg.NoReordering):
        r0, r1 = (_cpp.benchmark_cudss_batch_raw(*args, n_refactorize=2, ir_n_steps=ir, reordering_alg=alg,
                                                 linear_solver=_cpp.LinearSolver.OwnLu) for ir in (0, 1))
        for r in (r0, r1):
            assert r.n_nonfinite_slots == 0
            assert r.context_init_ms == 0.          # no cuDSS handle
        assert r0.max_rel_residual < residual_atol
        assert r1.max_rel_residual <= max(r0.max_rel_residual, 1e-13)
    amd = _cpp.benchmark_cudss_batch_raw(*args, n_refactorize=1, reordering_alg=_cpp.ReorderingAlg.Amd,
                                         linear_solver=_cpp.LinearSolver.OwnLu)
    ident = _cpp.benchmark_cudss_batch_raw(*args, n_refactorize=1, reordering_alg=_cpp.ReorderingAlg.NoReordering,
                                           linear_solver=_cpp.LinearSolver.OwnLu)
    assert 0 < amd.lu_nnz < ident.lu_nnz      # AMD reduces the fill
    assert ref.lu_nnz > 0


# =============================================================================
# The batch facades, own LU against cuDSS
# =============================================================================
@pytest.mark.parametrize("scheduling", ["chunked", "continuous"])
def test_contingency_analysis(case118, scheduling, solver_atol):
    """N-1 of case118 with handle_disconnected_grid (masked islands); the
    continuous run with a tail of groups of 3 (bind_slice on its own handle)."""
    runs = {}
    for ls in SOLVERS:
        ca, nbr = _n1(case118, ls, scheduling)
        if scheduling == "continuous":
            ca.tail_max_groups, ca.tail_group_size = 10 ** 6, 3
        ca.compute(batch_size=64)
        assert ca.linear_solver == ls
        runs[ls] = _outcome(ca, nbr)
        if scheduling == "continuous":
            assert ca.timings.n_tail_rounds > 0
    assert np.any(np.isnan(runs["cudss"][2]))
    _same_rows(runs["own_lu"], runs["cudss"], solver_atol)


def test_injection_sweep(case118, solver_atol):
    from gpusim2grid import InjectionSweepGPU
    n_rows = 40
    n_bus = case118.get_Ybus_solver().shape[0]
    sn = case118.get_sn_mva()
    Sbus = case118.get_Sbus_solver()
    rng = np.random.default_rng(0)
    scale = rng.uniform(0.6, 1.4, size=(n_rows, 1)) * rng.uniform(0.9, 1.1, size=(n_rows, n_bus))
    runs = {}
    for ls in SOLVERS:
        sw = InjectionSweepGPU(case118, precision=PREC, nb_iter=10, linear_solver=ls)
        sw.set_injections(Sbus.real * sn * scale, Sbus.imag * sn * scale, sn)
        sw.compute(batch_size=16)
        runs[ls] = _outcome(sw, n_rows)
    _same_rows(runs["own_lu"], runs["cudss"], solver_atol)


def test_array_mode_takes_amd(case118, solver_atol):
    """A tuple grid has no lightsim2grid constraints: 'default' is AMD."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid._ls2g_utils import extract_grid_arrays, extract_branch_data
    d = extract_grid_arrays(case118)
    arrays = (d["Ybus"], d["v_converged"], d["Sbus"], d["slack"], d["slack_weights"], d["pv"], d["pq"])
    branch_args, _, _ = extract_branch_data(case118)
    nbr = len(branch_args[0])
    runs = {}
    for ls in SOLVERS:
        ca = ContingencyAnalysisGPU(arrays, precision=PREC, nb_iter=10, linear_solver=ls)
        ca.set_branch_data(*branch_args)
        ca.add_contingencies_by_branch_id([[i] for i in range(nbr)])
        ca.compute(batch_size=64)
        runs[ls] = _outcome(ca, nbr)
    _same_rows(runs["own_lu"], runs["cudss"], solver_atol)


def test_switching_solver_rebuilds_the_driver(case118, solver_atol):
    """linear_solver is mutable: a change builds a new driver on the next run."""
    ca, nbr = _n1(case118, "cudss")
    ca.compute(batch_size=64)
    ref = _outcome(ca, nbr)
    builds = ca._inner._s.driver_build_counter
    ca.linear_solver = "own_lu"
    ca.compute(batch_size=64)
    assert ca._inner._s.driver_build_counter == builds + 1
    _same_rows(_outcome(ca, nbr), ref, solver_atol)


def test_refusals(case118):
    with pytest.raises(ValueError, match="linear_solver"):
        _n1(case118, "klu")
    for kw, match in (({"reordering_alg": "nested_dissection"}, "reordering_alg"),
                      ({"matching_alg": "max_diag_product"}, "matching_alg")):
        ca, _ = _n1(case118, "own_lu", **kw)
        with pytest.raises(ValueError, match=match):
            ca.compute(batch_size=64)


# =============================================================================
# The outer loops: lightsim2grid's constrained order by default, nothing clamped
# =============================================================================
def _outer_ca(grid, linear_solver, reordering_alg):
    from gpusim2grid import ContingencyAnalysisGPU
    from test_ls2g_constrained_ordering import MAX_IT, TOL_MVA
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=False,
                                tol=TOL_MVA / grid.get_sn_mva(), reordering_alg=reordering_alg,
                                linear_solver=linear_solver)
    lines = [i for i, line in enumerate(grid.get_lines()) if line.connected_global]
    ca.add_contingencies_by_branch_id([[c] for c in lines])
    ca.compute(batch_size=4)
    return ca, len(lines)


@pytest.mark.parametrize("dummy", [0], ids=[""])
@needs_outer[0]
@needs_outer[1]
def test_outer_loops_orders(dummy, monkeypatch, solver_atol):
    """case14 with a remote VoltageControl group (rows the loops switch by
    value). The own LU with 'default' takes lightsim2grid's constrained order
    (the fill of 'ls2g_constrained', nothing clamped); with 'amd' it clamps the
    pivots those rows zero and still ends as cuDSS. Same statuses, outer
    iterations and V everywhere."""
    from test_ls2g_constrained_ordering import _case14_remote_group
    monkeypatch.setenv("GPUSIM2GRID_COUNT_PIVOTS", "1")
    grid = _case14_remote_group()
    ref, n = _outer_ca(grid, "cudss", "default")
    st0 = np.asarray(ref.get_row_status())
    assert (st0 == CONVERGED).any()
    V0 = ref.V_results.to_numpy().reshape(n, -1)
    outer0 = [r.nb_outer_iterations for r in ref.get_outer_results()]
    t = {}
    for alg in ("default", "ls2g_constrained", "amd"):
        ca, _ = _outer_ca(grid, "own_lu", alg)
        np.testing.assert_array_equal(np.asarray(ca.get_row_status()), st0, err_msg=alg)
        assert [r.nb_outer_iterations for r in ca.get_outer_results()] == outer0, alg
        ok = st0 == CONVERGED
        np.testing.assert_allclose(ca.V_results.to_numpy().reshape(n, -1)[ok], V0[ok],
                                   atol=solver_atol, rtol=0, err_msg=alg)
        t[alg] = ca.timings
        assert t[alg].cudss_npivots_reads > 0, alg
    assert t["default"].cudss_lu_nnz == t["ls2g_constrained"].cudss_lu_nnz
    assert t["default"].cudss_npivots_total == 0
    assert t["ls2g_constrained"].cudss_npivots_total == 0
    assert t["amd"].cudss_npivots_total > 0


# =============================================================================
# The adjoint (a second batch solver, over J^T)
# =============================================================================
@pytest.mark.skipif(bool(_cpp.is_fp32), reason="finite-difference gradient checks need the FP64 build")
def test_batch_power_flow_gradient(ieee14_base_case):
    """BatchPowerFlow's backward solves J^T on the own LU: central differences
    on every input kind, a row with a tripped line included."""
    torch = pytest.importorskip("torch")
    from test_batch_power_flow import _pf, _base_inputs, _all_connected, _fd_check, RDT
    pf = _pf(ieee14_base_case["grid"], linear_solver="own_lu")
    n = 3
    load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1, 0.9])
    line_status, _ = _all_connected(pf, n)
    line_status[1, 3] = False
    gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
    gen_v[:, 1] = 1.04
    coords = [("load_p", (0, 2)), ("load_p", (1, 5)), ("load_q", (2, 4)),
              ("gen_p", (1, 1)), ("gen_v", (0, 1)), ("gen_v", (2, 1))]
    _fd_check(pf, load_p, load_q, gen_p, gen_v, line_status, coords)

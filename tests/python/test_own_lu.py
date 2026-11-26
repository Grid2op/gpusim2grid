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

from conftest import parity_tol, requires_gpu
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


def _same_rows(run, ref, atol):
    st0, it0, V0 = ref
    st, it, V = run
    np.testing.assert_array_equal(st, st0)
    np.testing.assert_array_equal(it, it0)
    conv = st0 == CONVERGED
    assert conv.any()      # else the V comparison is vacuous
    np.testing.assert_array_equal(np.isnan(V[conv]), np.isnan(V0[conv]))
    m = conv[:, None] & ~np.isnan(V0)
    np.testing.assert_allclose(V[m], V0[m], rtol=0, atol=atol)


def _assert_solver(timings, linear_solver):
    """The batch really ran on that solver: the own LU creates no cuDSS object,
    so cuDSS's memory estimates stay unreported (-1)."""
    if linear_solver == "own_lu":
        assert timings.cudss_mem_device_permanent_bytes == -1
        assert timings.cudss_lu_nnz > 0
    else:
        assert timings.cudss_mem_device_permanent_bytes > 0


def _dumped_solvers(path):
    """GPUSIM2GRID_DUMP_J: one meta file per batch solver initialized; only a
    cuDSS one writes the order cuDSS chose (perm_cudss)."""
    ids = sorted(m.name.split("_")[0] for m in path.glob("h*_meta.txt"))
    return {h: ("cudss" if (path / f"{h}_perm_cudss.i32").exists() else "own_lu") for h in ids}


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
    """benchmark_cudss_batch_raw(linear_solver=OwnLu): the own LU factorizes
    and solves case118's Jacobian, with and without a refinement step, on
    'default' (= AMD: no grid, no constraints), AMD and the identity order."""
    from gpusim2grid import AcPfGPU
    pf = AcPfGPU(case118, precision=PREC)
    indptr, indices, data = (np.asarray(a) for a in pf._s.get_J())
    n = len(indptr) - 1
    args = (n, indptr.tolist(), indices.tolist(), data.astype(float).tolist(), 20)
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
    default = _cpp.benchmark_cudss_batch_raw(*args, n_refactorize=1, linear_solver=_cpp.LinearSolver.OwnLu)
    assert 0 < amd.lu_nnz < ident.lu_nnz      # AMD reduces the fill
    assert default.lu_nnz == amd.lu_nnz       # 'default' without constraints is AMD


def test_raw_refinement_recovers_a_clamped_pivot(monkeypatch):
    """ir_n_steps refines the own LU's solution: a saddle matrix whose first
    pivot is 0 in the identity order gets it clamped (the factors are those of
    a perturbed matrix); each refinement step against the true matrix cuts the
    residual by orders of magnitude. On the default clamp, which depends on the
    precision: 1e-12 of max|A| in FP64 would be below float resolution in FP32
    (garbage factors, refinement diverging), so FP32 clamps at 1e-5."""
    monkeypatch.delenv("GPUSIM2GRID_OWN_LU_EPS", raising=False)
    M = (3, [0, 3, 6, 9], [0, 1, 2] * 3, [0., 1., 2., 1., 1., 0., 2., 0., 1.])
    res = [_cpp.benchmark_cudss_batch_raw(*M, 4, n_refactorize=1, ir_n_steps=ir,
                                          reordering_alg=_cpp.ReorderingAlg.NoReordering,
                                          linear_solver=_cpp.LinearSolver.OwnLu)
           for ir in (0, 1, 2)]
    assert all(r.n_nonfinite_slots == 0 for r in res)
    r0, r1, r2 = (r.max_rel_residual for r in res)
    assert r0 > 1e-8                          # the clamp is visible
    assert r1 < 1e-2 * r0 and r2 < 1e-2 * max(r1, 1e-13), (r0, r1, r2)


@pytest.mark.skipif(bool(_cpp.is_fp32), reason="FP32's 1e-5 clamp; the parity bound is an FP64 one")
@pytest.mark.parametrize("scheduling", ["chunked", "continuous"])
@pytest.mark.parametrize("top", ["0", "1"], ids=["own_lu", "own_lu_top"])
def test_refinement_only_where_a_pivot_was_clamped(case118, monkeypatch, top, scheduling):
    """ir_n_steps refines only the matrices whose factorization clamped a pivot (the saddle matrix
    above): the others' factors are those of their own matrix, a step changes their solution by
    rounding only, and it cost a second triangular solve of every matrix (PtFige N-1: 53 ms of
    0.354 s). case118's N-1 clamps nothing, so ir_n_steps=1 is ir_n_steps=0 to the bit -- statuses,
    iterations, V -- on both own LUs; GPUSIM2GRID_OWN_LU_IR=all refines every matrix as before, which
    moves V by rounding only."""
    monkeypatch.setenv("GPUSIM2GRID_OWN_LU_TOP", top)
    monkeypatch.setenv("GPUSIM2GRID_COUNT_PIVOTS", "1")
    monkeypatch.delenv("GPUSIM2GRID_OWN_LU_EPS", raising=False)
    monkeypatch.delenv("GPUSIM2GRID_OWN_LU_IR", raising=False)
    runs = {}
    for name, ir, ir_env in (("ir0", 0, None), ("ir1", 1, None), ("all", 1, "all")):
        if ir_env is not None:
            monkeypatch.setenv("GPUSIM2GRID_OWN_LU_IR", ir_env)
        ca, nbr = _n1(case118, "own_lu", scheduling, ir_n_steps=ir)
        ca.compute(batch_size=64)
        assert ca.timings.cudss_npivots_total == 0          # nothing clamped, nothing to refine
        runs[name] = _outcome(ca, nbr)
    for got, want in zip(runs["ir1"], runs["ir0"]):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(runs["all"][0], runs["ir0"][0])
    np.testing.assert_allclose(runs["all"][2], runs["ir0"][2], rtol=0, atol=parity_tol("state"))


@pytest.mark.parametrize("n_active", [1, 7, 20])
def test_raw_active_prefix_leaves_the_other_matrices_alone(case118, residual_atol, n_active):
    """BatchedLu's active prefix (what the continuous tail runs on): the first
    n_active matrices are refactorized and solved alone; past them, no
    solution entry is written and the factors stay those of the earlier values.
    Before the prefix call the benchmark triples the prefix's values (solved
    right only if refactorized) and doubles the others (solved right only if
    not)."""
    from gpusim2grid import AcPfGPU
    pf = AcPfGPU(case118, precision=PREC)
    indptr, indices, data = (np.asarray(a) for a in pf._s.get_J())
    n = len(indptr) - 1
    r = _cpp.benchmark_cudss_batch_raw(n, indptr.tolist(), indices.tolist(), data.astype(float).tolist(), 20,
                                       n_refactorize=1, linear_solver=_cpp.LinearSolver.OwnLu,
                                       n_active=n_active)
    assert r.n_active == n_active
    assert r.active_max_rel_residual < residual_atol
    assert r.n_inactive_touched == 0
    if n_active < 20:
        assert r.inactive_max_rel_residual < residual_atol
    if n_active == 20:
        assert r.inactive_max_rel_residual == 0.   # nothing past the prefix
    with pytest.raises(RuntimeError, match="n_active"):
        _cpp.benchmark_cudss_batch_raw(n, indptr.tolist(), indices.tolist(), data.astype(float).tolist(), 20,
                                       n_active=3)   # cuDSS: no active prefix


# =============================================================================
# The batch facades, own LU against cuDSS
# =============================================================================
@pytest.mark.parametrize("scheduling", ["chunked", "continuous"])
def test_contingency_analysis(case118, scheduling, solver_atol):
    """N-1 of case118 with handle_disconnected_grid (masked islands); the
    continuous run with a tail of groups of 3 (cuDSS: a second handle cycled
    over them; own LU: the forward solver's active prefix, no second solver).
    The own LU ends every row as cuDSS does."""
    runs = {}
    for ls in SOLVERS:
        ca, nbr = _n1(case118, ls, scheduling)
        if scheduling == "continuous":
            ca.tail_max_groups, ca.tail_group_size = 10 ** 6, 3
        ca.compute(batch_size=64)
        assert ca.linear_solver == ls
        _assert_solver(ca.timings, ls)
        runs[ls] = _outcome(ca, nbr)
        if scheduling == "continuous":
            assert ca.timings.n_tail_rounds > 0
    st, _, V = runs["cudss"]
    conv = st == CONVERGED
    nan_bus = np.isnan(V[conv])
    assert nan_bus.any() and not nan_bus.all(axis=1).any()   # converged rows with a masked island
    _same_rows(runs["own_lu"], runs["cudss"], solver_atol)


@pytest.mark.parametrize("G", [1, 3, 15])
def test_continuous_tail_on_the_active_prefix(case118, G, solver_atol, monkeypatch, tmp_path):
    """The own LU's continuous tail runs on its forward solver's active prefix
    (no second solver -- cuDSS initializes one more for its tail): every row
    ends as without the tail, statuses and iterations equal, V to rounding
    (~1e-14 in FP64: the prefix SpMV)."""
    runs, timings, solvers = {}, {}, {}
    for ls, mg in (("own_lu", 0), ("own_lu", 10 ** 6), ("cudss", 10 ** 6)):
        d = tmp_path / f"{ls}_{mg}"
        d.mkdir()
        monkeypatch.setenv("GPUSIM2GRID_DUMP_J", str(d))
        monkeypatch.setenv("GPUSIM2GRID_DUMP_J_MAX", "0")   # skeletons only
        ca, nbr = _n1(case118, ls, "continuous")
        ca.tail_max_groups, ca.tail_group_size = mg, G
        ca.compute(batch_size=64)
        runs[ls, mg], timings[ls, mg] = _outcome(ca, nbr), ca.timings
        solvers[ls, mg] = list(_dumped_solvers(d).values())
    assert timings["own_lu", 0].n_tail_rounds == 0
    t = timings["own_lu", 10 ** 6]
    assert t.n_tail_rounds > 0 and t.n_slot_moves > 0
    assert timings["cudss", 10 ** 6].n_tail_rounds > 0
    assert solvers["own_lu", 0] == solvers["own_lu", 10 ** 6] == ["own_lu"]
    assert solvers["cudss", 10 ** 6] == ["cudss", "cudss"]
    _same_rows(runs["own_lu", 10 ** 6], runs["own_lu", 0], 1e-12 if PREC == "fp64" else solver_atol)


def test_injection_sweep(case118, solver_atol):
    """An injection sweep of case118 (40 rows scaled 0.6-1.4, per-bus noise)
    ends on the own LU exactly as on cuDSS: statuses, iterations, V."""
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
        _assert_solver(sw.timings, ls)
        runs[ls] = _outcome(sw, n_rows)
    _same_rows(runs["own_lu"], runs["cudss"], solver_atol)


def test_array_mode_takes_amd(case118, solver_atol):
    """A tuple grid has no lightsim2grid constraints: 'default' is AMD (the LU
    fill of an explicit 'amd', not that of the identity order), and the own LU
    ends every row as cuDSS does."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid._ls2g_utils import extract_grid_arrays, extract_branch_data
    d = extract_grid_arrays(case118)
    arrays = (d["Ybus"], d["v_converged"], d["Sbus"], d["slack"], d["slack_weights"], d["pv"], d["pq"])
    branch_args, _, _ = extract_branch_data(case118)
    nbr = len(branch_args[0])
    runs, nnz = {}, {}
    for ls, alg in (("cudss", "default"), ("own_lu", "default"), ("own_lu", "amd"), ("own_lu", "none")):
        ca = ContingencyAnalysisGPU(arrays, precision=PREC, nb_iter=10, linear_solver=ls, reordering_alg=alg)
        ca.set_branch_data(*branch_args)
        ca.add_contingencies_by_branch_id([[i] for i in range(nbr)])
        ca.compute(batch_size=64)
        _assert_solver(ca.timings, ls)
        runs[ls, alg], nnz[ls, alg] = _outcome(ca, nbr), ca.timings.cudss_lu_nnz
    assert nnz["own_lu", "default"] == nnz["own_lu", "amd"] < nnz["own_lu", "none"]
    _same_rows(runs["own_lu", "default"], runs["cudss", "default"], solver_atol)


def test_switching_solver_rebuilds_the_driver(case118, solver_atol):
    """linear_solver is mutable: a change builds a new driver on the next run,
    which solves on the new solver (and ends as the old one did)."""
    ca, nbr = _n1(case118, "cudss")
    ca.compute(batch_size=64)
    _assert_solver(ca.timings, "cudss")
    ref = _outcome(ca, nbr)
    builds = ca._inner._s.driver_build_counter
    ca.linear_solver = "own_lu"
    ca.compute(batch_size=64)
    assert ca._inner._s.driver_build_counter == builds + 1
    _assert_solver(ca.timings, "own_lu")
    _same_rows(_outcome(ca, nbr), ref, solver_atol)
    ca.linear_solver = "cudss"
    ca.compute(batch_size=64)
    assert ca._inner._s.driver_build_counter == builds + 2
    _assert_solver(ca.timings, "cudss")


@pytest.mark.parametrize("facade", ["ca", "is", "ss"])
def test_refusals(case118, facade):
    """What the own LU cannot take is refused when it is set -- at
    construction, or by the setter that makes the combination -- not at the
    first compute(), and the refused setter leaves the setting as it was."""
    from gpusim2grid import ContingencyAnalysisGPU, InjectionSweepGPU, ScenarioSweepGPU
    cls = {"ca": ContingencyAnalysisGPU, "is": InjectionSweepGPU, "ss": ScenarioSweepGPU}[facade]
    with pytest.raises(ValueError, match="linear_solver"):
        cls(case118, precision=PREC, linear_solver="klu")
    for kw, match in (({"reordering_alg": "nested_dissection"}, "reordering_alg"),
                      ({"matching_alg": "max_diag_product"}, "matching_alg")):
        with pytest.raises(ValueError, match=match):
            cls(case118, precision=PREC, linear_solver="own_lu", **kw)
        # the combination made by a setter, either way round
        obj = cls(case118, precision=PREC, linear_solver="own_lu")
        (name, value), = kw.items()
        before = getattr(obj, name)
        with pytest.raises(ValueError, match=match):
            setattr(obj, name, value)
        assert getattr(obj, name) == before
        obj = cls(case118, precision=PREC, **kw)
        with pytest.raises(ValueError, match=match):
            obj.linear_solver = "own_lu"
        assert obj.linear_solver == "cudss"


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
        _assert_solver(t[alg], "own_lu")
        assert t[alg].cudss_npivots_reads > 0, alg
    assert t["default"].cudss_lu_nnz == t["ls2g_constrained"].cudss_lu_nnz
    assert t["default"].cudss_npivots_total == 0
    assert t["ls2g_constrained"].cudss_npivots_total == 0
    assert t["amd"].cudss_npivots_total > 0


def _ieee118_outer(min_fill):
    """pypowsybl's ieee118 on NROuter_KLU (it declares ordering constraints),
    its Newton set to lightsim2grid's minimum-fill order or not, solved. AMD
    and MinFill give it different fills."""
    pp = pytest.importorskip("pypowsybl")
    from lightsim2grid.network.from_pypowsybl import init as init_from_pypowsybl
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pypowsybl(pp.network.create_ieee118(), sort_index=False, buses_for_sub=False)
    grid.change_algorithm("NROuter_KLU")
    if min_fill:
        cfg = grid.get_ac_algo_config()
        ip = list(cfg.int_params)
        ip[4] |= 2   # bit 1 of the ordering flags: FillReducingOrdering.MinFill
        cfg.int_params = ip
        grid.set_ac_algo_config(cfg)
    assert grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), 30, 1e-6).size
    return grid


@pytest.mark.skipif(bool(_cpp.is_fp32), reason="tol 1e-6 MVA (lightsim2grid's) is out of FP32's reach")
@pytest.mark.parametrize("min_fill", [False, True], ids=["grid_amd", "grid_minfill"])
def test_constrained_order_follows_the_grids_fill_reducing_ordering(min_fill, monkeypatch, solver_atol):
    """'default' / 'ls2g_constrained' start lightsim2grid's constrained order
    from the fill-reducing order the grid's own Newton uses;
    'ls2g_constrained_minfill' forces minimum fill. Read through the LU fill,
    which differs between the two on this grid; nothing clamped, same V."""
    from gpusim2grid import ContingencyAnalysisGPU
    monkeypatch.setenv("GPUSIM2GRID_COUNT_PIVOTS", "1")
    grid = _ieee118_outer(min_fill)
    sn = grid.get_sn_mva()
    t, V = {}, {}
    for alg in ("default", "ls2g_constrained", "ls2g_constrained_minfill"):
        ca = ContingencyAnalysisGPU(grid, precision=PREC, outer_loops=True, nb_iter=30, tol=1e-6 / sn,
                                    reordering_alg=alg, linear_solver="own_lu")
        ca.add_contingencies_by_branch_id([[0], [1], [2]])
        ca.compute(batch_size=3)
        assert (np.asarray(ca.get_row_status()) == CONVERGED).all(), alg
        t[alg], V[alg] = ca.timings, ca.V_results.to_numpy()
        assert t[alg].cudss_npivots_total == 0, alg
        np.testing.assert_allclose(V[alg], V["default"], atol=solver_atol, rtol=0, err_msg=alg)
    assert t["default"].cudss_lu_nnz == t["ls2g_constrained"].cudss_lu_nnz
    same = t["default"].cudss_lu_nnz == t["ls2g_constrained_minfill"].cudss_lu_nnz
    assert same == min_fill


# =============================================================================
# The adjoint (a second batch solver, over J^T)
# =============================================================================
@pytest.mark.skipif(bool(_cpp.is_fp32), reason="finite-difference gradient checks need the FP64 build")
def test_batch_power_flow_gradient(ieee14_base_case, monkeypatch, tmp_path):
    """BatchPowerFlow's backward solves J^T on the own LU (the forward too:
    every batch solver initialized is an own LU, the J^T one included):
    central differences on every differentiable input (load_p, load_q, gen_p,
    gen_v), a row with a tripped line included."""
    torch = pytest.importorskip("torch")
    from test_batch_power_flow import _pf, _base_inputs, _all_connected, _fd_check, RDT
    monkeypatch.setenv("GPUSIM2GRID_DUMP_J", str(tmp_path))
    monkeypatch.setenv("GPUSIM2GRID_DUMP_J_MAX", "0")
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
    assert pf.timings.adjoint_n_solve > 0
    solvers = _dumped_solvers(tmp_path)
    assert len(solvers) >= 2 and set(solvers.values()) == {"own_lu"}, solvers


@pytest.mark.skipif(bool(_cpp.is_fp32), reason="tol 1e-6 MVA (lightsim2grid's) is out of FP32's reach")
def test_mutated_reordering_alg_gets_its_own_order(monkeypatch):
    """reordering_alg is mutable: a session built with 'ls2g_constrained' and
    switched to 'ls2g_constrained_minfill' (own LU) must solve with the
    minimum-fill order, as a session built with it does, not with the order the
    base state computed for its construction value. Read through the LU fill
    (AMD and MinFill differ on this grid). The cuDSS side: switched back to
    'default', cuDSS must get no user permutation at all."""
    from gpusim2grid import ContingencyAnalysisGPU
    grid = _ieee118_outer(False)
    sn = grid.get_sn_mva()

    def ca_(alg, solver):
        ca = ContingencyAnalysisGPU(grid, precision=PREC, outer_loops=True, nb_iter=30, tol=1e-6 / sn,
                                    reordering_alg=alg, linear_solver=solver)
        ca.add_contingencies_by_branch_id([[0], [1], [2]])
        return ca

    fresh = ca_("ls2g_constrained_minfill", "own_lu")
    fresh.compute(batch_size=3)
    amd = ca_("ls2g_constrained", "own_lu")
    amd.compute(batch_size=3)
    assert fresh.timings.cudss_lu_nnz != amd.timings.cudss_lu_nnz   # the grid tells them apart
    amd.reordering_alg = "ls2g_constrained_minfill"
    amd.compute(batch_size=3)
    assert amd.timings.cudss_lu_nnz == fresh.timings.cudss_lu_nnz

    cu = ca_("ls2g_constrained", "cudss")
    cu.compute(batch_size=3)
    assert cu.timings.cudss_user_perm_honoured != -1
    cu.reordering_alg = "default"
    cu.compute(batch_size=3)
    assert cu.timings.cudss_user_perm_honoured == -1


_RAW_BAD_INPUTS = r"""
import sys
import gpusim2grid._gpusim2grid as _cpp
case = sys.argv[1]
dim, indptr, indices, data = {
    "short_indptr": (200000, [0, 1, 2], [0, 1], [1.0, 1.0]),
    "dim_zero": (0, [0], [], []),
    "dim_negative": (-3, [0], [], []),
    "indices_out_of_range": (2, [0, 1, 2], [0, 7], [1.0, 1.0]),
}[case]
try:
    _cpp.benchmark_cudss_batch_raw(dim, indptr, indices, data, 2, n_refactorize=1,
                                   linear_solver=_cpp.LinearSolver.OwnLu)
except RuntimeError as e:
    print("REFUSED:", e)
"""


@pytest.mark.parametrize("case", ["short_indptr", "dim_zero", "dim_negative", "indices_out_of_range"])
def test_raw_own_lu_validates_before_ordering(case):
    """benchmark_cudss_batch_raw(linear_solver=OwnLu) checks the CSR it is
    handed before computing the own LU's order on it: a malformed matrix is
    refused with the benchmark's own message, never read out of bounds (run in
    a subprocess: an out-of-bounds read may crash the interpreter)."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", _RAW_BAD_INPUTS, case],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
    assert "REFUSED: benchmark_cudss_batch_raw:" in r.stdout, (r.stdout, r.stderr[-2000:])



def test_calibrate_tail_max_groups_own_lu(case118):
    """The own LU's tail runs k groups as ONE prefix call of k·G matrices, so
    k groups pay while that prefix is faster than the full batch -- not while
    k · t(G) is (a prefix costs far less than k separate groups). Re-measured
    up to timing noise: the calibrated value's prefix is not slower than the
    full batch, and one group more is either impossible (k·G reaches the
    batch) or not clearly cheaper than it."""
    from gpusim2grid import ContingencyAnalysisGPU
    S, G = 512, 15
    ca = ContingencyAnalysisGPU(case118, precision=PREC, nb_iter=10, scheduling="continuous",
                                linear_solver="own_lu")
    ca.tail_group_size = G
    c = ca.calibrate_tail_max_groups(n_repeat=5, batch_size=S)
    assert c.t_full_ms > 0 and c.t_group_ms > 0
    assert 1 <= c.max_groups and c.max_groups * G < S
    if c.max_groups > 1:
        ca.tail_group_size = c.max_groups * G
        cur = ca.calibrate_tail_max_groups(n_repeat=5, batch_size=S)
        assert cur.t_group_ms <= 1.25 * cur.t_full_ms, (c.max_groups, cur.t_group_ms, cur.t_full_ms)
    if (c.max_groups + 1) * G < S:
        ca.tail_group_size = (c.max_groups + 1) * G
        nxt = ca.calibrate_tail_max_groups(n_repeat=5, batch_size=S)
        assert nxt.t_group_ms >= 0.8 * c.t_full_ms, (c.max_groups, nxt.t_group_ms, c.t_full_ms)


@pytest.mark.parametrize("linear_solver", SOLVERS)
def test_dump_j_names_values_by_their_precision(case118, linear_solver, monkeypatch, tmp_path):
    """GPUSIM2GRID_DUMP_J (diagnostic): every batch solver writes its J values
    as raw files whose extension says the element type the build writes
    (``.f64`` in FP64, ``.f32`` in FP32), each B · nnz elements of the
    ``real_bytes`` its meta file records."""
    monkeypatch.setenv("GPUSIM2GRID_DUMP_J", str(tmp_path))
    monkeypatch.setenv("GPUSIM2GRID_DUMP_J_MAX", "2")
    ca, _ = _n1(case118, linear_solver)
    ca.compute(batch_size=16)
    metas = sorted(tmp_path.glob("h*_meta.txt"))
    assert metas
    for meta in metas:
        kv = dict(line.split() for line in meta.read_text().splitlines())
        rb, ub, nnz = int(kv["real_bytes"]), int(kv["ubatch"]), int(kv["nnz"])
        assert rb == (4 if _cpp.is_fp32 else 8)
        h = meta.name.split("_")[0]
        vals = sorted(tmp_path.glob(f"{h}_vals_*"))
        assert vals, meta.name
        for v in vals:
            assert v.suffix == f".f{8 * rb}", v.name
            assert v.stat().st_size == ub * nnz * rb, v.name


@requires_gpu
def test_fused_solve_levels_give_the_same_residual(case118, residual_atol, monkeypatch):
    """the triangular solves' narrow levels run fused (one kernel per run of levels, a block barrier
    between them; GPUSIM2GRID_OWN_LU_FUSE=0: one launch per level): every value computed as the
    level kernels do, so the raw benchmark's residual is the same to the bit. The solutions
    themselves, and that the levels did fuse, are compared in tests/cpp/own_lu/test_fused_solve.cu
    (the raw benchmark returns no solution)"""
    from gpusim2grid import AcPfGPU
    pf = AcPfGPU(case118, precision=PREC)
    indptr, indices, data = (np.asarray(a) for a in pf._s.get_J())
    n = len(indptr) - 1
    res = {}
    for fuse in ("0", "1"):
        monkeypatch.setenv("GPUSIM2GRID_OWN_LU_FUSE", fuse)
        r = _cpp.benchmark_cudss_batch_raw(n, indptr.tolist(), indices.tolist(), data.astype(float).tolist(), 40,
                                           n_refactorize=1, ir_n_steps=1, linear_solver=_cpp.LinearSolver.OwnLu)
        res[fuse] = r.max_rel_residual
    assert 0. < res["0"] < residual_atol
    assert res["1"] == res["0"]

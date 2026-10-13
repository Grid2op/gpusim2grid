# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""handle_disconnected_grid: solve the largest connected component of a split grid.

When an N-k contingency disconnects the grid, gpusim2grid can (opt-in) solve the
largest connected component and report the frozen (islanded) buses as NaN — the
GPU analogue of lightsim2grid's ``ContingencyAnalysisCPP.handle_disconnected_grid``.
The reference is lightsim2grid's own masked solve (set to 0 there, NaN here).

These exercise the bridge path (the masking metadata — angle reference, controller
buses, per-bus identity rows — is read off the solved C++ grid). A radial spur bus
is added so a single line trip cleanly islands it.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="handle_disconnected_grid parity needs the lightsim2grid C++ bridge")


def _solved_spur_grid(distributed_slack=False):
    """case14 + a radial spur bus (one line, one load); ac-solved with NR_KLU.

    Returns (grid, n_bus_model, spur_line_id, spur_bus_id). Tripping the spur
    line islands exactly the spur bus (the largest component is the original
    grid). With ``distributed_slack`` a 2nd ext_grid sits ON the spur bus, so the
    trip strands a NON-reference distributed-slack participant — the case the GPU
    handles by identity-masking its P-row (auto-rescaling the live weights).
    """
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        spur_bus = pp.create_bus(net, vn_kv=float(net.bus.vn_kv.iloc[0]))
        pp.create_line_from_parameters(
            net, from_bus=0, to_bus=spur_bus, length_km=1.0,
            r_ohm_per_km=0.05, x_ohm_per_km=0.2, c_nf_per_km=0.0, max_i_ka=1.0)
        pp.create_load(net, bus=spur_bus, p_mw=5.0, q_mvar=2.0)
        if distributed_slack:
            pp.create_ext_grid(net, bus=spur_bus, vm_pu=1.0, va_degree=0.0,
                               slack_weight=1.0)
            net.ext_grid.loc[0, "slack_weight"] = 1.0
            pp.runpp(net, distributed_slack=True)
        else:
            pp.runpp(net)

        spur_line_id = len(net.line) - 1   # spur is the last line (lines-then-trafos)
        grid = init_from_pandapower(net)
        grid.change_algorithm(AlgorithmType.NR_KLU)
        n_bus = grid.get_bus_vn_kv().shape[0]
        v0 = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
        grid.ac_pf(v0.copy(), 30, 1e-10)
    return grid, n_bus, spur_line_id, spur_bus


def _ls2g_masked_reference(grid, n_bus, spur_line_id):
    """lightsim2grid's own handle_disconnected_grid solve (masked buses → 0)."""
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    ca = ContingencyAnalysisCPP(grid)
    ca.handle_disconnected_grid = True
    ca.add_n1(int(spur_line_id))
    ca.compute(np.ones(n_bus, dtype=complex), 30, 1e-10)
    return np.asarray(ca.get_voltages())[0]


def _gpu_masked(grid, n_bus, spur_line_id, **kwargs):
    from gpusim2grid import ContingencyAnalysisGPU

    g = ContingencyAnalysisGPU(grid, handle_disconnected_grid=True,
                               precision=None, nb_iter=15, tol_base=1e-10, **kwargs)
    g.add_contingencies_by_branch_id([[int(spur_line_id)]])
    g.compute(batch_size=8)
    V = g.V_results.to_numpy().reshape(1, n_bus)[0]
    return V, g.last_residuals()[0]


@requires_gpu
@needs_bridge
@pytest.mark.parametrize("distributed_slack", [False, True],
                         ids=["single_slack", "distributed_slack"])
def test_largest_component_matches_ls2g(distributed_slack, solver_atol):
    """The masked contingency converges; the live block matches lightsim2grid's
    masked solve and the islanded spur bus is reported as NaN."""
    grid, n_bus, spur_line, spur_bus = _solved_spur_grid(distributed_slack)
    V_ref = _ls2g_masked_reference(grid, n_bus, spur_line)
    V_gpu, residual = _gpu_masked(grid, n_bus, spur_line)

    # lightsim2grid masks the islanded bus to 0; the GPU reports NaN.
    assert V_ref[spur_bus] == 0
    assert np.isnan(V_gpu[spur_bus])

    # The contingency is actually solved (not skipped): residual is ~0.
    assert np.isfinite(residual) and residual < 100 * solver_atol

    # The live (main-component) buses match lightsim2grid bus-by-bus.
    main = np.ones(n_bus, dtype=bool)
    main[spur_bus] = False
    assert np.all(np.isfinite(V_gpu[main]))
    np.testing.assert_allclose(V_gpu[main], V_ref[main], atol=10 * solver_atol)


def _solved_spur_primary_slack_grid():
    """case14 whose PRIMARY (reference) slack sits on a radial spur, plus a 2nd
    distributed slack on a central bus. The default reference (the spur) is
    stranded by the spur-line trip; a contingency-aware reference choice (the
    central slack) is not. Returns (grid, n_bus, n_ctg, spur_line_id)."""
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        net.ext_grid = net.ext_grid.iloc[0:0]               # drop the original slack
        spur = pp.create_bus(net, vn_kv=float(net.bus.vn_kv.iloc[0]))
        pp.create_line_from_parameters(
            net, from_bus=0, to_bus=spur, length_km=1.0,
            r_ohm_per_km=0.05, x_ohm_per_km=0.2, c_nf_per_km=0.0, max_i_ka=1.0)
        pp.create_load(net, bus=spur, p_mw=3.0, q_mvar=1.0)
        # spur slack is created FIRST → it is the default angle reference
        pp.create_ext_grid(net, bus=spur, vm_pu=1.0, va_degree=0.0, slack_weight=1.0)
        pp.create_ext_grid(net, bus=4, vm_pu=1.0, va_degree=0.0, slack_weight=1.0)
        pp.runpp(net, distributed_slack=True)
        spur_line_id = len(net.line) - 1
        n_ctg = len(net.line) + len(net.trafo)
        grid = init_from_pandapower(net)
        grid.change_algorithm(AlgorithmType.NR_KLU)
        n_bus = grid.get_bus_vn_kv().shape[0]
        grid.ac_pf(np.ones(n_bus, dtype=complex), 30, 1e-10)
    return grid, n_bus, n_ctg, spur_line_id


@requires_gpu
@needs_bridge
def test_optimize_reference_slack_reduces_skips():
    """optimize_reference_slack picks the slack stranded by the fewest
    contingencies and re-solves the base case with it as the reference, so the
    GPU skips strictly fewer split contingencies."""
    from gpusim2grid import ContingencyAnalysisGPU, optimize_reference_slack

    grid, n_bus, n_ctg, _ = _solved_spur_primary_slack_grid()
    cont = [[c] for c in range(n_ctg)]

    def n_skips():
        # the grid's own reference (the facades pick one themselves by default,
        # see test_automatic_reference_slack below)
        g = ContingencyAnalysisGPU(grid, handle_disconnected_grid=True, reference_slack="grid",
                                   precision=None, nb_iter=12, tol_base=1e-10)
        g.add_contingencies_by_branch_id(cont)
        g.compute(batch_size=64)
        return int(np.isnan(g.last_residuals()).sum())

    skips_default = n_skips()
    ref = optimize_reference_slack(grid, cont)
    skips_optimized = n_skips()

    assert ref is not None and ref >= 0
    assert skips_optimized < skips_default


def _gpu_ca_spur(grid, cont, **kw):
    from gpusim2grid import ContingencyAnalysisGPU
    g = ContingencyAnalysisGPU(grid, handle_disconnected_grid=True, precision=None,
                               nb_iter=12, tol_base=1e-10, **kw)
    g.add_contingencies_by_branch_id(cont)
    g.compute(batch_size=64)
    n_bus = grid.get_Ybus_solver().shape[0]
    return g, g.V_results.to_numpy().reshape(len(cont), n_bus)


@requires_gpu
@needs_bridge
@pytest.mark.parametrize("path", ["contingency_analysis", "scenario_sweep"])
def test_automatic_reference_slack(solver_atol, path):
    """reference_slack="auto" (the default, lightsim2grid PR #216's batch rule):
    the reference becomes the slack participant the fewest contingencies strand
    -- the central one here --, so the spur trip is solved instead of skipped;
    the other rows are the same state as with the grid's reference, up to a
    constant angle shift, and the whole batch equals lightsim2grid's own."""
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    grid, n_bus, n_ctg, spur_line = _solved_spur_primary_slack_grid()
    cont = [[c] for c in range(n_ctg)]
    V0 = grid.get_V_solver() if hasattr(grid, "get_V_solver") else None

    if path == "contingency_analysis":
        g_auto, V_auto = _gpu_ca_spur(grid, cont)
        g_grid, V_grid = _gpu_ca_spur(grid, cont, reference_slack="grid")
        res_auto, res_grid = g_auto.last_residuals(), g_grid.last_residuals()
    else:
        from gpusim2grid import ScenarioSweepGPU

        def ss(**kw):
            sw = ScenarioSweepGPU(grid, handle_disconnected_grid=True, nb_iter=12,
                                  tol_base=1e-10, precision=None, **kw)
            S = grid.get_Sbus_solver()
            sn = grid.get_sn_mva()
            sw.set_injections(np.repeat((S.real * sn)[None, :], n_ctg, axis=0),
                              np.repeat((S.imag * sn)[None, :], n_ctg, axis=0), sn)
            sw.set_topology(cont)
            sw.compute(batch_size=64)
            return sw, sw.solver.V_results.to_numpy().reshape(n_ctg, n_bus)
        g_auto, V_auto = ss()
        g_grid, V_grid = ss(reference_slack="grid")
        res_auto, res_grid = g_auto.last_residuals(), g_grid.last_residuals()

    assert g_auto.reference_bus != g_grid.reference_bus
    assert np.isnan(res_grid[spur_line]) and np.isfinite(res_auto[spur_line])
    assert int(np.isnan(res_auto).sum()) < int(np.isnan(res_grid).sum())
    # the rows both solve: same magnitudes, and within a row every angle
    # shifted by the same constant (each row pins another bus at its base-case
    # angle, so the constant is the row's own)
    for r in np.flatnonzero(np.isfinite(res_auto) & np.isfinite(res_grid)):
        live = np.isfinite(V_auto[r]) & np.isfinite(V_grid[r])
        np.testing.assert_allclose(np.abs(V_auto[r][live]), np.abs(V_grid[r][live]),
                                   atol=10 * solver_atol)
        shift = np.angle(V_auto[r][live]) - np.angle(V_grid[r][live])
        shift = np.angle(np.exp(1j * (shift - shift[0])))
        np.testing.assert_allclose(shift, 0., atol=10 * solver_atol)

    # lightsim2grid's own batch picks the same reference: same voltages
    ls = ContingencyAnalysisCPP(grid)
    ls.handle_disconnected_grid = True
    for c in cont:
        ls.add_n1(int(c[0]))
    n_bus_grid = grid.get_bus_vn_kv().shape[0]
    ls.compute(np.asarray(grid.get_V()) if hasattr(grid, "get_V") else np.ones(n_bus_grid, dtype=complex),
               30, 1e-10)
    rows = {int(list(c)[0]): i for i, c in enumerate(ls.my_defaults())}
    V_ls = np.asarray(ls.get_voltages())
    conv = np.asarray(ls.converged_mask(), dtype=bool)
    s2me = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    for c in range(n_ctg):
        i = rows[c]
        assert conv[i] == bool(np.isfinite(res_auto[c])), c
        if not conv[i]:
            continue
        ok = np.isfinite(V_auto[c])
        np.testing.assert_allclose(V_auto[c][ok], V_ls[i][s2me][ok], atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_moved_reference_is_lightsim2grids_own_ledger():
    """move_reference (what the automatic reference slack builds) gives exactly
    the augmented-J skeleton, maps and registrations lightsim2grid itself poses
    once the reference is forced there and the grid re-solved."""
    from gpusim2grid._gpusim2grid import _ledger_skeleton
    grid, n_bus, n_ctg, spur_line = _solved_spur_primary_slack_grid()
    # the grid's reference is the spur slack; the other participant is bus 4
    target = int(np.asarray(grid.id_me_to_ac_solver())[4])
    assert _ledger_skeleton(grid)["reference_bus"] != target
    ref_grid = grid.copy()
    ref_grid.set_reference_slack_bus(4)
    ref_grid.ac_pf(np.ones(grid.get_bus_vn_kv().shape[0], dtype=complex), 30, 1e-10)
    native = _ledger_skeleton(ref_grid)
    moved = _ledger_skeleton(grid, target)
    assert native["reference_bus"] == moved["reference_bus"] == target
    for key in native:
        a, b = moved[key], native[key]
        if isinstance(a, int):
            assert a == b, key
        else:
            assert list(a) == list(b), key


@requires_gpu
@needs_bridge
def test_forced_reference_slack_is_kept():
    """A reference forced on the grid (LSGrid.set_reference_slack_bus) is kept
    by the automatic choice, like lightsim2grid's batch keeps it."""
    grid, n_bus, n_ctg, spur_line = _solved_spur_primary_slack_grid()
    cont = [[c] for c in range(n_ctg)]
    g_auto, _ = _gpu_ca_spur(grid, cont)
    g_grid, _ = _gpu_ca_spur(grid, cont, reference_slack="grid")
    s2me = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    grid.set_reference_slack_bus(int(s2me[g_grid.reference_bus]))
    grid.ac_pf(np.ones(grid.get_bus_vn_kv().shape[0], dtype=complex), 30, 1e-10)
    g_forced, _ = _gpu_ca_spur(grid, cont)
    assert g_forced.reference_bus == g_grid.reference_bus != g_auto.reference_bus
    assert np.isnan(g_forced.last_residuals()[spur_line])


@requires_gpu
@needs_bridge
def test_flag_off_skips_the_contingency():
    """With handle_disconnected_grid disabled, an islanding contingency is skipped
    (legacy behaviour): NaN voltages and a NaN residual."""
    from gpusim2grid import ContingencyAnalysisGPU

    grid, n_bus, spur_line, _ = _solved_spur_grid(distributed_slack=False)
    g = ContingencyAnalysisGPU(grid, handle_disconnected_grid=False,
                               precision=None, nb_iter=15, tol_base=1e-10)
    g.add_contingencies_by_branch_id([[int(spur_line)]])
    g.compute(batch_size=8)
    V = g.V_results.to_numpy().reshape(1, n_bus)[0]
    assert np.isnan(g.last_residuals()[0])
    assert np.all(np.isnan(V))


def _solved_double_spur_grid():
    """case14 + a radial spur bus fed by TWO identical parallel lines (a double
    circuit) and one load; ac-solved with NR_KLU.

    Returns (grid, n_bus_model, (line_a, line_b), spur_bus_id). Tripping one
    circuit leaves the spur fed by the other; tripping both islands it.
    """
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        spur_bus = pp.create_bus(net, vn_kv=float(net.bus.vn_kv.iloc[0]))
        for _ in range(2):
            pp.create_line_from_parameters(
                net, from_bus=0, to_bus=spur_bus, length_km=1.0,
                r_ohm_per_km=0.05, x_ohm_per_km=0.2, c_nf_per_km=0.0, max_i_ka=1.0)
        pp.create_load(net, bus=spur_bus, p_mw=5.0, q_mvar=2.0)
        pp.runpp(net)
        line_b = len(net.line) - 1
        line_a = line_b - 1
        grid = init_from_pandapower(net)
        grid.change_algorithm(AlgorithmType.NR_KLU)
        n_bus = grid.get_bus_vn_kv().shape[0]
        v0 = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
        grid.ac_pf(v0.copy(), 30, 1e-10)
    return grid, n_bus, (line_a, line_b), spur_bus


@requires_gpu
@needs_bridge
def test_double_circuit_n2_islands_the_spur(solver_atol):
    """Tripping ONE circuit of a double line keeps the spur connected; tripping
    BOTH in a single N-2 contingency islands it. The split is only visible on
    the SUMMED Ybus patch (each circuit's own delta leaves the shared entry
    non-zero), which the connectivity check must account for -- otherwise the
    spur bus is left live on a singular system instead of being masked."""
    from gpusim2grid import ContingencyAnalysisGPU
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    grid, n_bus, (line_a, line_b), spur_bus = _solved_double_spur_grid()
    ctg = [[line_a], [line_b], [line_a, line_b]]

    g = ContingencyAnalysisGPU(grid, handle_disconnected_grid=True,
                               precision=None, nb_iter=15, tol_base=1e-10)
    g.add_contingencies_by_branch_id(ctg)
    g.compute(batch_size=8)
    V = g.V_results.to_numpy().reshape(len(ctg), n_bus)
    residual = g.last_residuals()

    # One circuit out: the spur stays fed, nothing is masked, all converge.
    assert np.all(np.isfinite(V[:2]))
    assert np.all(residual[:2] < 100 * solver_atol)

    # Both circuits out: the spur is islanded (NaN) and the rest is solved.
    assert np.isnan(V[2, spur_bus])
    assert np.isfinite(residual[2]) and residual[2] < 100 * solver_atol
    main = np.ones(n_bus, dtype=bool)
    main[spur_bus] = False
    assert np.all(np.isfinite(V[2, main]))

    # ... and matches lightsim2grid's own masked N-2 solve bus-by-bus.
    ca = ContingencyAnalysisCPP(grid)
    ca.handle_disconnected_grid = True
    ca.add_nk([int(line_a), int(line_b)])
    ca.compute(np.ones(n_bus, dtype=complex), 30, 1e-10)
    V_ref = np.asarray(ca.get_voltages())[0]
    assert V_ref[spur_bus] == 0
    np.testing.assert_allclose(V[2, main], V_ref[main], atol=10 * solver_atol)

    # With the flag off, the same N-2 is skipped outright (legacy behaviour).
    g_off = ContingencyAnalysisGPU(grid, handle_disconnected_grid=False,
                                   precision=None, nb_iter=15, tol_base=1e-10)
    g_off.add_contingencies_by_branch_id(ctg)
    g_off.compute(batch_size=8)
    residual_off = g_off.last_residuals()
    assert np.all(np.isfinite(residual_off[:2]))
    assert np.isnan(residual_off[2])

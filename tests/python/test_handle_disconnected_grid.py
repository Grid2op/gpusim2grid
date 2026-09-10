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
        g = ContingencyAnalysisGPU(grid, handle_disconnected_grid=True,
                                   precision=None, nb_iter=12, tol_base=1e-10)
        g.add_contingencies_by_branch_id(cont)
        g.compute(batch_size=64)
        return int(np.isnan(g.last_residuals()).sum())

    skips_default = n_skips()
    ref = optimize_reference_slack(grid, cont)
    skips_optimized = n_skips()

    assert ref is not None and ref >= 0
    assert skips_optimized < skips_default


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

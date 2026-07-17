"""Augmented Newton-Raphson features (lightsim2grid parity).

These exercise gpusim2grid solving the SAME augmented system lightsim2grid does
— more than the bare ``[pvpq | pq]`` Jacobian. Phase 2 covers the MultiSlack
extension (distributed slack in the Jacobian). The reference is the solved
``LSGrid`` (itself validated against pypowsybl / OpenLoadFlow upstream).

The augmented path requires the C++ bridge (``have_ls2g_bridge``) since the
NRLedger + J skeleton are read off the solved C++ grid.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="augmented features need the lightsim2grid C++ bridge")


def _solved_multislack_grid(extra_slack_bus=3):
    """case14 with a 2nd ext_grid → genuine distributed slack (N=2), ac-solved."""
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        pp.create_ext_grid(net, bus=extra_slack_bus, vm_pu=1.0, va_degree=0.0,
                           slack_weight=1.0)
        net.ext_grid.loc[0, "slack_weight"] = 1.0
        pp.runpp(net, distributed_slack=True)
        grid = init_from_pandapower(net)
        grid.change_algorithm(AlgorithmType.NR_KLU)   # MultiSlackNRSystem
        n_model = grid.get_bus_vn_kv().shape[0]
        v0 = grid.dc_pf(np.ones(n_model, dtype=complex), 1, 1e-6)
        grid.ac_pf(v0.copy(), 30, 1e-10)
    return grid


@requires_gpu
@needs_bridge
def test_acpf_gpu_single_slack_augmented_matches(ieee14_base_case, solver_atol):
    """AcPfGPU via the bridge solves the augmented (slack_absorbed) system and
    matches lightsim2grid even for an ordinary single-slack grid (N=1)."""
    from gpusim2grid import AcPfGPU

    grid = ieee14_base_case["grid"]
    V_ref = grid.get_V_solver()

    ac = AcPfGPU(grid, max_iter=30, tol=1e-10)   # use_bridge auto-detected
    V_gpu = ac.solve()

    # NR_KLU uses MultiSlackNRSystem, so the augmented J carries +1 column
    # (slack_absorbed) over the bare pvpq/pq system even with one slack.
    n_pv = grid.get_pv().shape[0]
    n_pq = grid.get_pq().shape[0]
    assert ac.session.dim_J == n_pv + 2 * n_pq + 1

    np.testing.assert_allclose(V_gpu, V_ref, atol=10 * solver_atol)


_BIG_Q = 1.0e6


def _solved_hvdc_droop_grid(b1=3, b2=9, p0=10.0, droop_mw_per_deg=5.0,
                            lf1=0.01, lf2=0.02):
    """case14 with one droop-enabled HVDC line, ac-solved (mirrors the
    lightsim2grid HVDC test helper ``make_case14_hvdc``)."""
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        model = init_from_pandapower(net)
        if not hasattr(model, "init_hvdc_lines"):
            pytest.skip("this lightsim2grid build has no init_hvdc_lines")
        model.init_hvdc_lines(
            np.array([b1], dtype=np.int32), np.array([b2], dtype=np.int32),
            [0], [0],                                  # VSC / VSC
            np.array([lf1]), np.array([lf2]),
            [False], [False],                          # no local voltage reg
            np.array([1.0]), np.array([1.0]),
            np.array([0.0]), np.array([0.0]),
            np.array([-_BIG_Q]), np.array([_BIG_Q]),
            np.array([-_BIG_Q]), np.array([_BIG_Q]),
            np.array([1.0]), np.array([1.0]),
            [0],
            np.array([20.0]),
            np.array([0.0]), np.array([0.0]),
            [True],                                    # droop enabled
            np.array([p0]), np.array([droop_mw_per_deg]),
            np.array([300.0]), np.array([300.0]),
        )
        model.tell_solver_need_reset()
        model.change_algorithm(AlgorithmType.NR_KLU)
        n_model = model.get_bus_vn_kv().shape[0]
        v0 = model.dc_pf(np.ones(n_model, dtype=complex), 1, 1e-6)
        model.ac_pf(v0.copy(), 30, 1e-10)
    return model


@requires_gpu
@needs_bridge
def test_acpf_gpu_hvdc_droop_matches(solver_atol):
    """AcPfGPU via the bridge reproduces lightsim2grid on a grid with an HVDC
    angle-droop line (Va-dependent slopes + theta-dependent flows)."""
    from gpusim2grid import AcPfGPU

    model = _solved_hvdc_droop_grid()
    V_ref = model.get_V_solver()

    ac = AcPfGPU(model, max_iter=40, tol=1e-11)
    V_gpu = ac.solve()

    np.testing.assert_allclose(V_gpu, V_ref, atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_injection_sweep_distributed_slack_base_case(solver_atol):
    """Batched InjectionSweep on a distributed-slack grid: a single scenario equal
    to the base case reproduces lightsim2grid's V (augmented batch path)."""
    from gpusim2grid import InjectionSweepGPU

    grid = _solved_multislack_grid()
    n_bus = grid.get_Ybus_solver().shape[0]
    sn = grid.get_sn_mva()
    Sbus = grid.get_Sbus_solver()
    p_mw = (Sbus.real * sn).reshape(1, n_bus)
    q_mvar = (Sbus.imag * sn).reshape(1, n_bus)

    sweep = InjectionSweepGPU(grid, nb_iter=12)
    sweep.set_injections(p_mw, q_mvar, sn)
    sweep.compute(batch_size=4)

    V = sweep.V_results.to_numpy().reshape(1, n_bus)
    assert sweep.last_residuals()[0] < 10 * solver_atol
    np.testing.assert_allclose(np.abs(V[0]), np.abs(grid.get_V_solver()),
                               atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_contingency_distributed_slack_converges(solver_atol):
    """Batched N-1 contingency on a distributed-slack grid converges (augmented
    system solved under Ybus patching)."""
    from gpusim2grid import ContingencyAnalysisGPU

    grid = _solved_multislack_grid()
    ca = ContingencyAnalysisGPU(grid, init_from_n_powerflow=True, nb_iter=12)
    n_ctg = min(5, len(grid.get_lines()))
    ca.add_contingencies_by_branch_id([[c] for c in range(n_ctg)])
    ca.compute(batch_size=8)

    res = ca.last_residuals()
    finite = res[np.isfinite(res)]
    assert finite.size > 0 and np.all(finite < 1e3 * solver_atol)


def _solve_with_fallback(model, nb_bus):
    """Solve with NR_KLU; fall back to NRSing_SparseLU if a flat start diverges
    (NR_KLU can struggle to start an SVC from a flat profile)."""
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    model.change_algorithm(AlgorithmType.NR_KLU)
    V = model.ac_pf(np.ones(nb_bus, dtype=complex), 40, 1e-11)
    if V.shape[0] == 0:
        model.tell_solver_need_reset()
        model.change_algorithm(AlgorithmType.NRSing_SparseLU)
        V = model.ac_pf(np.ones(nb_bus, dtype=complex), 40, 1e-11)
    assert V.shape[0] > 0, "lightsim2grid diverged"


@requires_gpu
@needs_bridge
def test_acpf_gpu_remote_gen_voltage_control_matches(solver_atol):
    """Remote-regulating generator (bordered VoltageControl): gen on bus 7
    regulating remote bus 9. The augmented J gains a q-column + voltage row."""
    from gpusim2grid import AcPfGPU
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        model = init_from_pandapower(net)
        if not hasattr(model, "set_gen_regulated_bus"):
            pytest.skip("this lightsim2grid build has no set_gen_regulated_bus")
        model.set_gen_regulated_bus(3, 9)   # gen 3 (bus 7) regulates bus 9
        model.tell_solver_need_reset()
        _solve_with_fallback(model, net.bus.shape[0])

    n_pv = model.get_pv().shape[0]
    n_pq = model.get_pq().shape[0]
    V_ref = model.get_V_solver()
    ac = AcPfGPU(model, max_iter=50, tol=1e-11)
    assert ac.session.dim_J > n_pv + 2 * n_pq   # bordered columns/rows added
    np.testing.assert_allclose(ac.solve(), V_ref, atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_acpf_gpu_voltage_control_shared_bus_matches(solver_atol):
    """Two controllers CO-LOCATED at the same bus, sharing reactive power to
    regulate one remote bus (a bordered VoltageControl group with N=2, both
    members at the same vc_bus). Their J columns must stay distinct: the
    ledger's bus-keyed q_to_J_col map only keeps the LAST controller
    registered at a bus (NRLedger::add_q_unknown's own doc), so deriving qcol
    from it aliases the two controllers onto one column and corrupts the
    sharing equations (regression: this used to blow up ||dx||_inf to ~1e11
    and freeze the iterative NR loop on a real ~7000-bus RTE grid with grouped
    voltage control -- gen 3 and a co-located gen both regulating bus 9 here
    reproduces the same structural collision at IEEE 14-bus scale)."""
    from gpusim2grid import AcPfGPU
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        # co-locate a second controllable generator on bus 7 (same bus as gen 3)
        pp.create_gen(net, bus=7, p_mw=0.0, vm_pu=1.09, controllable=True,
                      min_q_mvar=-50., max_q_mvar=50.)
        net.gen.loc[3, ["min_q_mvar", "max_q_mvar"]] = [-50., 50.]
        model = init_from_pandapower(net)
        if not hasattr(model, "set_gen_regulated_bus"):
            pytest.skip("this lightsim2grid build has no set_gen_regulated_bus")
        model.set_gen_regulated_bus(3, 9)   # gen 3 (bus 7) regulates bus 9
        model.set_gen_regulated_bus(4, 9)   # gen 4 (ALSO bus 7) regulates bus 9 too
        model.tell_solver_need_reset()
        _solve_with_fallback(model, net.bus.shape[0])

    qcol = model.get_controller_q_col_solver()
    assert qcol.shape[0] == 2
    assert qcol[0] != qcol[1], "co-located controllers must get distinct J columns"

    V_ref = model.get_V_solver()
    ac = AcPfGPU(model, max_iter=50, tol=1e-11)
    np.testing.assert_allclose(ac.solve(), V_ref, atol=10 * solver_atol)


@requires_gpu
@needs_bridge
@pytest.mark.parametrize("slope_pu,reg_bus", [(0.0, 9), (0.02, 9)])
def test_acpf_gpu_svc_matches(solver_atol, slope_pu, reg_bus):
    """Voltage-mode SVC (a sloped SVC exercises the (v_row, q_col)=slope feature)."""
    from gpusim2grid import AcPfGPU
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        model = init_from_pandapower(net)
        if not hasattr(model, "init_svcs"):
            pytest.skip("this lightsim2grid build has no init_svcs")
        model.init_svcs([1], np.array([1.03]), np.array([0.0]), np.array([slope_pu]),
                        np.array([-100.0]), np.array([100.0]),
                        np.array([reg_bus], dtype=np.int32),
                        np.array([10], dtype=np.int32))   # SVC on bus 10
        model.tell_solver_need_reset()
        _solve_with_fallback(model, net.bus.shape[0])

    V_ref = model.get_V_solver()
    ac = AcPfGPU(model, max_iter=50, tol=1e-11)
    np.testing.assert_allclose(ac.solve(), V_ref, atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_acpf_gpu_distributed_slack_matches(solver_atol):
    """AcPfGPU via the bridge matches lightsim2grid on a genuine multi-slack
    grid (N=2): non-ref slack theta column + slack_absorbed column."""
    from gpusim2grid import AcPfGPU

    grid = _solved_multislack_grid()
    assert grid.get_slack_ids_solver().shape[0] >= 2, "expected >= 2 slacks"

    V_ref = grid.get_V_solver()
    ac = AcPfGPU(grid, max_iter=40, tol=1e-10)
    V_gpu = ac.solve()

    # bare would be n_pv + 2*n_pq; the augmented system adds the slack columns.
    n_pv = grid.get_pv().shape[0]
    n_pq = grid.get_pq().shape[0]
    assert ac.session.dim_J > n_pv + 2 * n_pq

    np.testing.assert_allclose(V_gpu, V_ref, atol=10 * solver_atol)

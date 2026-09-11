# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""handle_disconnected_grid + a stranded lone remote voltage controller.

The GPU counterpart of lightsim2grid PR #192: in ``handle_disconnected_grid``
mode, a contingency that islands the OWN bus of a remote voltage controller
(a remotely-regulating generator or a voltage-mode SVC, the ``VoltageControl``
extension) while its regulated bus stays live used to be NaN-skipped. For a
control group with exactly one controller, its bordered voltage row is now
repurposed by value into ``Q_c == 0`` (a structural zero reserved once in the
J skeleton), and the regulated bus floats as an ordinary PQ bus -- what a
rebuilt topology without that controller gives.

Oracle: a one-off ``ac_pf`` on a fresh grid with the branch AND the stranded
controller actually deactivated (the reference lightsim2grid's own C++ test
of the feature uses). Islanding the REGULATED bus itself, or every controller
of a group, stays a skip (NaN row).

case14 geometry: bus 7 is a leaf hanging off bus 6 through trafo 3; gen 3
sits on bus 7. Tripping that trafo (branch id n_lines + 3) strands bus 7.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="the augmented Jacobian needs the lightsim2grid C++ bridge")

GEN_REMOTE = 3      # on bus 7 ...
REG_BUS = 9         # ... regulating bus 9, remotely
LEAF_BUS = 7
TRAFO_BEHIND = 3    # buses 6-7: the controller's own bus is behind it
MAX_IT, TOL = 40, 1e-11


def _case14(setup):
    """A solved case14 after `setup(net, model)` mutated it. Returns (model, V)."""
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        model = init_from_pandapower(net)
    setup(net, model)
    model.tell_solver_need_reset()
    model.change_algorithm(AlgorithmType.NR_KLU)
    V = model.ac_pf(np.ones(net.bus.shape[0], dtype=complex), MAX_IT, TOL)
    if V.shape[0] == 0:
        model.tell_solver_need_reset()
        model.change_algorithm(AlgorithmType.NRSing_SparseLU)
        V = model.ac_pf(np.ones(net.bus.shape[0], dtype=complex), MAX_IT, TOL)
    assert V.shape[0] > 0, "lightsim2grid diverged"
    return model, V


def _remote_gen(net, model):
    model.set_gen_regulated_bus(GEN_REMOTE, REG_BUS)


def _svc_on_leaf(net, model):
    # bus 7 must be PQ for an SVC to sit there: drop its local generator first
    model.deactivate_gen(GEN_REMOTE)
    model.init_svcs([1], np.array([1.03]), np.array([0.0]), np.array([0.0]),
                    np.array([-100.0]), np.array([100.0]),
                    np.array([REG_BUS], dtype=np.int32),
                    np.array([LEAF_BUS], dtype=np.int32))


def _branch_id(model, trafo_id):
    return len(model.get_lines()) + trafo_id


def _reference(setup, trafo_off, gens_off=(), svcs_off=()):
    """One-off solve with the trafo and the stranded controller(s) removed."""
    def both(net, model):
        setup(net, model)
        model.deactivate_trafo(int(trafo_off))
        for g in gens_off:
            model.deactivate_gen(int(g))
        for s in svcs_off:
            model.deactivate_svc(int(s))
    _, V = _case14(both)
    return V


def _gpu_ca(model, branch, **kw):
    from gpusim2grid import ContingencyAnalysisGPU
    g = ContingencyAnalysisGPU(model, handle_disconnected_grid=True, nb_iter=15,
                               tol_base=1e-10, **kw)
    g.add_contingencies_by_branch_id([[int(branch)]])
    g.compute(batch_size=8)
    n_bus = model.get_Ybus_solver().shape[0]
    return g.V_results.to_numpy().reshape(1, n_bus)[0], g.last_residuals()[0]


def _gpu_ss(model, branch, **kw):
    from gpusim2grid import ScenarioSweepGPU
    sw = ScenarioSweepGPU(model, handle_disconnected_grid=True, nb_iter=15,
                          tol_base=1e-10, **kw)
    sn = model.get_sn_mva()
    S = model.get_Sbus_solver()
    sw.set_injections((S.real * sn)[None, :], (S.imag * sn)[None, :], sn)
    sw.set_topology([[int(branch)]])
    sw.compute(batch_size=4)
    n_bus = model.get_Ybus_solver().shape[0]
    return (sw.solver.V_results.to_numpy().reshape(1, n_bus)[0],
            sw.last_residuals()[0], sw.get_disconnected()[0])


@requires_gpu
@needs_bridge
@pytest.mark.parametrize("path", ["contingency_analysis", "scenario_sweep"])
def test_stranded_lone_remote_gen_is_recovered(solver_atol, path):
    """Tripping the trafo behind gen 3's own bus: the row converges, bus 7 is
    NaN, and bus 9 (regulated) floats to the one-off solution without gen 3."""
    model, V0 = _case14(_remote_gen)
    n_bus = model.get_Ybus_solver().shape[0]
    branch = _branch_id(model, TRAFO_BEHIND)
    ref = _reference(_remote_gen, TRAFO_BEHIND, gens_off=[GEN_REMOTE])
    vset = float(model.get_generators()[GEN_REMOTE].target_vm_pu)
    assert abs(abs(ref[REG_BUS]) - vset) > 1e-4   # the fallback really frees bus 9

    if path == "contingency_analysis":
        V, res = _gpu_ca(model, branch)
    else:
        V, res, disc = _gpu_ss(model, branch)
        assert disc == 0
    assert np.isfinite(res) and res < 100 * solver_atol
    assert np.isnan(V[LEAF_BUS])
    main = np.ones(n_bus, dtype=bool)
    main[LEAF_BUS] = False
    assert np.all(np.isfinite(V[main]))
    np.testing.assert_allclose(V[main], ref[main], atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_stranded_svc_is_recovered(solver_atol):
    """A voltage-mode SVC (whose (v_row, q_col) slot always exists) on the leaf."""
    model, V0 = _case14(_svc_on_leaf)
    if not hasattr(model, "init_svcs"):
        pytest.skip("this lightsim2grid build has no init_svcs")
    n_bus = model.get_Ybus_solver().shape[0]
    branch = _branch_id(model, TRAFO_BEHIND)
    ref = _reference(_svc_on_leaf, TRAFO_BEHIND, svcs_off=[0])
    V, res = _gpu_ca(model, branch)
    assert np.isfinite(res) and res < 100 * solver_atol
    assert np.isnan(V[LEAF_BUS])
    main = np.ones(n_bus, dtype=bool)
    main[LEAF_BUS] = False
    np.testing.assert_allclose(V[main], ref[main], atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_control_path_trafo_that_islands_nothing_is_unchanged(solver_atol):
    """Tripping a trafo on the control path that keeps the grid connected is an
    ordinary contingency: the regulated bus stays at its setpoint."""
    model, V0 = _case14(_remote_gen)
    branch = _branch_id(model, 4)   # buses 6-8, the grid stays connected
    vset = float(model.get_generators()[GEN_REMOTE].target_vm_pu)
    V, res = _gpu_ca(model, branch)
    assert np.isfinite(res) and res < 100 * solver_atol
    assert np.all(np.isfinite(V))
    assert abs(abs(V[REG_BUS]) - vset) < 10 * solver_atol
    ref = _reference(_remote_gen, 4)
    np.testing.assert_allclose(V, ref, atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_stranding_the_regulated_bus_is_still_skipped():
    """Gen 1 (bus 2) regulates bus 7 remotely; tripping the trafo behind bus 7
    strands the REGULATED bus: no value-only fallback, the row stays NaN."""
    def setup(net, model):
        model.deactivate_gen(GEN_REMOTE)          # bus 7 becomes a plain PQ leaf
        model.set_gen_regulated_bus(1, LEAF_BUS)  # gen 1 (bus 2) now holds it
    model, V0 = _case14(setup)
    branch = _branch_id(model, TRAFO_BEHIND)
    V, res = _gpu_ca(model, branch)
    assert np.isnan(res)
    assert np.all(np.isnan(V))


@requires_gpu
@needs_bridge
def test_shared_group_with_one_member_stranded(solver_atol):
    """Two controllers on different buses regulate bus 9; stranding one of
    them is not a skip: the other keeps the regulated bus at the setpoint and
    the live buses match the one-off solve without the stranded machine."""
    pp = pytest.importorskip("pandapower")

    def setup(net, model):
        model.set_gen_regulated_bus(GEN_REMOTE, REG_BUS)   # gen 3 (bus 7)
        model.set_gen_regulated_bus(4, REG_BUS)            # gen 4 (bus 8, added below)

    def make(extra):
        import pandapower.networks as pn
        from lightsim2grid.gridmodel import init_from_pandapower
        from lightsim2grid.lightsim2grid_cpp import AlgorithmType
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = pn.case14()
            pp.create_gen(net, bus=8, p_mw=0.0, vm_pu=float(net.gen.vm_pu.iloc[GEN_REMOTE]),
                          controllable=True, min_q_mvar=-50., max_q_mvar=50.)
            net.gen.loc[3, ["min_q_mvar", "max_q_mvar"]] = [-50., 50.]
            model = init_from_pandapower(net)
        setup(net, model)
        extra(model)
        model.tell_solver_need_reset()
        model.change_algorithm(AlgorithmType.NR_KLU)
        V = model.ac_pf(np.ones(net.bus.shape[0], dtype=complex), MAX_IT, TOL)
        if V.shape[0] == 0:
            model.tell_solver_need_reset()
            model.change_algorithm(AlgorithmType.NRSing_SparseLU)
            V = model.ac_pf(np.ones(net.bus.shape[0], dtype=complex), MAX_IT, TOL)
        assert V.shape[0] > 0
        return model, V

    model, V0 = make(lambda m: None)
    assert model.get_controller_q_col_solver().shape[0] == 2
    n_bus = model.get_Ybus_solver().shape[0]
    branch = _branch_id(model, TRAFO_BEHIND)
    vset = float(model.get_generators()[GEN_REMOTE].target_vm_pu)

    def ref_setup(m):
        m.deactivate_trafo(int(TRAFO_BEHIND))
        m.deactivate_gen(int(GEN_REMOTE))
    _, ref = make(ref_setup)

    V, res = _gpu_ca(model, branch)
    assert np.isfinite(res) and res < 100 * solver_atol
    assert np.isnan(V[LEAF_BUS])
    main = np.ones(n_bus, dtype=bool)
    main[LEAF_BUS] = False
    assert abs(abs(V[REG_BUS]) - vset) < 10 * solver_atol
    np.testing.assert_allclose(V[main], ref[main], atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_flag_off_still_skips():
    model, V0 = _case14(_remote_gen)
    from gpusim2grid import ContingencyAnalysisGPU
    g = ContingencyAnalysisGPU(model, handle_disconnected_grid=False, nb_iter=15,
                               tol_base=1e-10)
    g.add_contingencies_by_branch_id([[int(_branch_id(model, TRAFO_BEHIND))]])
    g.compute(batch_size=8)
    assert np.isnan(g.last_residuals()[0])

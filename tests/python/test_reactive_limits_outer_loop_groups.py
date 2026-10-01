# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""reactive_limits_outer_loop on VoltageControl groups: a group of generators
every controller bus of which reports LOW_Q / HIGH_Q (in one direction) is
re-solved with each controller held at its own limit, the bus it regulated
floating (``ScenarioSweepSession::set_vc_controller_pins``).

The reference is lightsim2grid, as in test_reactive_limits_outer_loop.py: the
grid rebuilt with the switches applied by hand (a pinned controller PQ at its
limit, still regulating the same remote bus on paper and flagged ``can_be_pv``
so that lightsim2grid reports its release) and solved one-off.

Two grids, both 138 kV with generators behind a short step-up line regulating
the HV bus (the French case: every saturated group of the RTE snapshots is a
remote regulator):
- GROUPS: a one-bus group (gen 1 on leaf 6, regulating bus 2) and a two-bus
  group (gens 2 and 3 on leaves 7 and 8, regulating bus 4) whose reactive ranges
  are proportional, so that they saturate together;
- MIXED: a one-bus group next to a plain PV bus and a frozen machine flagged
  ``can_be_pv``, so that group pins, plain switches and releases share rows.
"""

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid import _gpusim2grid as _cpp

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_gen_pv_release", False),
                       reason="needs the bridge built against a lightsim2grid with can_be_pv"),
]

VN_KV = 138.
TOL_MVA, TOL_VM = 1e-3, 1e-4
LOW_Q, HIGH_Q, REL_LOW, REL_HIGH = 5, 6, 9, 10

# (bus, p_mw, vm_pu, q_mvar, voltage_regulator_on, min_q, max_q, regulated bus)
GROUPS = dict(
    n_bus=9,
    lines=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4), (0, 3), (2, 6), (4, 7), (4, 8)],
    step_up={8, 9, 10},
    loads=[(2, 60., 30.), (3, 70., 30.), (4, 60., 25.), (5, 50., 20.)],
    gens=[(0, 0., 1.03, 0., True, -500., 500., 0),
          (6, 40., 1.02, 0., True, -10., 35., 2),
          (7, 30., 1.02, 0., True, -10., 30., 4),
          (8, 20., 1.02, 0., True, -5., 15., 4)],
    can_be_pv=(False, False, False, False),
)
MIXED = dict(
    n_bus=9,
    lines=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4), (2, 6), (6, 7), (7, 3), (0, 6), (5, 7), (2, 8)],
    step_up={12},
    loads=[(3, 60., 25.), (4, 50., 20.), (5, 40., 15.), (6, 30., 10.), (7, 45., 18.)],
    gens=[(0, 0., 1.03, 0., True, -500., 500., 0),
          (8, 60., 1.041, 0., True, 30., 56., 2),    # a one-bus group regulating bus 2
          (4, 20., 1.005, -8., False, -8., 30., 4),  # frozen PQ at min_q, flagged
          (7, 30., 1.02, 0., True, 15., 30., 7)],    # plain PV
    can_be_pv=(False, False, True, False),
)


def _build(spec, gens=None, can_be_pv=None):
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    gens = spec["gens"] if gens is None else gens
    can_be_pv = spec["can_be_pv"] if can_be_pv is None else can_be_pv
    lines, n_bus = spec["lines"], spec["n_bus"]
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(n_bus, 1, np.full(n_bus, VN_KV), 0, 0)
    x = np.array([0.03 if k in spec["step_up"] else 0.08 for k in range(len(lines))])
    grid.init_powerlines(np.full(len(lines), 0.01), x, np.full(len(lines), 0.02j),
                         np.array([a for a, _ in lines]), np.array([b for _, b in lines]))
    loads = spec["loads"]
    grid.init_loads(np.array([l[1] for l in loads]), np.array([l[2] for l in loads]),
                    np.array([l[0] for l in loads]))
    grid.init_generators_full(np.array([g[1] for g in gens]), np.array([g[2] for g in gens]),
                              np.array([g[3] for g in gens]), [g[4] for g in gens],
                              np.array([g[5] for g in gens]), np.array([g[6] for g in gens]),
                              np.array([g[0] for g in gens]))
    for k, g in enumerate(gens):
        if g[7] != g[0]:
            grid.set_gen_regulated_bus(k, g[7])
    grid.set_gen_can_be_pv(np.array(can_be_pv))
    grid.add_gen_slackbus(0, 1.)
    grid.tell_solver_need_reset()
    V = grid.ac_pf(np.full(n_bus, 1.0 + 0j), 50, 1e-12)
    assert V.shape[0] > 0
    return grid, V


def _ca(grid, n_lines, outer_loop=True):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True,
                                reactive_limits_outer_loop=outer_loop)
    ca.physical_violation_tol_mva = TOL_MVA
    ca.physical_violation_tol_vm_pu = TOL_VM
    ca.add_contingencies_by_branch_id([[i] for i in range(n_lines)])
    ca.compute(batch_size=16)
    return ca


def _switched(spec, sw, gens=None):
    """The grid with one contingency's switches applied by hand (see the module doc)."""
    gens = [list(g) for g in (spec["gens"] if gens is None else gens)]
    can_be_pv = list(spec["can_be_pv"])
    for b, (_q, at_min) in sw["to_pq"].items():
        for k, g in enumerate(gens):
            if g[0] == b and g[4] and g[7] == b:
                g[4], g[3], can_be_pv[k] = False, (g[5] if at_min else g[6]), True
    for _grp, (at_min, gen_ids) in sw["vc_pin"].items():
        for k in gen_ids:
            g = gens[k]
            g[4], g[3], can_be_pv[k] = False, (g[5] if at_min else g[6]), True
    for _b, (gen_ids, vm) in sw["to_pv"].items():
        for k in gen_ids:
            gens[k][4], gens[k][2], can_be_pv[k] = True, vm, False
    return _build(spec, [tuple(g) for g in gens], tuple(can_be_pv))[0]


def _records(viols):
    return sorted((int(v.violation_type), int(v.element_id), round(float(v.value), 6))
                  for v in viols)


@pytest.fixture(scope="module", params=["groups", "mixed"])
def run(request):
    spec = GROUPS if request.param == "groups" else MIXED
    grid, V0 = _build(spec)
    return request.param, spec, V0, _ca(grid, len(spec["lines"]))


def test_second_pass_is_lightsim2grids_switched_solve(run, solver_atol):
    """Every re-solved contingency -- group pins alone or together with plain
    switches and releases -- gives lightsim2grid's voltages and physical
    records for the grid with the same switches applied."""
    from gpusim2grid import ReactiveLimitsStatus as S
    name, spec, V0, ca = run
    n = len(spec["lines"])
    st, sw = ca.get_outer_loop_status(), ca.get_outer_loop_switches()
    V = ca.V_results.to_numpy().reshape(n, -1)
    phys = ca.get_physical_violations()
    rows = [i for i in range(n) if st[i] == S.RECOMPUTED and sw[i]["vc_pin"]]
    assert rows, f"{name}: no contingency pinned a group"
    for i in [i for i in range(n) if st[i] == S.RECOMPUTED]:
        ref = _switched(spec, sw[i])
        ref.deactivate_powerline(i)
        V_ref = ref.ac_pf(V0.copy(), 50, 1e-12)
        assert V_ref.shape[0] > 0
        np.testing.assert_allclose(V[i], V_ref, atol=solver_atol, rtol=0)
        assert _records(phys[i]) == _records(ref.get_physical_violations(True, TOL_MVA, TOL_VM)), \
            f"{name}, contingency {i}"


def test_two_bus_group_and_cascade():
    """GROUPS: tripping line 5-0 saturates both controller buses of the two-bus
    group (pinned together); tripping line 0-3 saturates the one-bus group, whose
    pin then pushes the two-bus group past its range -- reported, for a next
    outer iteration."""
    from gpusim2grid import ReactiveLimitsStatus as S
    grid, _ = _build(GROUPS)
    ca = _ca(grid, len(GROUPS["lines"]))
    st, sw = ca.get_outer_loop_status(), ca.get_outer_loop_switches()
    assert st[5] == S.RECOMPUTED
    assert sorted(gens for _at_min, gens in sw[5]["vc_pin"].values()) == [(2, 3)]
    assert all(not at_min for at_min, _ in sw[5]["vc_pin"].values())
    assert st[7] == S.RECOMPUTED and [g for _a, g in sw[7]["vc_pin"].values()] == [(1,)]
    assert [(t, b) for t, b, _ in _records(ca.get_physical_violations()[7])] == [(HIGH_Q, 7), (HIGH_Q, 8)]


def test_group_switch_back_is_reported():
    """MIXED, line 2-3: the group is held at its min_q while the plain PV bus 7 is
    held at its max_q; its regulated bus 2 then sits below the group's target --
    the group would regulate again, reported on its generator against bus 2."""
    grid, _ = _build(MIXED)
    ca = _ca(grid, len(MIXED["lines"]))
    sw = ca.get_outer_loop_switches()[2]
    assert [v for v in sw["vc_pin"].values()] == [(True, (1,))]
    assert list(sw["to_pq"]) == [7]
    rel = [v for v in ca.get_physical_violations()[2]
           if int(v.violation_type) == REL_LOW and int(v.element_id) == 1]
    assert len(rel) == 1
    assert rel[0].limit == pytest.approx(1.041 * VN_KV)
    assert rel[0].value < rel[0].limit


def test_partial_group_is_unsupported(solver_atol):
    """GROUPS with the second controller's max_q raised to 22 MVAr: on line 5-0
    only one of the two controller buses saturates -- not handled (the other
    controller would have to take the group's voltage alone), first pass kept."""
    from gpusim2grid import ReactiveLimitsStatus as S
    gens = [list(g) for g in GROUPS["gens"]]
    gens[3][6] = 22.
    gens = [tuple(g) for g in gens]
    grid, _ = _build(GROUPS, gens)
    n = len(GROUPS["lines"])
    ca_on, ca_off = _ca(grid, n), _ca(grid, n, outer_loop=False)
    assert ca_on.get_outer_loop_status()[5] == S.UNSUPPORTED
    assert [(t, b) for t, b, _ in _records(ca_off.get_physical_violations()[5])] == [(HIGH_Q, 7)]
    np.testing.assert_allclose(ca_on.V_results.to_numpy().reshape(n, -1)[5],
                               ca_off.V_results.to_numpy().reshape(n, -1)[5], atol=solver_atol, rtol=0)
    assert _records(ca_on.get_physical_violations()[5]) == _records(ca_off.get_physical_violations()[5])


def test_session_refuses_a_partial_group():
    """A row holding a group's first controller must hold all of them (their
    sharing rows refer to it): checked when the row is solved."""
    from gpusim2grid import ScenarioSweepGPU
    grid, _ = _build(GROUPS)
    sweep = ScenarioSweepGPU(grid, nb_iter=10)
    sess = sweep.solver._s
    groups = np.asarray(sess.vc_ctrl_group)
    two = [j for j in range(groups.size) if np.sum(groups == groups[j]) == 2]
    first = min(two)
    sess.set_vc_controller_pins([[(first, 0.1)]])
    sweep.set_injections_from_elements(
        np.array([[l.target_p_mw for l in grid.get_loads()]]),
        np.array([[l.target_q_mvar for l in grid.get_loads()]]),
        np.array([[g.target_p_mw for g in grid.get_generators()]]))
    with pytest.raises(RuntimeError, match="either all of them are held"):
        sweep.compute(batch_size=1)

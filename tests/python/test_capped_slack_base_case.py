# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A grid whose base case lightsim2grid capped (``LSGrid.cap_slack_at_active_limits``:
OpenLoadFlow's distributed-slack rule, a unit the base solve pushes beyond an
active limit leaves the slack at that limit, flagged ``can_participate_slack``,
and the grid is re-solved). gpusim2grid reads the capped units through the
bridge as pre-pass-only units: out of the Newton slack, still moved by the
``redistribute_slack`` pre-pass within their range.

The reference is lightsim2grid's own contingency analysis of that grid
(``redistribute_slack`` on): the GPU must give its voltages and slack
active-power records, and a clean base-case report. The grid: pandapower case14
with the slack shared by the ext_grid and generators 1 and 2, generator 1's
``max_p`` set 5 MW below the output the shared slack gives it.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge
from test_bus_q_violations import MAX_IT, TOL, _solve, _all_n1
from test_gen_p_violations import _ac_pf_res_p, _p_records, _assert_same, _mw_atol

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not have_ls2g_bridge, reason="the slack data come through the C++ bridge"),
]

GEN, HIGH_P, LOW_P = 5, 7, 8


def _case14_over_max_p(below_mw=5.):
    pn = pytest.importorskip("pandapower.networks")
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(pn.case14())
    if not hasattr(grid, "cap_slack_at_active_limits"):
        pytest.skip("this lightsim2grid has no LSGrid.cap_slack_at_active_limits")
    grid.add_gen_slackbus(1, 2.)
    grid.add_gen_slackbus(2, 1.)
    grid.tell_solver_need_reset()
    gens_p, _ = _ac_pf_res_p(grid)
    n_gen = len(grid.get_generators())
    pmin, pmax = np.full(n_gen, -1e4), np.full(n_gen, 1e4)
    pmax[1] = gens_p[1] - below_mw
    grid.set_gen_p_limits(pmin, pmax)
    grid.tell_solver_need_reset()
    v0 = _solve(grid)
    return grid, v0, float(pmax[1])


def _ls_ca(grid, v0, ctgs):
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    ca = ContingencyAnalysisCPP(grid)
    for c in ctgs:
        ca.add_n1(int(c[0]))
    ca.redistribute_slack = True
    ca.compute_physical_violations = True
    ca.physical_violation_tol_mva = 0.
    ca.compute(v0.copy(), MAX_IT, TOL)
    order = [[sorted(int(x) for x in d) for d in ca.my_defaults()].index(sorted(c)) for c in ctgs]
    V = np.asarray(ca.get_voltages())
    rows = ca.get_physical_violations()
    return V[order], [rows[i] for i in order]


def _gpu_ca(grid, ctgs):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, tol_base=1e-10, compute_physical_violations=True,
                                redistribute_slack=True)
    ca.physical_violation_tol_mva = 0.
    ca.add_contingencies_by_branch_id(ctgs)
    ca.compute(batch_size=8)
    return ca


def _slack_p_n(ca):
    return [(int(v.element_id), int(v.violation_type)) for v in ca.get_physical_violations_n()
            if int(v.violation_type) in (HIGH_P, LOW_P)]


def test_capped_grid_is_lightsim2grids(solver_atol):
    grid, v0, pmax = _case14_over_max_p()
    ctgs = _all_n1(grid)
    # as handed over: generator 1 over its max_p in the base case
    assert _slack_p_n(_gpu_ca(grid, ctgs[:2])) == [(1, HIGH_P)]

    capped = grid.cap_slack_at_active_limits(MAX_IT, TOL)
    assert [(int(v.element_id), float(v.limit)) for v in capped] == [(1, pmax)]
    ca = _gpu_ca(grid, ctgs)
    assert _slack_p_n(ca) == []
    # every contingency: lightsim2grid's own analysis of the capped grid. Generator 1
    # held the angle reference: the reference moved to another participant, whose angle
    # each side keeps where its own start put it -- compared up to that constant.
    V_ref, rows_ref = _ls_ca(grid, v0, ctgs)
    V = ca.V_results.to_numpy().reshape(len(ctgs), -1)
    ok = np.isfinite(V).all(axis=1) & (np.abs(V_ref) > 0).all(axis=1)
    assert ok.sum() > len(ctgs) // 2

    def _rot(X):
        return X * np.exp(-1j * np.angle(X[:, :1]))
    np.testing.assert_allclose(_rot(V[ok]), _rot(V_ref[ok]), atol=solver_atol, rtol=0)
    _assert_same([r for r, k in zip(_p_records(rows_ref), ok) if k],
                 [r for r, k in zip(_p_records(ca.get_physical_violations()), ok) if k],
                 _mw_atol(grid, solver_atol))


def test_capped_unit_is_a_pre_pass_unit():
    """The bridge hands the capped unit over outside the solve's slack but in the
    redistribution pre-pass (``in_slack`` 0)."""
    grid, _, _ = _case14_over_max_p()
    grid.cap_slack_at_active_limits(MAX_IT, TOL)
    g1 = grid.get_generators()[1]
    assert not g1.is_slack and g1.can_participate_slack
    from gpusim2grid import _gpusim2grid as _cpp
    rd = _cpp._extract_slack_redistribution_data_from_lsgrid(grid, int(grid.get_V_solver().shape[0]))
    units = {(int(k), int(e)): int(s) for k, e, s in zip(rd.kind, rd.el_id, rd.in_slack)}
    assert units[(GEN, 1)] == 0
    assert all(s == 1 for (k, e), s in units.items() if (k, e) != (GEN, 1))

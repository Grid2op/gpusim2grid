# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Prototype: ``reordering_alg='ls2g_constrained'`` hands cuDSS lightsim2grid's
constrained order of the batch Jacobian (``ls2g::constrained_order`` on the
grid's own ``OrderingConstraints``) through ``CUDSS_DATA_USER_PERM``
(docs/dev_notes/batched_lu_and_constrained_ordering.md).

``'ls2g_constrained_nd'`` post-moves cuDSS's own (nested dissection) order
with the same constraints instead of AMD.

Pinned here: cuDSS uses exactly that order (its reordering read back equals
it), the batch converges to the same states as with cuDSS's own order (another
elimination order changes the rounding only), and a ledger whose indices moved
is refused rather than ordered wrongly.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
import gpusim2grid._gpusim2grid as _cpp

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_loops", False),
                       reason="gpusim2grid built against a lightsim2grid without NROuter_*"),
    pytest.mark.skipif(bool(_cpp.is_fp32), reason="parity with lightsim2grid's double-precision NR"),
]

MAX_IT = 30
TOL_MVA = 1e-8


def _case14_remote_group():
    """case14, its generators short of reactive power, the bus 8 generator regulating
    bus 7 remotely: a VoltageControl group, whose rows the outer loops switch by value
    (the case lightsim2grid declares ordering constraints for)"""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    net = pn.case14()
    net.gen["min_q_mvar"] = -15.
    net.gen["max_q_mvar"] = 15.
    net.gen.loc[net.gen.bus == 7, ["min_q_mvar", "max_q_mvar"]] = (-300., 300.)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    nb_gen = len(grid.get_generators())
    grid.set_gen_p_limits(np.zeros(nb_gen), np.full(nb_gen, 300.))
    grid.set_gen_can_participate_slack([True] * nb_gen, np.full(nb_gen, 300.))
    gen_id = next(g.id for g in grid.get_generators() if g.bus_id == 7)
    grid.set_gen_regulated_bus(gen_id, 6)
    grid.change_algorithm("NROuter_KLU")
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    return grid


def _outer_ca(grid, reordering_alg):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=False,
                                tol=TOL_MVA / grid.get_sn_mva(), reordering_alg=reordering_alg)
    lines = [i for i, line in enumerate(grid.get_lines()) if line.connected_global]
    ca.add_contingencies_by_branch_id([[c] for c in lines])
    ca.compute(batch_size=4)
    return ca, len(lines)


@pytest.mark.parametrize("alg", ["ls2g_constrained", "ls2g_constrained_nd"])
def test_cudss_uses_lightsim2grids_order_and_converges_alike(alg, solver_atol):
    grid = _case14_remote_group()
    ref, n = _outer_ca(grid, "default")
    ca, _ = _outer_ca(grid, alg)
    assert ca.timings.cudss_user_perm_honoured == 1
    assert ref.timings.cudss_user_perm_honoured == -1
    status = np.asarray(ca.get_row_status())
    np.testing.assert_array_equal(status, np.asarray(ref.get_row_status()))
    assert (status == 0).any()
    ok = status == 0
    V = ca.V_results.to_numpy().reshape(n, -1)[ok]
    V_ref = ref.V_results.to_numpy().reshape(n, -1)[ok]
    np.testing.assert_allclose(V, V_ref, atol=solver_atol, rtol=0)
    assert ([r.nb_outer_iterations for r in ca.get_outer_results()]
            == [r.nb_outer_iterations for r in ref.get_outer_results()])


@pytest.mark.parametrize("alg", ["ls2g_constrained", "ls2g_constrained_nd"])
def test_renumbered_ledger_is_refused(alg):
    """use_distributed_slack=False deletes J indices: lightsim2grid's constraints no
    longer name the right unknowns, so the order is refused, not computed wrongly"""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from gpusim2grid import ContingencyAnalysisGPU
    net = pn.case14()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    nb_gen = len(grid.get_generators())
    grid.set_gen_can_participate_slack([True] * nb_gen, np.full(nb_gen, 300.))
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    with pytest.raises(RuntimeError, match="ordering constraints"):
        ContingencyAnalysisGPU(grid, use_distributed_slack=False, reordering_alg=alg)

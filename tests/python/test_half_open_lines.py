# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Half-open branches (connected on one side only) through the contingency path.

A branch that pypowsybl reports as connected on a single terminal is imported by
``init_from_pypowsybl(keep_half_open_lines=True)`` as *half-open*: the energized
side stays in the model and the open end is Kron-reduced out. lightsim2grid then
exposes the reduced π-model through ``yac_eff_*`` (the raw ``yac_*`` still hold the
full two-port admittances) and relabels the open end to bus id ``-1``, which has no
row/column in ``Ybus_solver``.

gpusim2grid must (a) patch Ybus with the *effective* admittances when such a
branch is tripped -- subtracting the raw ``yac_22`` at the energized bus would
remove an admittance that was never there -- and (b) skip the Ybus triplets and
branch-flow terms that touch the ``-1`` endpoint. The reference is lightsim2grid's
own ``ContingencyAnalysisCPP`` on the same grid.

``test_import_is_half_open`` guards the premise of the other tests: if lightsim2grid
ever stopped importing the branch as half-open, the parity tests below would pass
for the wrong reason.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu

pytest.importorskip("pypowsybl", reason="pypowsybl not installed")

# IEEE 118 (pypowsybl's bundled case): one line opened on side 1, one
# transformer opened on side 2. Same elements as lightsim2grid's own
# TestImportHalfOpen so the import semantics are known-good upstream.
LINE_ID = "L6-7-1"
TRAFO_ID = "T8-5-1"
GEN_SLACK_ID = 29


@pytest.fixture(scope="module")
def half_open_case():
    """IEEE 118 from pypowsybl with one half-open line and one half-open trafo,
    imported into lightsim2grid with keep_half_open_lines=True and AC-solved.

    Returns a dict with the solved grid, its pypowsybl network, the
    lines-then-trafos branch index of both half-open branches, and the
    branch/bus counts.
    """
    import pypowsybl.network as ppn
    from lightsim2grid.network import init_from_pypowsybl
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = ppn.create_ieee118()
        net.update_lines(id=LINE_ID, connected1=False)
        net.update_2_windings_transformers(id=TRAFO_ID, connected2=False)
        grid = init_from_pypowsybl(net, gen_slack_id=GEN_SLACK_ID,
                                   sort_index=False, keep_half_open_lines=True)
        grid.change_algorithm(AlgorithmType.NR_KLU)
        n_bus = grid.total_bus()
        V_base = grid.ac_pf(np.ones(n_bus, dtype=complex), 30, 1e-10)
    assert V_base.shape[0] == n_bus, "base case diverged"

    n_lines = len(grid.get_lines())
    n_trafos = len(grid.get_trafos())
    line_pos = list(net.get_lines().index).index(LINE_ID)
    trafo_pos = list(net.get_2_windings_transformers().index).index(TRAFO_ID)
    return {
        "grid": grid,
        "net": net,
        "n_bus": n_bus,
        "n_lines": n_lines,
        "n_trafos": n_trafos,
        "n_branches": n_lines + n_trafos,
        "line_pos": line_pos,                   # index into grid.get_lines()
        "trafo_pos": trafo_pos,                 # index into grid.get_trafos()
        "line_branch": line_pos,                # lines-then-trafos branch id
        "trafo_branch": n_lines + trafo_pos,    # lines-then-trafos branch id
        "V_base": V_base,
    }


def test_import_is_half_open(half_open_case):
    """Premise check: lightsim2grid really models both branches as half-open.

    The open end is Kron-reduced: every effective admittance involving it is
    zero, the surviving self-admittance is the reduced one (not the raw
    ``yac_*`` value), and the open endpoint carries bus id -1.
    """
    c = half_open_case
    line = c["grid"].get_lines()[c["line_pos"]]
    trafo = c["grid"].get_trafos()[c["trafo_pos"]]

    # line: side 1 open, side 2 energized, still globally connected
    assert not line.connected1
    assert line.connected2
    assert line.connected_global
    assert line.bus1_id == -1
    assert line.bus2_id >= 0
    assert line.yac_eff_11 == 0 and line.yac_eff_12 == 0 and line.yac_eff_21 == 0
    kron_22 = line.yac_22 - line.yac_21 * line.yac_12 / line.yac_11
    assert np.isclose(line.yac_eff_22, kron_22)
    # the effective value genuinely differs from the raw one (line charging
    # only survives) -- otherwise these tests could not tell eff from raw
    assert not np.isclose(line.yac_eff_22, line.yac_22)

    # transformer: side 1 energized, side 2 open
    assert trafo.connected1
    assert not trafo.connected2
    assert trafo.connected_global
    assert trafo.bus1_id >= 0
    assert trafo.bus2_id == -1
    assert trafo.yac_eff_22 == 0 and trafo.yac_eff_12 == 0 and trafo.yac_eff_21 == 0
    kron_11 = trafo.yac_11 - trafo.yac_21 * trafo.yac_12 / trafo.yac_22
    assert np.isclose(trafo.yac_eff_11, kron_11)
    assert not np.isclose(trafo.yac_eff_11, trafo.yac_11)


def _cpu_reference(c):
    """ContingencyAnalysisCPP N-1 on every branch: (V, flows_kA, solved_mask)."""
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    ca = ContingencyAnalysisCPP(c["grid"])
    ca.add_all_n1()
    ca.compute(np.ones(c["n_bus"], dtype=complex), 30, 1e-10)
    ca.compute_flows()
    V_ref = np.asarray(ca.get_voltages())
    flows_ref_ka = np.asarray(ca.get_flows())
    solved = np.abs(V_ref).sum(axis=1) > 0   # lightsim2grid zeroes a failed row
    return V_ref, flows_ref_ka, solved


def _gpu_n1(c):
    from gpusim2grid import ContingencyAnalysisGPU

    n_br = c["n_branches"]
    g = ContingencyAnalysisGPU(c["grid"], precision=None, nb_iter=15, tol_base=1e-10)
    g.add_contingencies_by_branch_id([[b] for b in range(n_br)])
    g.compute(batch_size=64)
    g.compute_flows()
    V = g.V_results.to_numpy().reshape(n_br, -1)
    or_amps = g.or_amps.to_numpy().reshape(n_br, n_br)
    ex_amps = g.ex_amps.to_numpy().reshape(n_br, n_br)
    return g, V, or_amps, ex_amps, g.last_residuals()


@pytest.fixture(scope="module")
def n1_results(half_open_case):
    V_ref, flows_ref_ka, solved = _cpu_reference(half_open_case)
    g, V_gpu, or_amps, ex_amps, residuals = _gpu_n1(half_open_case)
    me_to_solver = np.asarray(half_open_case["grid"].id_me_to_ac_solver())
    return {
        "V_ref": V_ref, "flows_ref_ka": flows_ref_ka, "solved": solved,
        "gpu": g, "V_gpu": V_gpu, "or_amps": or_amps, "ex_amps": ex_amps,
        "residuals": residuals, "me_to_solver": me_to_solver,
    }


def test_n1_matches_ls2g(half_open_case, n1_results, solver_atol):
    """Full N-1 (every branch, the two half-open ones included) matches the
    CPU reference bus-by-bus, with the same set of skipped contingencies."""
    c, r = half_open_case, n1_results
    me_to_solver = r["me_to_solver"]

    # No bus is Kron-reduced here (only the branch *ends* are): model and
    # solver numbering coincide and every model bus is in the solver.
    assert r["V_gpu"].shape == (c["n_branches"], c["n_bus"])
    assert np.all(me_to_solver >= 0)

    # Contingencies that split the grid are skipped on both sides.
    gpu_skipped = np.isnan(r["residuals"])
    np.testing.assert_array_equal(gpu_skipped, ~r["solved"])
    assert r["solved"].sum() > 0

    # Every solved contingency converged and matches lightsim2grid.
    solved = r["solved"]
    assert np.all(r["residuals"][solved] < 100 * solver_atol)
    V_gpu_model = r["V_gpu"][:, me_to_solver]
    np.testing.assert_allclose(V_gpu_model[solved], r["V_ref"][solved],
                               atol=10 * solver_atol)


def test_tripping_half_open_branches(half_open_case, n1_results, solver_atol):
    """Tripping a half-open branch itself is solved (not skipped) and matches
    lightsim2grid: only the effective self-admittance at the energized bus is
    removed. For the line that is the charging admittance (a small but
    measurable change to V); for the ideal transformer it is exactly zero."""
    c, r = half_open_case, n1_results
    V_base = c["V_base"][r["me_to_solver"]]

    for b in (c["line_branch"], c["trafo_branch"]):
        assert r["solved"][b]
        assert np.isfinite(r["residuals"][b]) and r["residuals"][b] < 100 * solver_atol
        np.testing.assert_allclose(r["V_gpu"][b][r["me_to_solver"]], r["V_ref"][b],
                                   atol=10 * solver_atol)

    # Removing the line's surviving charging admittance moves V a little ...
    line_delta = np.abs(r["V_gpu"][c["line_branch"]] - V_base).max()
    assert line_delta > 0
    # ... but far less than removing the raw (series) admittance would: had the
    # raw yac_22 been subtracted, bus 2 would have lost ~|yac_22| ≈ 47 pu of
    # admittance instead of ~0.0055 pu and V would move by orders of magnitude
    # more (or the solve would diverge).
    assert line_delta < 1e-2
    # An open-ended ideal transformer has zero effective admittance: no-op trip.
    trafo_delta = np.abs(r["V_gpu"][c["trafo_branch"]] - V_base).max()
    assert trafo_delta < 10 * solver_atol


def test_flows_on_half_open_branches(half_open_case, n1_results):
    """Branch currents: the open terminal carries no current, the energized
    terminal of the line carries only its charging current, and the origin-side
    currents of every branch match lightsim2grid's (kA → A)."""
    c, r = half_open_case, n1_results
    solved = r["solved"]
    or_amps, ex_amps = r["or_amps"], r["ex_amps"]

    # origin-side parity with lightsim2grid on every solved contingency
    np.testing.assert_allclose(or_amps[solved], 1e3 * r["flows_ref_ka"][solved],
                               rtol=1e-6, atol=1e-3)

    # line: side 1 (origin) open → 0 A; side 2 carries the charging current only
    # (in the row where the line itself is tripped both sides are 0 A)
    lb = c["line_branch"]
    assert np.all(or_amps[solved, lb] == 0)
    live = solved.copy()
    live[lb] = False
    assert np.all(ex_amps[live, lb] > 0)
    assert np.all(ex_amps[live, lb] < 100)         # a few amps of charging, not load
    assert ex_amps[lb, lb] == 0

    # transformer: side 2 (extremity) open → 0 A; no shunt → side 1 also 0 A
    tb = c["trafo_branch"]
    assert np.all(ex_amps[solved, tb] == 0)
    assert np.all(or_amps[solved, tb] == 0)

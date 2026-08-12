# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
test_scenario_sweep.py — GPU row-aligned combined topology + injection sweep
(ScenarioSweepGPU / _ScenarioSweepSolver / ScenarioSweepSession).

Row `i` of the injection matrices is solved together with row `i` of the
tripped-branch list, independently of every other row. Tests validate:

  1. Base case reproduction (no injection change, no topology change).
  2. Topology fixed (empty) per row: must match _InjectionSweepSolver exactly.
  3. Injection fixed (base case) per row: must match _ContingencyAnalysisSolver's
     N-1 exactly, including which rows are reported disconnected (NaN).
  4. Row-aligned zip: permuting the rows of the (injection, topology) inputs
     permutes the (finite) results identically.
  5. A row whose topology trip islands the grid reports NaN / is excluded
     without corrupting other rows.
  6. compute_flows(): each row's own tripped branches report exactly 0 A.
  7. All 4 linear-solve strategies converge with finite residuals.
  8. Facade coverage: ScenarioSweepGPU via the lightsim2grid bridge, both
     set_injections and set_injections_from_elements, set_topology optional.
"""
import numpy as np
import pytest

from conftest import requires_gpu, branch_data_arrays


pytestmark = requires_gpu

_RESIDUAL_TOL = 1e-4          # fp32 tolerance, matches the other batch suites
_VOLTAGE_ATOL = 5e-4          # |V_gpu| vs reference
_RESIDUAL_CONVERGED = 1e-4    # threshold used to call a row "converged"


def _make_solver(ieee14_base_case, branch_data=None, batch_size=5, nb_iter=10):
    from gpusim2grid.scenario_sweep import _ScenarioSweepSolver

    d = ieee14_base_case
    solver = _ScenarioSweepSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=batch_size, nb_iter=nb_iter, max_iter_base=10, tol_base=1e-6,
    )
    if branch_data is not None:
        solver.set_branch_data(*branch_data)
    return solver


def _scaled_injections(ieee14_base_case, scales):
    """Return (p_mw, q_mvar, sn_mva) — one row per scale, base Sbus * scale."""
    d = ieee14_base_case
    sn_mva = d["grid"].get_sn_mva()
    n_bus = d["n_bus"]
    n_scen = len(scales)

    p_base_mw = d["Sbus"].real * sn_mva
    q_base_mvar = d["Sbus"].imag * sn_mva

    p_mw = np.zeros((n_scen, n_bus), dtype=np.float64)
    q_mvar = np.zeros((n_scen, n_bus), dtype=np.float64)
    for s, scale in enumerate(scales):
        p_mw[s, :] = p_base_mw * scale
        q_mvar[s, :] = q_base_mvar * scale
    return p_mw, q_mvar, sn_mva


class TestBaseCaseReproduction:
    def test_no_injection_no_topology_change_reproduces_base_case(
        self, ieee14_grid, ieee14_base_case
    ):
        d = ieee14_base_case
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
        n_bus = d["n_bus"]

        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, [1.0, 1.0])
        solver = _make_solver(ieee14_base_case, branch_data, batch_size=2)
        solver.set_injections(p_mw, q_mvar, sn_mva)
        solver.set_topology([[], []])   # nothing tripped for either row
        solver.run()

        V = solver.V_results.to_numpy().reshape(2, n_bus)
        res = solver.residuals.to_numpy()
        assert np.all(np.isfinite(res))
        assert np.abs(res).max() < _RESIDUAL_TOL

        err = np.abs(V - d["v_ref"][None, :])
        assert err.max() < _VOLTAGE_ATOL

    def test_topology_never_set_defaults_to_no_trips(
        self, ieee14_grid, ieee14_base_case
    ):
        """set_topology() is optional; a never-covered row means no trips."""
        d = ieee14_base_case
        branch_data, _, _ = branch_data_arrays(ieee14_grid)
        n_bus = d["n_bus"]

        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, [1.0])
        solver = _make_solver(ieee14_base_case, branch_data, batch_size=1)
        solver.set_injections(p_mw, q_mvar, sn_mva)
        solver.run()

        V = solver.V_results.to_numpy().reshape(1, n_bus)
        err = np.abs(V[0] - d["v_ref"])
        assert err.max() < _VOLTAGE_ATOL


class TestTopologyFixedMatchesInjectionSweep:
    """Every row shares the same (empty) topology: must reduce exactly to
    InjectionSweepGPU's own batch source on the same injections."""

    def test_matches_injection_sweep(self, ieee14_grid, ieee14_base_case):
        from gpusim2grid.injection_sweep import _InjectionSweepSolver

        d = ieee14_base_case
        branch_data, _, _ = branch_data_arrays(ieee14_grid)
        n_bus = d["n_bus"]
        scales = [0.85, 0.95, 1.0, 1.05, 1.15]
        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, scales)
        n_scen = len(scales)

        ss = _make_solver(ieee14_base_case, branch_data, batch_size=n_scen)
        ss.set_injections(p_mw, q_mvar, sn_mva)
        ss.set_topology([[] for _ in scales])
        ss.run()
        V_ss = ss.V_results.to_numpy().reshape(n_scen, n_bus)
        res_ss = ss.residuals.to_numpy()

        inj = _InjectionSweepSolver(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            batch_size=n_scen, nb_iter=10, max_iter_base=10, tol_base=1e-6,
        )
        inj.set_injections(p_mw, q_mvar, sn_mva)
        inj.run()
        V_inj = inj.V_results.to_numpy().reshape(n_scen, n_bus)
        res_inj = inj.residuals.to_numpy()

        assert np.abs(V_ss - V_inj).max() < 1e-5
        assert np.abs(res_ss - res_inj).max() < 1e-5


class TestInjectionFixedMatchesContingencyAnalysis:
    """Every row shares the same (base-case) injection: must reduce exactly
    to ContingencyAnalysisGPU's own N-1 batch source on the same trips,
    including which rows come back disconnected (NaN)."""

    def test_matches_contingency_analysis_n1(self, ieee14_grid, ieee14_base_case):
        from gpusim2grid.contingency_analysis import _ContingencyAnalysisSolver

        d = ieee14_base_case
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
        n_bus = d["n_bus"]
        n_ctg = n_lines + n_trafos
        cont_branch_ids = [[c] for c in range(n_ctg)]

        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, [1.0] * n_ctg)

        ss = _make_solver(ieee14_base_case, branch_data, batch_size=n_ctg)
        ss.set_injections(p_mw, q_mvar, sn_mva)
        ss.set_topology(cont_branch_ids)
        ss.run()
        V_ss = ss.V_results.to_numpy().reshape(n_ctg, n_bus)
        res_ss = ss.residuals.to_numpy()
        disc_ss = ss.get_disconnected()

        ca = _ContingencyAnalysisSolver(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            batch_size=n_ctg, nb_iter=10, max_iter_base=10, tol_base=1e-6,
        )
        ca.set_branch_data(*branch_data)
        ca.build_contingencies(cont_branch_ids)
        ca.run()
        V_ca = ca.V_results.to_numpy().reshape(n_ctg, n_bus)
        res_ca = ca.residuals.to_numpy()

        nan_ss = ~np.isfinite(res_ss)
        nan_ca = ~np.isfinite(res_ca)
        assert np.array_equal(nan_ss, nan_ca), (
            "ScenarioSweepGPU and ContingencyAnalysisGPU disagree on which "
            "N-1s are disconnected"
        )
        assert np.array_equal(disc_ss.astype(bool), nan_ca)
        assert nan_ss.any(), "Expected at least one disconnecting N-1 on IEEE14"
        assert (~nan_ss).any(), "Expected at least one connected N-1 on IEEE14"

        finite = ~nan_ss
        err = np.abs(V_ss[finite] - V_ca[finite])
        assert err.max() < 1e-5
        assert np.abs(res_ss[finite] - res_ca[finite]).max() < 1e-5


class TestRowAlignedZip:
    """The core new property: row i's result depends only on row i's own
    (injection, topology) pair. Permuting the rows must permute the
    (finite) results identically."""

    def test_permuting_rows_permutes_results(self, ieee14_grid, ieee14_base_case):
        d = ieee14_base_case
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
        n_bus = d["n_bus"]

        # A handful of distinct (injection, topology) pairs. Some rows trip a
        # branch, some don't; some rows scale the injection, some don't.
        scales = [1.0, 0.9, 1.0, 1.1, 0.95]
        branch_ids_per_row = [[], [0], [], [1], [0, 1]]
        n_scen = len(scales)
        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, scales)

        solver_a = _make_solver(ieee14_base_case, branch_data, batch_size=n_scen)
        solver_a.set_injections(p_mw, q_mvar, sn_mva)
        solver_a.set_topology(branch_ids_per_row)
        solver_a.run()
        V_a = solver_a.V_results.to_numpy().reshape(n_scen, n_bus)
        res_a = solver_a.residuals.to_numpy()

        perm = [3, 0, 4, 1, 2]
        p_perm = p_mw[perm]
        q_perm = q_mvar[perm]
        topo_perm = [branch_ids_per_row[i] for i in perm]

        solver_b = _make_solver(ieee14_base_case, branch_data, batch_size=n_scen)
        solver_b.set_injections(p_perm, q_perm, sn_mva)
        solver_b.set_topology(topo_perm)
        solver_b.run()
        V_b = solver_b.V_results.to_numpy().reshape(n_scen, n_bus)
        res_b = solver_b.residuals.to_numpy()

        # A row's own topology may disconnect the grid (NaN) independent of
        # this test's point (row independence); permuting must permute the
        # NaN pattern identically too.
        finite_a_permuted = np.isfinite(res_a)[perm]
        finite_b = np.isfinite(res_b)
        assert np.array_equal(finite_a_permuted, finite_b), (
            "Permuting (injection, topology) rows did not permute the "
            "disconnected/converged pattern identically"
        )
        assert finite_b.any(), "Expected at least one row of this example to converge"

        V_a_permuted = V_a[perm][finite_b]
        res_a_permuted = res_a[perm][finite_b]
        assert np.abs(V_a_permuted - V_b[finite_b]).max() < 1e-5, (
            "Permuting (injection, topology) rows did not permute the results "
            "identically -- rows are not independent"
        )
        assert np.abs(res_a_permuted - res_b[finite_b]).max() < 1e-5

    def test_permuting_rows_permutes_results_with_gen_v(self, ieee14_grid,
                                                          ieee14_base_case):
        """Same property as above, with a per-row gen_v (|V| reseed at the
        PV buses) added on top of injections + topology."""
        d = ieee14_base_case
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
        n_bus = d["n_bus"]
        pv = d["pv"]
        assert pv.size > 0, "IEEE-14 should have PV buses"

        scales = [1.0, 0.9, 1.0, 1.1, 0.95]
        branch_ids_per_row = [[], [0], [], [1], [0, 1]]
        n_scen = len(scales)
        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, scales)

        # A distinct target vm_pu per row at each PV bus, well within a
        # normal operating range.
        targets = np.array([1.00, 1.01, 1.02, 0.99, 0.98])
        gen_v = np.full((n_scen, n_bus), np.nan, dtype=np.float64)
        gen_v[:, pv] = targets[:, None]

        solver_a = _make_solver(ieee14_base_case, branch_data, batch_size=n_scen)
        solver_a.set_injections(p_mw, q_mvar, sn_mva)
        solver_a.set_topology(branch_ids_per_row)
        solver_a.set_gen_v(gen_v)
        solver_a.run()
        V_a = solver_a.V_results.to_numpy().reshape(n_scen, n_bus)
        res_a = solver_a.residuals.to_numpy()

        perm = [3, 0, 4, 1, 2]
        p_perm = p_mw[perm]
        q_perm = q_mvar[perm]
        topo_perm = [branch_ids_per_row[i] for i in perm]
        gen_v_perm = gen_v[perm]

        solver_b = _make_solver(ieee14_base_case, branch_data, batch_size=n_scen)
        solver_b.set_injections(p_perm, q_perm, sn_mva)
        solver_b.set_topology(topo_perm)
        solver_b.set_gen_v(gen_v_perm)
        solver_b.run()
        V_b = solver_b.V_results.to_numpy().reshape(n_scen, n_bus)
        res_b = solver_b.residuals.to_numpy()

        finite_a_permuted = np.isfinite(res_a)[perm]
        finite_b = np.isfinite(res_b)
        assert np.array_equal(finite_a_permuted, finite_b)
        assert finite_b.any()

        V_a_permuted = V_a[perm][finite_b]
        res_a_permuted = res_a[perm][finite_b]
        assert np.abs(V_a_permuted - V_b[finite_b]).max() < 1e-5
        assert np.abs(res_a_permuted - res_b[finite_b]).max() < 1e-5

        # Sanity: |V| at each PV bus actually landed on its row's target.
        for s in range(n_scen):
            if not np.isfinite(res_a[s]):
                continue
            assert np.abs(np.abs(V_a[s, pv]) - targets[s]).max() < 1e-4


class TestIslandedRowHandling:
    def test_islanded_row_is_nan_others_unaffected(self, ieee14_grid, ieee14_base_case):
        d = ieee14_base_case
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
        n_bus = d["n_bus"]
        n_ctg = n_lines + n_trafos

        # One row per N-1 branch trip, base-case injection throughout -- IEEE14
        # is known (see TestInjectionFixedMatchesContingencyAnalysis) to have a
        # mix of connected and disconnecting single-branch trips.
        cont_branch_ids = [[c] for c in range(n_ctg)]
        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, [1.0] * n_ctg)

        solver = _make_solver(ieee14_base_case, branch_data, batch_size=n_ctg)
        solver.set_injections(p_mw, q_mvar, sn_mva)
        solver.set_topology(cont_branch_ids)
        solver.run()

        res = solver.residuals.to_numpy()
        disc = solver.get_disconnected().astype(bool)
        assert disc.any(), "Expected at least one disconnecting N-1 on IEEE14"
        assert (~disc).any(), "Expected at least one connected N-1 on IEEE14"

        assert np.all(~np.isfinite(res[disc])), (
            "Disconnected rows must report a non-finite (NaN) residual"
        )
        assert np.all(np.isfinite(res[~disc])), (
            "A disconnecting row must not corrupt other (connected) rows' results"
        )
        assert np.abs(res[~disc]).max() < _RESIDUAL_TOL


class TestTrippedBranchZeroFlow:
    def test_tripped_branch_has_zero_or_amps(self, ieee14_grid, ieee14_base_case):
        d = ieee14_base_case
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
        n_branches = n_lines + n_trafos
        n_ctg = n_branches
        cont_branch_ids = [[c] for c in range(n_ctg)]
        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, [1.0] * n_ctg)

        solver = _make_solver(ieee14_base_case, branch_data, batch_size=n_ctg)
        solver.set_injections(p_mw, q_mvar, sn_mva)
        solver.set_topology(cont_branch_ids)
        solver.run()
        solver.compute_flows()

        or_amps = solver.or_amps.to_numpy().reshape(n_ctg, n_branches)
        res = solver.residuals.to_numpy()
        converged = np.abs(res) < _RESIDUAL_CONVERGED

        failures = []
        for c_id, branch_ids in enumerate(cont_branch_ids):
            if not converged[c_id]:
                continue
            for b in branch_ids:
                flow = or_amps[c_id, b]
                if flow != 0.0:
                    failures.append((c_id, b, flow))
        assert not failures, (
            "Tripped branch(es) carry non-zero flow:\n" +
            "\n".join(f"  scenario {c}, branch {b}: {f:.4f} A" for c, b, f in failures)
        )


class TestScenarioSweepStrategies:
    @pytest.mark.parametrize("strategy_name", [
        "DirectRefactorEvery",
        "DirectIter0Only",
        "DirectBaseCaseFactors",
        "DirectRefactorEveryN",
    ])
    def test_strategy_residuals_finite_for_connected_rows(
        self, ieee14_grid, ieee14_base_case, strategy_name
    ):
        from gpusim2grid._gpusim2grid import ContingencySolverType

        d = ieee14_base_case
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)

        scales = [0.9, 0.95, 1.0, 1.05, 1.1]
        # Keep topology fixed (no trips) so this test isolates the strategy's
        # numeric behavior from the connectivity-skip machinery.
        branch_ids_per_row = [[] for _ in scales]
        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, scales)

        solver = _make_solver(ieee14_base_case, branch_data, batch_size=len(scales))
        solver.set_injections(p_mw, q_mvar, sn_mva)
        solver.set_topology(branch_ids_per_row)
        solver.strategy = getattr(ContingencySolverType, strategy_name)
        if strategy_name == "DirectRefactorEveryN":
            solver.refactor_period = 2
        solver.run()

        res = solver.residuals.to_numpy()
        V = solver.V_results.to_numpy()
        assert np.all(np.isfinite(res)), f"Non-finite residuals with {strategy_name}"
        assert np.all(np.isfinite(V)), f"Non-finite voltages with {strategy_name}"


class TestScenarioSweepFacade:
    def test_facade_bridge_set_injections_and_topology(self, ieee14_grid, ieee14_base_case):
        from gpusim2grid import ScenarioSweepGPU

        d = ieee14_base_case
        n_bus = d["n_bus"]
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)

        sweep = ScenarioSweepGPU(ieee14_grid, nb_iter=10)
        scales = [1.0, 0.95]
        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, scales)
        sweep.set_injections(p_mw, q_mvar, sn_mva)
        sweep.set_topology([[], [0]])

        V_dlpack = sweep.compute(batch_size=2)
        try:
            import torch
            V = torch.from_dlpack(V_dlpack).cpu().numpy()
        except ImportError:
            V = np.asarray(sweep.V_results.to_numpy()).reshape(2, n_bus)

        res = sweep.last_residuals()
        assert res.shape == (2,)
        assert np.isfinite(res[0])
        assert np.abs(res[0]) < _RESIDUAL_TOL

        disc = sweep.get_disconnected()
        assert disc.shape == (2,)
        assert disc[0] == 0

    def test_facade_set_injections_from_elements(self, ieee14_grid, ieee14_base_case):
        from gpusim2grid import ScenarioSweepGPU

        sweep = ScenarioSweepGPU(ieee14_grid, nb_iter=10)
        n_scen = 3

        # (load_p, load_q, gen_p) as lightsim2grid itself stamped them into
        # Sbus -- same pattern as test_injection_elements.py's _base_injections.
        load_res = ieee14_grid.get_loads_res_full()
        load_p_base = np.asarray(load_res[0], dtype=float)
        load_q_base = np.asarray(load_res[1], dtype=float)
        gen_p_base = np.asarray(ieee14_grid.get_gen_target_p(), dtype=float)

        scale = np.array([0.9, 1.0, 1.1]).reshape(-1, 1)
        load_p = np.tile(load_p_base, (n_scen, 1)) * scale
        load_q = np.tile(load_q_base, (n_scen, 1)) * scale
        gen_p = np.tile(gen_p_base, (n_scen, 1))

        sweep.set_injections_from_elements(load_p, load_q, gen_p)
        sweep.set_topology([[], [0], []])
        sweep.compute(batch_size=n_scen)
        res = sweep.last_residuals()
        assert res.shape == (n_scen,)
        assert np.isfinite(res[0]) and np.isfinite(res[2])

    def test_facade_topology_optional(self, ieee14_grid, ieee14_base_case):
        """compute() must work without ever calling set_topology()."""
        from gpusim2grid import ScenarioSweepGPU

        d = ieee14_base_case
        sweep = ScenarioSweepGPU(ieee14_grid, nb_iter=10)
        scales = [1.0, 1.05]
        p_mw, q_mvar, sn_mva = _scaled_injections(ieee14_base_case, scales)
        sweep.set_injections(p_mw, q_mvar, sn_mva)
        sweep.compute(batch_size=2)
        res = sweep.last_residuals()
        assert np.all(np.isfinite(res))
        assert np.abs(res).max() < _RESIDUAL_TOL

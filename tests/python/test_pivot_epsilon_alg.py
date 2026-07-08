"""
test_pivot_epsilon_alg.py — CUDSS_CONFIG_PIVOT_EPSILON_ALG plumbing.

Covers the PivotEpsilonAlg enum and the pivot_epsilon_alg kwarg/property on
all three public facades (AcPfGPU, ContingencyAnalysisGPU, InjectionSweepGPU),
using the explicit-array-tuple constructor path so the tests do not depend on
whether the lightsim2grid zero-copy bridge was compiled in.

Unlike MatchingAlg, no batch-mode restriction is assumed up front -- 'scaled'
and 'static' are tested for convergence on all three facades exactly like
'default'. See test_matching_alg.py / test_reordering_alg.py module docstrings
for how those restrictions were discovered empirically; this file follows the
same approach rather than presuming one here.
"""
import numpy as np
import pytest

from conftest import requires_gpu, branch_data_arrays

pytestmark = requires_gpu

_ALL_ALGS = ['default', 'scaled', 'static']


class TestPivotEpsilonAlgEnum:
    def test_enum_exists(self):
        from gpusim2grid._gpusim2grid import PivotEpsilonAlg
        for name in ("Default", "Scaled", "Static"):
            assert hasattr(PivotEpsilonAlg, name)


class TestAcPfGpuPivotEpsilonAlg:
    @pytest.mark.parametrize("alg", _ALL_ALGS)
    def test_converges_for_each_alg(self, ieee14_base_case, solver_atol, alg):
        from gpusim2grid import AcPfGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        ac = AcPfGPU(grid_tuple, init_from_n_powerflow=False,
                     max_iter=20, tol=1e-8, pivot_epsilon_alg=alg)
        V = ac.solve()
        assert not np.any(np.isnan(V))
        np.testing.assert_allclose(
            np.abs(V), np.abs(d["v_ref"]), atol=10 * solver_atol)

    def test_enum_passthrough(self, ieee14_base_case, solver_atol):
        from gpusim2grid import AcPfGPU
        from gpusim2grid._gpusim2grid import PivotEpsilonAlg

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        ac = AcPfGPU(grid_tuple, init_from_n_powerflow=False, max_iter=20,
                     tol=1e-8, pivot_epsilon_alg=PivotEpsilonAlg.Scaled)
        V = ac.solve()
        np.testing.assert_allclose(
            np.abs(V), np.abs(d["v_ref"]), atol=10 * solver_atol)

    def test_invalid_string_raises(self, ieee14_base_case):
        from gpusim2grid import AcPfGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        with pytest.raises(ValueError):
            AcPfGPU(grid_tuple, init_from_n_powerflow=False,
                    pivot_epsilon_alg='not_a_real_alg')


class TestContingencyAnalysisGpuPivotEpsilonAlg:
    def test_default_converges(self, ieee14_base_case, ieee14_grid, residual_atol):
        from gpusim2grid import ContingencyAnalysisGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        branch_data, n_lines, _ = branch_data_arrays(ieee14_grid)

        ca = ContingencyAnalysisGPU(grid_tuple, init_from_n_powerflow=False,
                                    nb_iter=10)
        assert ca.pivot_epsilon_alg == 'default'

        ca.set_branch_data(*branch_data)
        ca.add_contingencies_by_branch_id([[i] for i in range(min(5, n_lines))])
        ca.compute(batch_size=5)
        assert np.all(ca.last_residuals() < 100 * residual_atol)

    @pytest.mark.parametrize("alg", ['scaled', 'static'])
    def test_converges_for_each_alg(
            self, ieee14_base_case, ieee14_grid, residual_atol, alg):
        from gpusim2grid import ContingencyAnalysisGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        branch_data, n_lines, _ = branch_data_arrays(ieee14_grid)

        ca = ContingencyAnalysisGPU(grid_tuple, init_from_n_powerflow=False,
                                    nb_iter=10)
        ca.pivot_epsilon_alg = alg
        ca.set_branch_data(*branch_data)
        ca.add_contingencies_by_branch_id([[i] for i in range(min(5, n_lines))])
        ca.compute(batch_size=5)
        assert np.all(ca.last_residuals() < 100 * residual_atol)

    def test_invalid_string_raises(self, ieee14_base_case):
        from gpusim2grid import ContingencyAnalysisGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        ca = ContingencyAnalysisGPU(grid_tuple, init_from_n_powerflow=False)
        with pytest.raises(ValueError):
            ca.pivot_epsilon_alg = 'not_a_real_alg'


class TestInjectionSweepGpuPivotEpsilonAlg:
    def test_default_converges(self, ieee14_base_case):
        from gpusim2grid import InjectionSweepGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        sweep = InjectionSweepGPU(grid_tuple, init_from_n_powerflow=False,
                                  nb_iter=10)
        assert sweep.pivot_epsilon_alg == 'default'

        p_mw = np.tile(d["Sbus"].real * 100.0, (2, 1))
        q_mvar = np.tile(d["Sbus"].imag * 100.0, (2, 1))
        sweep.set_injections(p_mw, q_mvar, sn_mva=100.0)
        sweep.compute(batch_size=2)
        residuals = sweep.last_residuals()
        assert np.all(residuals < 1e-4)

    @pytest.mark.parametrize("alg", ['scaled', 'static'])
    def test_converges_for_each_alg(self, ieee14_base_case, alg):
        from gpusim2grid import InjectionSweepGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        sweep = InjectionSweepGPU(grid_tuple, init_from_n_powerflow=False,
                                  nb_iter=10)
        sweep.pivot_epsilon_alg = alg
        p_mw = np.tile(d["Sbus"].real * 100.0, (2, 1))
        q_mvar = np.tile(d["Sbus"].imag * 100.0, (2, 1))
        sweep.set_injections(p_mw, q_mvar, sn_mva=100.0)
        sweep.compute(batch_size=2)
        residuals = sweep.last_residuals()
        assert np.all(residuals < 1e-4)

    def test_invalid_string_raises(self, ieee14_base_case):
        from gpusim2grid import InjectionSweepGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        sweep = InjectionSweepGPU(grid_tuple, init_from_n_powerflow=False)
        with pytest.raises(ValueError):
            sweep.pivot_epsilon_alg = 'not_a_real_alg'

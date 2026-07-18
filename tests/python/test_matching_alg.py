# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
test_matching_alg.py — CUDSS_CONFIG_MATCHING_ALG plumbing.

Covers the MatchingAlg enum and the matching_alg kwarg/property on all three
public facades (AcPfGPU, ContingencyAnalysisGPU, InjectionSweepGPU), using the
explicit-array-tuple constructor path so the tests do not depend on whether
the lightsim2grid zero-copy bridge was compiled in.

Empirical findings (verified against the IEEE-14 reference solution, not
assumed):

- Batch mode (ContingencyAnalysisGPU / InjectionSweepGPU, always uniform-batch
  via CUDSS_CONFIG_UBATCH_SIZE): cuDSS accepts ONLY 'none' (matching
  disabled). Every other value raises RuntimeError
  (CUDSS_STATUS_NOT_SUPPORTED) -- exactly like BTF_COLAMD/COLAMD did for
  reordering_alg, but here it is ALL non-default matching values, not just two.
- Single-system (AcPfGPU): 'none', 'max_diag_count', 'max_min_diag',
  'max_min_diag_alt', 'max_diag_sum' all reproduce the reference solution.
  'max_diag_product' and 'auto' silently produce NaN voltages (all non-slack
  buses) while ``timings.converged`` still reports True -- the ||F||_inf
  residual check does not catch NaN (NaN comparisons are always False). This
  is a cuDSS/matching-scaling interaction with gpusim2grid's power-flow
  Jacobians, not a gpusim2grid plumbing bug -- see
  test_single_system_max_diag_product_and_auto_produce_nan below. Do not
  treat MaxDiagProduct/Auto as safe defaults.
"""
import numpy as np
import pytest

from conftest import requires_gpu, branch_data_arrays

pytestmark = requires_gpu

# Verified against the IEEE-14 reference solution -- see module docstring.
_SINGLE_SYSTEM_SAFE_ALGS = [
    'none', 'max_diag_count', 'max_min_diag', 'max_min_diag_alt', 'max_diag_sum',
]
_SINGLE_SYSTEM_NAN_ALGS = ['max_diag_product', 'auto']
_BATCH_UNSUPPORTED_ALGS = [
    'max_diag_count', 'max_min_diag', 'max_min_diag_alt',
    'max_diag_sum', 'max_diag_product', 'auto',
]


class TestMatchingAlgEnum:
    def test_enum_exists(self):
        from gpusim2grid._gpusim2grid import MatchingAlg
        for name in ("NoMatching", "MaxDiagCount", "MaxMinDiag", "MaxMinDiagAlt",
                     "MaxDiagSum", "MaxDiagProduct", "Auto"):
            assert hasattr(MatchingAlg, name)


class TestAcPfGpuMatchingAlg:
    @pytest.mark.parametrize("alg", _SINGLE_SYSTEM_SAFE_ALGS)
    def test_converges_for_each_safe_alg(self, ieee14_base_case, solver_atol, alg):
        from gpusim2grid import AcPfGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        ac = AcPfGPU(grid_tuple, init_from_n_powerflow=False,
                     max_iter=20, tol=1e-8, matching_alg=alg)
        V = ac.solve()
        assert not np.any(np.isnan(V))
        np.testing.assert_allclose(
            np.abs(V), np.abs(d["v_ref"]), atol=10 * solver_atol)

    @pytest.mark.parametrize("alg", _SINGLE_SYSTEM_NAN_ALGS)
    def test_single_system_max_diag_product_and_auto_produce_nan(
            self, ieee14_base_case, alg):
        """Documents a real (non-plumbing) finding: cuDSS's MaxDiagProduct/Auto
        matching silently produces NaN voltages for gpusim2grid's power-flow
        Jacobians, and the existing ||F||_inf convergence check does not catch
        it (NaN comparisons are always False). This is not something to
        silently work around here -- callers must be warned (see docstrings)
        rather than have gpusim2grid quietly "fix" or hide it."""
        from gpusim2grid import AcPfGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        ac = AcPfGPU(grid_tuple, init_from_n_powerflow=False,
                     max_iter=20, tol=1e-8, matching_alg=alg)
        V = ac.solve()
        assert np.any(np.isnan(V)), (
            f"matching_alg={alg!r} was expected to reproduce the known NaN "
            "issue; if this now passes, cuDSS behavior may have changed -- "
            "update _SINGLE_SYSTEM_SAFE_ALGS / the docstrings accordingly.")

    def test_enum_passthrough(self, ieee14_base_case, solver_atol):
        from gpusim2grid import AcPfGPU
        from gpusim2grid._gpusim2grid import MatchingAlg

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        ac = AcPfGPU(grid_tuple, init_from_n_powerflow=False, max_iter=20,
                     tol=1e-8, matching_alg=MatchingAlg.MaxDiagCount)
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
                    matching_alg='not_a_real_alg')


class TestContingencyAnalysisGpuMatchingAlg:
    def test_none_converges(self, ieee14_base_case, ieee14_grid, residual_atol):
        from gpusim2grid import ContingencyAnalysisGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        branch_data, n_lines, _ = branch_data_arrays(ieee14_grid)

        ca = ContingencyAnalysisGPU(grid_tuple, init_from_n_powerflow=False,
                                    nb_iter=10)
        assert ca.matching_alg == 'none'

        ca.set_branch_data(*branch_data)
        ca.add_contingencies_by_branch_id([[i] for i in range(min(5, n_lines))])
        ca.compute(batch_size=5)
        assert np.all(ca.last_residuals() < 100 * residual_atol)

    @pytest.mark.parametrize("alg", _BATCH_UNSUPPORTED_ALGS)
    def test_batch_rejects_every_non_default_alg(
            self, ieee14_base_case, ieee14_grid, alg):
        """cuDSS raises CUDSS_STATUS_NOT_SUPPORTED for every matching_alg
        other than 'none' when CUDSS_CONFIG_UBATCH_SIZE is also set -- a
        cuDSS-side limitation of uniform-batch mode, not a gpusim2grid bug."""
        from gpusim2grid import ContingencyAnalysisGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        branch_data, n_lines, _ = branch_data_arrays(ieee14_grid)

        ca = ContingencyAnalysisGPU(grid_tuple, init_from_n_powerflow=False,
                                    nb_iter=10)
        ca.matching_alg = alg
        ca.set_branch_data(*branch_data)
        ca.add_contingencies_by_branch_id([[i] for i in range(min(5, n_lines))])
        with pytest.raises(RuntimeError, match="CudssContext::analyze"):
            ca.compute(batch_size=5)

    def test_invalid_string_raises(self, ieee14_base_case):
        from gpusim2grid import ContingencyAnalysisGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        ca = ContingencyAnalysisGPU(grid_tuple, init_from_n_powerflow=False)
        with pytest.raises(ValueError):
            ca.matching_alg = 'not_a_real_alg'


class TestInjectionSweepGpuMatchingAlg:
    def test_none_converges(self, ieee14_base_case):
        from gpusim2grid import InjectionSweepGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        sweep = InjectionSweepGPU(grid_tuple, init_from_n_powerflow=False,
                                  nb_iter=10)
        assert sweep.matching_alg == 'none'

        p_mw = np.tile(d["Sbus"].real * 100.0, (2, 1))
        q_mvar = np.tile(d["Sbus"].imag * 100.0, (2, 1))
        sweep.set_injections(p_mw, q_mvar, sn_mva=100.0)
        sweep.compute(batch_size=2)
        residuals = sweep.last_residuals()
        assert np.all(residuals < 1e-4)

    @pytest.mark.parametrize("alg", _BATCH_UNSUPPORTED_ALGS)
    def test_batch_rejects_every_non_default_alg(self, ieee14_base_case, alg):
        """See TestContingencyAnalysisGpuMatchingAlg's test of the same name:
        cuDSS-side limitation of uniform-batch mode."""
        from gpusim2grid import InjectionSweepGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        sweep = InjectionSweepGPU(grid_tuple, init_from_n_powerflow=False,
                                  nb_iter=10)
        sweep.matching_alg = alg
        p_mw = np.tile(d["Sbus"].real * 100.0, (2, 1))
        q_mvar = np.tile(d["Sbus"].imag * 100.0, (2, 1))
        sweep.set_injections(p_mw, q_mvar, sn_mva=100.0)
        with pytest.raises(RuntimeError, match="CudssContext::analyze"):
            sweep.compute(batch_size=2)

    def test_invalid_string_raises(self, ieee14_base_case):
        from gpusim2grid import InjectionSweepGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        sweep = InjectionSweepGPU(grid_tuple, init_from_n_powerflow=False)
        with pytest.raises(ValueError):
            sweep.matching_alg = 'not_a_real_alg'

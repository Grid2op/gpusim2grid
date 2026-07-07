"""
test_reordering_alg.py — CUDSS_CONFIG_REORDERING_ALG plumbing.

Covers the ReorderingAlg enum and the reordering_alg kwarg/property on all
three public facades (AcPfGPU, ContingencyAnalysisGPU, InjectionSweepGPU),
using the explicit-array-tuple constructor path so the tests do not depend on
whether the lightsim2grid zero-copy bridge was compiled in.
"""
import numpy as np
import pytest

from conftest import requires_gpu, branch_data_arrays

pytestmark = requires_gpu

_ALL_ALGS = ['default', 'btf_colamd', 'colamd', 'amd', 'nested_dissection', 'none']

# cuDSS rejects BTF_COLAMD/COLAMD (CUDSS_STATUS_NOT_SUPPORTED) when
# CUDSS_CONFIG_UBATCH_SIZE is also set -- i.e. for the batch workloads
# (ContingencyAnalysisGPU / InjectionSweepGPU), which always run in uniform-batch
# mode. They work fine for AcPfGPU's single-system solve. This is a cuDSS-side
# constraint, not a gpusim2grid bug -- see test_batch_rejects_btf_colamd_and_colamd.
_BATCH_SUPPORTED_ALGS = ['default', 'amd', 'nested_dissection', 'none']
_BATCH_UNSUPPORTED_ALGS = ['btf_colamd', 'colamd']


class TestReorderingAlgEnum:
    def test_enum_exists(self):
        from gpusim2grid._gpusim2grid import ReorderingAlg
        for name in ("Default", "BtfColamd", "Colamd", "Amd",
                     "NestedDissection", "NoReordering"):
            assert hasattr(ReorderingAlg, name)


class TestAcPfGpuReorderingAlg:
    @pytest.mark.parametrize("alg", _ALL_ALGS)
    def test_converges_for_each_alg(self, ieee14_base_case, solver_atol, alg):
        from gpusim2grid import AcPfGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        ac = AcPfGPU(grid_tuple, init_from_n_powerflow=False,
                     max_iter=20, tol=1e-8, reordering_alg=alg)
        V = ac.solve()
        np.testing.assert_allclose(
            np.abs(V), np.abs(d["v_ref"]), atol=10 * solver_atol)

    def test_enum_passthrough(self, ieee14_base_case, solver_atol):
        from gpusim2grid import AcPfGPU
        from gpusim2grid._gpusim2grid import ReorderingAlg

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        ac = AcPfGPU(grid_tuple, init_from_n_powerflow=False, max_iter=20,
                     tol=1e-8, reordering_alg=ReorderingAlg.Colamd)
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
                    reordering_alg='not_a_real_alg')


class TestContingencyAnalysisGpuReorderingAlg:
    @pytest.mark.parametrize("alg", _BATCH_SUPPORTED_ALGS)
    def test_converges_for_each_alg(self, ieee14_base_case, ieee14_grid, alg,
                                     residual_atol):
        from gpusim2grid import ContingencyAnalysisGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        branch_data, n_lines, _ = branch_data_arrays(ieee14_grid)

        ca = ContingencyAnalysisGPU(grid_tuple, init_from_n_powerflow=False,
                                    nb_iter=10)
        assert ca.reordering_alg == 'default'
        ca.reordering_alg = alg
        assert ca.reordering_alg == alg

        ca.set_branch_data(*branch_data)
        ca.add_contingencies_by_branch_id([[i] for i in range(min(5, n_lines))])
        ca.compute(batch_size=5)
        assert np.all(ca.last_residuals() < 100 * residual_atol)

    @pytest.mark.parametrize("alg", _BATCH_UNSUPPORTED_ALGS)
    def test_batch_rejects_btf_colamd_and_colamd(
            self, ieee14_base_case, ieee14_grid, alg):
        """cuDSS raises CUDSS_STATUS_NOT_SUPPORTED for BTF_COLAMD/COLAMD when
        CUDSS_CONFIG_UBATCH_SIZE is also set -- a cuDSS-side limitation of
        uniform-batch mode, not a gpusim2grid bug. Only AcPfGPU's single-system
        solve supports these two."""
        from gpusim2grid import ContingencyAnalysisGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        branch_data, n_lines, _ = branch_data_arrays(ieee14_grid)

        ca = ContingencyAnalysisGPU(grid_tuple, init_from_n_powerflow=False,
                                    nb_iter=10)
        ca.reordering_alg = alg
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
            ca.reordering_alg = 'not_a_real_alg'


class TestInjectionSweepGpuReorderingAlg:
    @pytest.mark.parametrize("alg", _BATCH_SUPPORTED_ALGS)
    def test_converges_for_each_alg(self, ieee14_base_case, alg):
        from gpusim2grid import InjectionSweepGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        sweep = InjectionSweepGPU(grid_tuple, init_from_n_powerflow=False,
                                  nb_iter=10)
        assert sweep.reordering_alg == 'default'
        sweep.reordering_alg = alg
        assert sweep.reordering_alg == alg

        p_mw = np.tile(d["Sbus"].real * 100.0, (2, 1))
        q_mvar = np.tile(d["Sbus"].imag * 100.0, (2, 1))
        sweep.set_injections(p_mw, q_mvar, sn_mva=100.0)
        sweep.compute(batch_size=2)
        residuals = sweep.last_residuals()
        assert np.all(residuals < 1e-4)

    @pytest.mark.parametrize("alg", _BATCH_UNSUPPORTED_ALGS)
    def test_batch_rejects_btf_colamd_and_colamd(self, ieee14_base_case, alg):
        """See TestContingencyAnalysisGpuReorderingAlg's test of the same
        name: cuDSS-side limitation of uniform-batch mode."""
        from gpusim2grid import InjectionSweepGPU

        d = ieee14_base_case
        grid_tuple = (d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"])
        sweep = InjectionSweepGPU(grid_tuple, init_from_n_powerflow=False,
                                  nb_iter=10)
        sweep.reordering_alg = alg
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
            sweep.reordering_alg = 'not_a_real_alg'

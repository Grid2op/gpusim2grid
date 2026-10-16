# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
test_ir_n_steps.py — CUDSS_CONFIG_IR_N_STEPS plumbing.

``ir_n_steps`` (int, default 0) is the number of cuDSS iterative refinement
steps after each linear solve, on the base case and the batch, forward and
adjoint. cuDSS factorizes with static pivoting: a pivot too small is replaced
by an epsilon, so the factors solve a slightly different matrix, and a step of
refinement (one residual, one more solve) recovers the lost accuracy.

The facades are built in explicit-array mode, so the tests do not depend on
whether the lightsim2grid bridge was compiled in.
"""
import numpy as np
import pytest

from conftest import requires_gpu, branch_data_arrays

pytestmark = requires_gpu


def _tuple(d):
    return (d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"])


class TestValidation:
    @pytest.mark.parametrize("bad, exc", [(-1, ValueError), (1.5, TypeError),
                                          (True, TypeError), ("1", TypeError)])
    def test_rejected_values(self, ieee14_base_case, bad, exc):
        from gpusim2grid import AcPfGPU, ContingencyAnalysisGPU
        with pytest.raises(exc):
            AcPfGPU(_tuple(ieee14_base_case), init_from_n_powerflow=False, ir_n_steps=bad)
        ca = ContingencyAnalysisGPU(_tuple(ieee14_base_case), init_from_n_powerflow=False)
        with pytest.raises(exc):
            ca.ir_n_steps = bad

    def test_numpy_int_accepted(self, ieee14_base_case):
        from gpusim2grid import ContingencyAnalysisGPU
        ca = ContingencyAnalysisGPU(_tuple(ieee14_base_case), init_from_n_powerflow=False,
                                    ir_n_steps=np.int64(2))
        assert ca.ir_n_steps == 2


class TestAcPfGpu:
    @pytest.mark.parametrize("n", [0, 1, 3])
    def test_converges(self, ieee14_base_case, solver_atol, n):
        from gpusim2grid import AcPfGPU
        d = ieee14_base_case
        V = AcPfGPU(_tuple(d), init_from_n_powerflow=False, max_iter=20, tol=1e-8,
                    ir_n_steps=n).solve()
        np.testing.assert_allclose(V, d["v_ref"], atol=10 * solver_atol)


class TestBatchFacades:
    def test_contingency_analysis(self, ieee14_base_case, ieee14_grid, solver_atol):
        from gpusim2grid import ContingencyAnalysisGPU
        branch_data, n_lines, _ = branch_data_arrays(ieee14_grid)
        V = {}
        for n in (0, 2):
            ca = ContingencyAnalysisGPU(_tuple(ieee14_base_case), init_from_n_powerflow=False,
                                        nb_iter=10, ir_n_steps=n)
            assert ca.ir_n_steps == n
            ca.set_branch_data(*branch_data)
            ca.add_contingencies_by_branch_id([[i] for i in range(min(5, n_lines))])
            ca.compute(batch_size=5)
            V[n] = ca.V_results.to_numpy()
        np.testing.assert_allclose(V[2], V[0], atol=solver_atol)

    def test_setter_rebuilds_the_driver(self, ieee14_base_case, ieee14_grid):
        """A change takes effect on the next compute(): the driver snapshot
        carries the value, so the session builds a new driver."""
        from gpusim2grid import ContingencyAnalysisGPU
        branch_data, n_lines, _ = branch_data_arrays(ieee14_grid)
        ca = ContingencyAnalysisGPU(_tuple(ieee14_base_case), init_from_n_powerflow=False,
                                    nb_iter=10)
        assert ca.ir_n_steps == 0
        ca.set_branch_data(*branch_data)
        ca.add_contingencies_by_branch_id([[i] for i in range(min(5, n_lines))])
        ca.compute(batch_size=5)
        s = ca._inner._s
        built = s.driver_build_counter
        ca.compute(batch_size=5)
        assert s.driver_build_counter == built          # same settings: hot
        ca.ir_n_steps = 1
        assert s.ir_n_steps == 1
        ca.compute(batch_size=5)
        assert s.driver_build_counter == built + 1

    def test_injection_sweep(self, ieee14_base_case):
        from gpusim2grid import InjectionSweepGPU
        d = ieee14_base_case
        sweep = InjectionSweepGPU(_tuple(d), init_from_n_powerflow=False, nb_iter=10,
                                  ir_n_steps=1)
        assert sweep.ir_n_steps == 1
        sweep.set_injections(np.tile(d["Sbus"].real * 100.0, (2, 1)),
                             np.tile(d["Sbus"].imag * 100.0, (2, 1)), sn_mva=100.0)
        sweep.compute(batch_size=2)
        assert np.all(sweep.last_residuals() < 1e-4)

    def test_scenario_sweep(self, ieee14_base_case, ieee14_grid):
        from gpusim2grid import ScenarioSweepGPU
        d = ieee14_base_case
        branch_data, _, _ = branch_data_arrays(ieee14_grid)
        sweep = ScenarioSweepGPU(_tuple(d), init_from_n_powerflow=False, nb_iter=10,
                                 ir_n_steps=1)
        assert sweep.ir_n_steps == 1
        sweep.set_branch_data(*branch_data)
        sweep.set_injections(np.tile(d["Sbus"].real * 100.0, (2, 1)),
                             np.tile(d["Sbus"].imag * 100.0, (2, 1)), sn_mva=100.0)
        sweep.compute(batch_size=2)
        assert np.all(sweep.last_residuals() < 1e-4)
        sweep.ir_n_steps = 0
        assert sweep.ir_n_steps == 0


def _saddle_point_matrix(n=200):
    """[[H, Bᵀ], [B, 0]]: the zero block leaves cuDSS's static pivoting pivots
    to perturb, so the factors alone solve it only approximately."""
    import scipy.sparse as sp
    m = n // 4
    B = sp.random(m, n, density=0.05, random_state=2)
    H = sp.random(n, n, density=0.02, random_state=3)
    H = H + H.T + 4 * sp.eye(n)
    K = sp.bmat([[H, B.T], [B, None]], format="csr")
    K.sort_indices()
    return K


def test_refinement_recovers_a_perturbed_pivot_solve():
    """benchmark_cudss_batch_raw runs the driver's own CudssBatchSolver: with
    refinement the solve of a matrix whose pivots were perturbed is accurate,
    without it it need not be."""
    from gpusim2grid import _gpusim2grid as c
    K = _saddle_point_matrix()
    res = {}
    for n in (0, 2):
        r = c.benchmark_cudss_batch_raw(K.shape[0], K.indptr.astype(np.int32),
                                        K.indices.astype(np.int32), K.data, 4, 1,
                                        ir_n_steps=n)
        assert r.n_nonfinite_slots == 0
        res[n] = r.max_rel_residual
    assert res[2] < 1e-6
    assert res[2] <= res[0]

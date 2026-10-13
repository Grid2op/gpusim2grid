# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""cuDSS factor statistics (CUDSS_DATA_LU_NNZ / CUDSS_DATA_MEMORY_ESTIMATES).

Surfaced on every batch session's timings (``cudss_lu_nnz``, ``cudss_mem_*``)
and by ``benchmark_cudss_batch_raw``, which runs the driver's own
``CudssBatchSolver`` on an arbitrary CSR matrix.
"""
import pytest
import scipy.sparse as sp

from conftest import requires_gpu, branch_data_arrays

pytestmark = requires_gpu


def _random_system(n=200, seed=0):
    A = sp.random(n, n, density=0.02, random_state=seed, format="csr") + 5.0 * sp.eye(n)
    A = sp.csr_matrix(A)
    A.sort_indices()
    return A


def _bench(A, batch_size, n_refactorize=2):
    from gpusim2grid import _gpusim2grid as _c
    return _c.benchmark_cudss_batch_raw(
        A.shape[0], A.indptr.tolist(), A.indices.tolist(), A.data.tolist(),
        batch_size, n_refactorize)


def test_benchmark_raw_solves_and_reports_fill_in(solver_atol):
    A = _random_system()
    r = _bench(A, batch_size=4)
    assert r.dim == A.shape[0] and r.nnz == A.nnz and r.batch_size == 4
    assert r.n_nonfinite_slots == 0
    assert r.max_rel_residual < solver_atol
    # the factors hold at least the matrix itself
    assert r.lu_nnz >= A.nnz
    assert r.mem_device_peak >= r.mem_device_permanent > 0
    assert min(r.analysis_ms, r.factorize_ms, r.refactorize_ms, r.solve_ms) > 0.0


def test_lu_nnz_is_per_system():
    A = _random_system()
    assert _bench(A, batch_size=1).lu_nnz == _bench(A, batch_size=16).lu_nnz


def test_lu_nnz_grows_with_fill_in():
    # a dense row + column (arrow pointing the wrong way for no reordering)
    # can only add fill-in, never remove it
    A = _random_system()
    n = A.shape[0]
    B = sp.lil_matrix(A)
    B[0, :] = 1.0
    B[:, 0] = 1.0
    B[0, 0] = 10.0 * n
    B = sp.csr_matrix(B)
    B.sort_indices()
    assert _bench(B, batch_size=2).lu_nnz > _bench(A, batch_size=2).lu_nnz


def test_session_timings_carry_factor_stats(ieee14_grid, ieee14_base_case):
    from gpusim2grid.contingency_analysis import _ContingencyAnalysisSolver

    d = ieee14_base_case
    solver = _ContingencyAnalysisSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=4, nb_iter=4, max_iter_base=10, tol_base=1e-6,
    )
    branch_data, n_lines, _ = branch_data_arrays(ieee14_grid)
    solver.set_branch_data(*branch_data)
    solver.build_contingencies([[c] for c in range(min(n_lines, 8))])
    solver.run()
    t = solver.timings
    assert t.cudss_lu_nnz > 0
    assert t.cudss_mem_device_peak_bytes >= t.cudss_mem_device_permanent_bytes > 0
    assert t.cudss_mem_host_peak_bytes >= 0

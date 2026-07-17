# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
test_dlpack.py — Zero-copy DLPack tensor export.

Tests v_base_dlpack() and v_results_dlpack() on both the raw C++ bindings
(ContingencyAnalysisSession / InjectionSweepSession) and the Python wrappers
(_ContingencyAnalysisSolver / _InjectionSweepSolver).

For each path, asserts:
  - shape is correct
  - dtype matches the compiled precision (complex64 / FP32, complex128 / FP64)
  - values agree with the copy-to-host path (get_V_results)
"""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping DLPack tests")

from gpusim2grid._gpusim2grid import is_fp32  # noqa: E402

_EXPECTED_DTYPE = torch.complex64 if is_fp32 else torch.complex128
_ABS_TOL = 1e-5 if is_fp32 else 1e-12


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def branch_data(ieee14_grid):
    """Branch admittance arrays and bus connectivity for the IEEE 14-bus grid."""
    grid   = ieee14_grid
    lines  = grid.get_lines()
    trafos = grid.get_trafos()

    branch_from = np.concatenate([
        np.array(lines.get_bus_id_side_1(),  dtype=np.int32),
        np.array(trafos.get_bus_id_side_1(), dtype=np.int32),
    ])
    branch_to = np.concatenate([
        np.array(lines.get_bus_id_side_2(),  dtype=np.int32),
        np.array(trafos.get_bus_id_side_2(), dtype=np.int32),
    ])
    yff = np.concatenate([lines.get_yac_eff_11().copy(), trafos.get_yac_eff_11()])
    yft = np.concatenate([lines.get_yac_eff_12().copy(), trafos.get_yac_eff_12()])
    ytf = np.concatenate([lines.get_yac_eff_21().copy(), trafos.get_yac_eff_21()])
    ytt = np.concatenate([lines.get_yac_eff_22().copy(), trafos.get_yac_eff_22()])
    n_branches = len(branch_from)

    return {
        "branch_from": branch_from,
        "branch_to":   branch_to,
        "yff": yff, "yft": yft, "ytf": ytf, "ytt": ytt,
        "vn_kv":      grid.get_bus_vn_kv().copy(),
        "sn_mva":     grid.get_sn_mva(),
        "n_branches": n_branches,
        "ctg_ids":    [[i] for i in range(n_branches)],
    }


@pytest.fixture(scope="module")
def cas_session_ran(ieee14_base_case, branch_data):
    """ContingencyAnalysisSession after a full N-1 run (for v_results_dlpack)."""
    from gpusim2grid._gpusim2grid import ContingencyAnalysisSession as _CAS

    d = ieee14_base_case
    b = branch_data
    n_ctg = b["n_branches"]

    s = _CAS(d["Ybus"], d["v_init"].copy(), d["Sbus"],
             d["slack"], d["slack_weights"], d["pv"], d["pq"],
             n_ctg, 4)
    s.set_branch_data(
        b["branch_from"], b["branch_to"],
        b["yff"], b["yft"], b["ytf"], b["ytt"],
        b["vn_kv"], b["sn_mva"],
    )
    s.build_contingencies(b["ctg_ids"])
    s.run()
    return s, d["n_bus"], n_ctg


@pytest.fixture(scope="module")
def iss_session_ran(ieee14_base_case):
    """InjectionSweepSession after a small injection sweep (for v_results_dlpack)."""
    from gpusim2grid._gpusim2grid import InjectionSweepSession as _ISS

    d = ieee14_base_case
    n_bus       = d["n_bus"]
    n_scenarios = 8
    sn_mva      = d["grid"].get_sn_mva()
    rng         = np.random.default_rng(42)
    p_mw        = rng.uniform(-30, 30, (n_scenarios, n_bus))
    q_mvar      = rng.uniform(-10, 10, (n_scenarios, n_bus))

    s = _ISS(d["Ybus"], d["v_init"].copy(), d["Sbus"],
             d["slack"], d["slack_weights"], d["pv"], d["pq"],
             n_scenarios, 4)
    s.set_injections(p_mw, q_mvar, sn_mva)
    s.run()
    return s, n_bus, n_scenarios


# ---------------------------------------------------------------------------
# ContingencyAnalysisSession — raw C++ binding
# ---------------------------------------------------------------------------

class TestContingencySessionDLPack:
    def test_v_base_shape_dtype(self, ieee14_base_case):
        """v_base_dlpack() before any run: shape (n_bus,) with correct dtype."""
        from gpusim2grid._gpusim2grid import ContingencyAnalysisSession as _CAS

        d = ieee14_base_case
        s = _CAS(d["Ybus"], d["v_init"].copy(), d["Sbus"],
                 d["slack"], d["slack_weights"], d["pv"], d["pq"],
                 1, 4)
        V_base = torch.from_dlpack(s.v_base_dlpack())
        assert V_base.shape == (d["n_bus"],)
        assert V_base.dtype == _EXPECTED_DTYPE

    def test_v_results_shape_dtype(self, cas_session_ran):
        """v_results_dlpack() after N-1 run: shape (n_ctg, n_bus) with correct dtype."""
        s, n_bus, n_ctg = cas_session_ran
        V_gpu = torch.from_dlpack(s.v_results_dlpack())
        assert V_gpu.shape == (n_ctg, n_bus)
        assert V_gpu.dtype == _EXPECTED_DTYPE

    def test_v_results_match_copy_path(self, cas_session_ran):
        """DLPack tensor values must match get_V_results() for non-NaN entries."""
        s, n_bus, n_ctg = cas_session_ran
        V_gpu  = torch.from_dlpack(s.v_results_dlpack()).cpu().numpy()
        V_copy = s.get_V_results().reshape(n_ctg, n_bus)

        valid = ~np.isnan(V_copy.real)
        assert np.all(np.isnan(V_gpu.real[~valid])), \
            "DLPack tensor must propagate NaN where copy path has NaN"
        if valid.any():
            max_err = float(np.max(np.abs(V_gpu[valid] - V_copy[valid])))
            assert max_err < _ABS_TOL, \
                f"max|error| = {max_err:.2e} ≥ tol {_ABS_TOL:.2e}"


# ---------------------------------------------------------------------------
# _ContingencyAnalysisSolver — Python wrapper
# ---------------------------------------------------------------------------

class TestContingencyWrapperDLPack:
    @pytest.fixture(scope="class")
    def solver(self, ieee14_base_case, branch_data):
        from gpusim2grid.contingency_analysis import _ContingencyAnalysisSolver

        d = ieee14_base_case
        b = branch_data
        s = _ContingencyAnalysisSolver(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            batch_size=b["n_branches"], nb_iter=4,
        )
        s.set_branch_data(
            b["branch_from"], b["branch_to"],
            b["yff"], b["yft"], b["ytf"], b["ytt"],
            b["vn_kv"], b["sn_mva"],
        )
        s.build_contingencies(b["ctg_ids"])
        s.run()
        return s, d["n_bus"], b["n_branches"]

    def test_v_results_shape_dtype(self, solver):
        s, n_bus, n_ctg = solver
        V = torch.from_dlpack(s.v_results_dlpack())
        assert V.shape == (n_ctg, n_bus)
        assert V.dtype == _EXPECTED_DTYPE

    def test_v_base_shape_dtype(self, solver):
        s, n_bus, _ = solver
        V = torch.from_dlpack(s.v_base_dlpack())
        assert V.shape == (n_bus,)
        assert V.dtype == _EXPECTED_DTYPE


# ---------------------------------------------------------------------------
# InjectionSweepSession — raw C++ binding
# ---------------------------------------------------------------------------

class TestInjectionSessionDLPack:
    def test_v_base_shape_dtype(self, ieee14_base_case):
        """v_base_dlpack() before any run: shape (n_bus,) with correct dtype."""
        from gpusim2grid._gpusim2grid import InjectionSweepSession as _ISS

        d = ieee14_base_case
        s = _ISS(d["Ybus"], d["v_init"].copy(), d["Sbus"],
                 d["slack"], d["slack_weights"], d["pv"], d["pq"],
                 1, 4)
        V_base = torch.from_dlpack(s.v_base_dlpack())
        assert V_base.shape == (d["n_bus"],)
        assert V_base.dtype == _EXPECTED_DTYPE

    def test_v_results_shape_dtype(self, iss_session_ran):
        """v_results_dlpack() after injection sweep: shape (n_scenarios, n_bus)."""
        s, n_bus, n_scenarios = iss_session_ran
        V_gpu = torch.from_dlpack(s.v_results_dlpack())
        assert V_gpu.shape == (n_scenarios, n_bus)
        assert V_gpu.dtype == _EXPECTED_DTYPE

    def test_v_results_match_copy_path(self, iss_session_ran):
        """DLPack tensor values must match get_V_results() for valid entries."""
        s, n_bus, n_scenarios = iss_session_ran
        V_gpu  = torch.from_dlpack(s.v_results_dlpack()).cpu().numpy()
        V_copy = s.get_V_results().reshape(n_scenarios, n_bus)

        valid = ~np.isnan(V_copy.real)
        if valid.any():
            max_err = float(np.max(np.abs(V_gpu[valid] - V_copy[valid])))
            assert max_err < _ABS_TOL, \
                f"max|error| = {max_err:.2e} ≥ tol {_ABS_TOL:.2e}"


# ---------------------------------------------------------------------------
# _InjectionSweepSolver — Python wrapper
# ---------------------------------------------------------------------------

class TestInjectionWrapperDLPack:
    @pytest.fixture(scope="class")
    def solver(self, ieee14_base_case):
        from gpusim2grid.injection_sweep import _InjectionSweepSolver

        d = ieee14_base_case
        n_bus       = d["n_bus"]
        n_scenarios = 8
        sn_mva      = d["grid"].get_sn_mva()
        rng         = np.random.default_rng(42)
        p_mw        = rng.uniform(-30, 30, (n_scenarios, n_bus))
        q_mvar      = rng.uniform(-10, 10, (n_scenarios, n_bus))

        s = _InjectionSweepSolver(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            batch_size=n_scenarios, nb_iter=4,
        )
        s.set_injections(p_mw, q_mvar, sn_mva)
        s.run()
        return s, n_bus, n_scenarios

    def test_v_results_shape_dtype(self, solver):
        s, n_bus, n_scenarios = solver
        V = torch.from_dlpack(s.v_results_dlpack())
        assert V.shape == (n_scenarios, n_bus)
        assert V.dtype == _EXPECTED_DTYPE

    def test_v_base_shape_dtype(self, solver):
        s, n_bus, _ = solver
        V = torch.from_dlpack(s.v_base_dlpack())
        assert V.shape == (n_bus,)
        assert V.dtype == _EXPECTED_DTYPE


# ---------------------------------------------------------------------------
# AcPfNrSession — standalone single-system NR with DLPack
# ---------------------------------------------------------------------------

class TestAcPfNrSessionDLPack:
    @pytest.fixture(scope="class")
    def session(self, ieee14_base_case):
        from gpusim2grid._gpusim2grid import AcPfNrSession

        d = ieee14_base_case
        return AcPfNrSession(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            max_iter=10, tol=1e-6,
        ), d["n_bus"], d["v_ref"]

    def test_v_dlpack_shape_dtype(self, session):
        """v_dlpack() returns shape (n_bus,) with the compiled-precision dtype."""
        s, n_bus, _ = session
        V = torch.from_dlpack(s.v_dlpack())
        assert V.shape == (n_bus,)
        assert V.dtype == _EXPECTED_DTYPE

    def test_v_dlpack_on_cuda(self, session):
        """v_dlpack() tensor must live on a CUDA device."""
        s, _, _ = session
        V = torch.from_dlpack(s.v_dlpack())
        assert V.is_cuda

    def test_v_dlpack_matches_get_v(self, session):
        """DLPack tensor and get_v() host copy must agree element-wise."""
        s, n_bus, _ = session
        V_gpu  = torch.from_dlpack(s.v_dlpack()).cpu().numpy()
        V_host = s.get_v()
        max_err = float(np.max(np.abs(V_gpu - V_host)))
        assert max_err < _ABS_TOL, f"max|error| = {max_err:.2e} ≥ tol {_ABS_TOL:.2e}"

    def test_v_dlpack_matches_reference(self, session):
        """Solved voltages must match the lightsim2grid AC reference."""
        s, n_bus, v_ref = session
        V_gpu = torch.from_dlpack(s.v_dlpack()).cpu().numpy()
        max_err = float(np.max(np.abs(V_gpu - v_ref)))
        assert max_err < _ABS_TOL, f"max|error| = {max_err:.2e} ≥ tol {_ABS_TOL:.2e}"

    def test_timings_accessible(self, session):
        """timings property must return an AcPfTimings object."""
        from gpusim2grid._gpusim2grid import AcPfTimings
        s, _, _ = session
        t = s.timings
        assert isinstance(t, AcPfTimings)

    def test_python_wrapper_export(self, ieee14_base_case):
        """AcPfNrSession is importable from gpusim2grid.acpf_nr."""
        from gpusim2grid.acpf_nr import AcPfNrSession
        d = ieee14_base_case
        s = AcPfNrSession(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            max_iter=10, tol=1e-6,
        )
        V = torch.from_dlpack(s.v_dlpack())
        assert V.shape == (d["n_bus"],)

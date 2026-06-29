"""
test_diff_flows.py — Differentiable branch-flow tests.

Tests:
  1. test_flows_match_oracle  — compute_flows (PyTorch) agrees with
                                compute_branch_flows_cpu (numpy reference).
  2. test_gradcheck_flows     — gradcheck on Sbus → solve_power_flow →
                                compute_flows → scalar loss.

All tests require GPU.  gradcheck tests are skipped in FP32 builds.
"""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_fp32() -> bool:
    try:
        from gpusim2grid._gpusim2grid import is_fp32 as _fp32
        return bool(_fp32)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def branch_data(ieee14_grid):
    """Branch admittance arrays and connectivity for the IEEE 14-bus grid."""
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

    return {
        "branch_from": branch_from,
        "branch_to":   branch_to,
        "yff": yff, "yft": yft, "ytf": ytf, "ytt": ytt,
        "vn_kv":  grid.get_bus_vn_kv().copy(),
        "sn_mva": grid.get_sn_mva(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeFlows:

    def test_flows_match_oracle(self, ieee14_base_case, branch_data):
        """compute_flows (PyTorch) matches compute_branch_flows_cpu oracle.

        Checks both amps outputs (the oracle returns only amps).
        Tolerance 1e-3 A — loose enough for FP32 rounding, tight enough to
        catch unit/base errors.
        """
        from gpusim2grid.acpf_nr._branch_flows import compute_branch_flows_cpu
        from gpusim2grid.differentiable import compute_flows

        d = ieee14_base_case
        b = branch_data
        V_ref = d["v_ref"]   # converged AC reference solution

        # Determine expected complex dtype
        if _is_fp32():
            cdtype = torch.complex64
            fdtype = torch.float32
            atol   = 1e-2   # FP32: ~1e-2 A tolerance
        else:
            cdtype = torch.complex128
            fdtype = torch.float64
            atol   = 1e-4   # FP64: ~1e-4 A tolerance

        V_t   = torch.from_numpy(V_ref.astype(np.complex128 if not _is_fp32() else np.complex64)).cuda()
        yff_t = torch.from_numpy(b["yff"].astype(np.complex128 if not _is_fp32() else np.complex64)).cuda()
        yft_t = torch.from_numpy(b["yft"].astype(np.complex128 if not _is_fp32() else np.complex64)).cuda()
        ytf_t = torch.from_numpy(b["ytf"].astype(np.complex128 if not _is_fp32() else np.complex64)).cuda()
        ytt_t = torch.from_numpy(b["ytt"].astype(np.complex128 if not _is_fp32() else np.complex64)).cuda()
        frm   = torch.from_numpy(b["branch_from"].astype(np.int64)).cuda()
        to_   = torch.from_numpy(b["branch_to"].astype(np.int64)).cuda()
        vn_t  = torch.from_numpy(b["vn_kv"].astype(np.float64 if not _is_fp32() else np.float32)).cuda()

        flows = compute_flows(V_t, yff_t, yft_t, ytf_t, ytt_t,
                              frm, to_, vn_t, b["sn_mva"])

        # Oracle (always FP64)
        or_a_ref, ex_a_ref = compute_branch_flows_cpu(
            V_ref, b["branch_from"], b["branch_to"],
            b["yff"], b["yft"], b["ytf"], b["ytt"],
            b["vn_kv"], b["sn_mva"])

        i_or_gpu = flows["i_or_a"].cpu().to(torch.float64).numpy()
        i_ex_gpu = flows["i_ex_a"].cpu().to(torch.float64).numpy()

        max_err_or = float(np.max(np.abs(i_or_gpu - or_a_ref)))
        max_err_ex = float(np.max(np.abs(i_ex_gpu - ex_a_ref)))
        assert max_err_or < atol, f"i_or_a max error {max_err_or:.2e} >= {atol}"
        assert max_err_ex < atol, f"i_ex_a max error {max_err_ex:.2e} >= {atol}"

    def test_sign_convention_p_or(self, ieee14_base_case, branch_data):
        """p_or_mw is in the expected range for a loaded IEEE 14-bus grid."""
        from gpusim2grid.differentiable import compute_flows
        from gpusim2grid._gpusim2grid import AcPfNrSession

        d = ieee14_base_case
        b = branch_data

        sess = AcPfNrSession(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            10, 1e-6,
        )
        V_t = torch.from_dlpack(sess.v_dlpack()).clone()

        cdtype = V_t.dtype
        fdtype = torch.float32 if cdtype == torch.complex64 else torch.float64

        flows = compute_flows(
            V_t,
            torch.from_numpy(b["yff"]).to(cdtype).cuda(),
            torch.from_numpy(b["yft"]).to(cdtype).cuda(),
            torch.from_numpy(b["ytf"]).to(cdtype).cuda(),
            torch.from_numpy(b["ytt"]).to(cdtype).cuda(),
            torch.from_numpy(b["branch_from"].astype(np.int64)).cuda(),
            torch.from_numpy(b["branch_to"].astype(np.int64)).cuda(),
            torch.from_numpy(b["vn_kv"]).to(fdtype).cuda(),
            b["sn_mva"],
        )
        p_or = flows["p_or_mw"].abs().cpu().numpy()
        # IEEE 14-bus has non-trivial loads — all branch flows should be > 0
        assert float(p_or.max()) > 1.0, "Expected at least one branch with |p| > 1 MW"

    def test_gradcheck_full_pipeline(self, ieee14_base_case, branch_data):
        """gradcheck through solve_power_flow → compute_flows → scalar loss.

        Uses sum(i_or_a) as the scalar loss.  Skipped in FP32 builds.
        """
        if _is_fp32():
            pytest.skip("FP32 build: gradcheck requires FP64")

        from gpusim2grid.differentiable import solve_power_flow, compute_flows

        d = ieee14_base_case
        b = branch_data

        Sbus_np = d["Sbus"].astype(np.complex128)
        Sr = torch.tensor(Sbus_np.real, dtype=torch.float64, device='cuda', requires_grad=True)
        Si = torch.tensor(Sbus_np.imag, dtype=torch.float64, device='cuda', requires_grad=True)

        yff_t = torch.from_numpy(b["yff"].astype(np.complex128)).cuda()
        yft_t = torch.from_numpy(b["yft"].astype(np.complex128)).cuda()
        ytf_t = torch.from_numpy(b["ytf"].astype(np.complex128)).cuda()
        ytt_t = torch.from_numpy(b["ytt"].astype(np.complex128)).cuda()
        frm   = torch.from_numpy(b["branch_from"].astype(np.int64)).cuda()
        to_   = torch.from_numpy(b["branch_to"].astype(np.int64)).cuda()
        vn_t  = torch.from_numpy(b["vn_kv"].astype(np.float64)).cuda()
        sn_mva = float(b["sn_mva"])

        # Capture all non-differentiable args in a closure so gradcheck only
        # sees (Sr, Si) as inputs to perturb (scipy Ybus recurses otherwise).
        Ybus  = d["Ybus"]
        Vinit = d["v_init"].copy().astype(np.complex128)
        pv, pq_arr = d["pv"], d["pq"]
        slack, slack_weights = d["slack"], d["slack_weights"]

        def loss_fn(Sr, Si):
            V = solve_power_flow(
                Sr, Si, Ybus, Vinit, pv, pq_arr, slack, slack_weights, 10, 1e-8)
            flows = compute_flows(V, yff_t, yft_t, ytf_t, ytt_t,
                                  frm, to_, vn_t, sn_mva)
            return flows["i_or_a"].sum()

        result = torch.autograd.gradcheck(
            loss_fn, (Sr, Si),
            eps=1e-5,
            atol=5e-3,
            rtol=1e-2,
            nondet_tol=1e-6,   # CUDA scatter_add has float rounding non-determinism
        )
        assert result

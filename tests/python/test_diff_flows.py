# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
    yff_eff = np.concatenate([lines.get_yac_eff_11().copy(), trafos.get_yac_eff_11()])
    yft_eff = np.concatenate([lines.get_yac_eff_12().copy(), trafos.get_yac_eff_12()])
    ytf_eff = np.concatenate([lines.get_yac_eff_21().copy(), trafos.get_yac_eff_21()])
    ytt_eff = np.concatenate([lines.get_yac_eff_22().copy(), trafos.get_yac_eff_22()])

    return {
        "branch_from": branch_from,
        "branch_to":   branch_to,
        "yff_eff": yff_eff, "yft_eff": yft_eff, "ytf_eff": ytf_eff, "ytt_eff": ytt_eff,
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
        yff_eff_t = torch.from_numpy(b["yff_eff"].astype(np.complex128 if not _is_fp32() else np.complex64)).cuda()
        yft_eff_t = torch.from_numpy(b["yft_eff"].astype(np.complex128 if not _is_fp32() else np.complex64)).cuda()
        ytf_eff_t = torch.from_numpy(b["ytf_eff"].astype(np.complex128 if not _is_fp32() else np.complex64)).cuda()
        ytt_eff_t = torch.from_numpy(b["ytt_eff"].astype(np.complex128 if not _is_fp32() else np.complex64)).cuda()
        frm   = torch.from_numpy(b["branch_from"].astype(np.int64)).cuda()
        to_   = torch.from_numpy(b["branch_to"].astype(np.int64)).cuda()
        vn_t  = torch.from_numpy(b["vn_kv"].astype(np.float64 if not _is_fp32() else np.float32)).cuda()

        flows = compute_flows(V_t, yff_eff_t, yft_eff_t, ytf_eff_t, ytt_eff_t,
                              frm, to_, vn_t, b["sn_mva"])

        # Oracle (always FP64)
        or_a_ref, ex_a_ref = compute_branch_flows_cpu(
            V_ref, b["branch_from"], b["branch_to"],
            b["yff_eff"], b["yft_eff"], b["ytf_eff"], b["ytt_eff"],
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
            torch.from_numpy(b["yff_eff"]).to(cdtype).cuda(),
            torch.from_numpy(b["yft_eff"]).to(cdtype).cuda(),
            torch.from_numpy(b["ytf_eff"]).to(cdtype).cuda(),
            torch.from_numpy(b["ytt_eff"]).to(cdtype).cuda(),
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

        yff_eff_t = torch.from_numpy(b["yff_eff"].astype(np.complex128)).cuda()
        yft_eff_t = torch.from_numpy(b["yft_eff"].astype(np.complex128)).cuda()
        ytf_eff_t = torch.from_numpy(b["ytf_eff"].astype(np.complex128)).cuda()
        ytt_eff_t = torch.from_numpy(b["ytt_eff"].astype(np.complex128)).cuda()
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
            flows = compute_flows(V, yff_eff_t, yft_eff_t, ytf_eff_t, ytt_eff_t,
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


class TestHalfOpenEndpoint:

    def test_minus_one_endpoint_matches_the_kernel_convention(self):
        """A side at bus -1 (Kron-reduced half-open end) must not index V[-1].

        Same convention as compute_branch_flows_kernel: that side has V = 0
        and no terminal current, and the base current uses the live
        endpoint's nominal voltage. The last bus is given a different voltage
        and nominal kV so that reading it by mistake changes every output.
        """
        from gpusim2grid.differentiable import compute_flows

        cdt, rdt, dev = torch.complex128, torch.float64, "cuda"
        V = torch.tensor([[1.0 + 0.0j, 0.98 - 0.05j, 1.02 + 0.01j, 0.7 + 0.3j],
                          [1.0 + 0.0j, 0.97 - 0.06j, 1.01 + 0.02j, 0.6 + 0.2j]],
                         dtype=cdt, device=dev, requires_grad=True)
        vn_kv = torch.tensor([138.0, 138.0, 20.0, 400.0], dtype=rdt, device=dev)
        # branch 0: regular (0 -> 1); branch 1: open "to" end (2 -> -1);
        # branch 2: open "from" end (-1 -> 1)
        b_from = torch.tensor([0, 2, -1], device=dev)
        b_to = torch.tensor([1, -1, 1], device=dev)
        yff = torch.tensor([1 - 5j, 0.0 + 0.02j, 7 - 7j], dtype=cdt, device=dev)
        yft = torch.tensor([-1 + 5j, 3 - 3j, 5 - 5j], dtype=cdt, device=dev)
        ytf = torch.tensor([-1 + 5j, 4 - 4j, 6 - 6j], dtype=cdt, device=dev)
        ytt = torch.tensor([1 - 5j, 8 - 8j, 0.0 + 0.03j], dtype=cdt, device=dev)
        sn_mva = 100.0

        out = compute_flows(V, yff, yft, ytf, ytt, b_from, b_to, vn_kv, sn_mva)

        Vd = V.detach()
        zero = torch.zeros(2, dtype=cdt, device=dev)
        I_or = torch.stack([yff[0] * Vd[:, 0] + yft[0] * Vd[:, 1],
                            yff[1] * Vd[:, 2],
                            zero], dim=1)
        I_ex = torch.stack([ytf[0] * Vd[:, 0] + ytt[0] * Vd[:, 1],
                            zero,
                            ytt[2] * Vd[:, 1]], dim=1)
        V_or = torch.stack([Vd[:, 0], Vd[:, 2], zero], dim=1)
        V_ex = torch.stack([Vd[:, 1], zero, Vd[:, 1]], dim=1)
        base = sn_mva * 1e6 / (np.sqrt(3.0) * vn_kv[[0, 2, 1]] * 1e3)
        S_or = V_or * I_or.conj()
        S_ex = V_ex * I_ex.conj()
        torch.testing.assert_close(out["i_or_a"], I_or.abs() * base)
        torch.testing.assert_close(out["i_ex_a"], I_ex.abs() * base)
        torch.testing.assert_close(out["p_or_mw"], S_or.real * sn_mva)
        torch.testing.assert_close(out["q_or_mvar"], S_or.imag * sn_mva)
        torch.testing.assert_close(out["p_ex_mw"], S_ex.real * sn_mva)
        torch.testing.assert_close(out["q_ex_mvar"], S_ex.imag * sn_mva)

        # The -1 side contributes nothing, and nothing reaches the last bus
        # (only branch 1's live end, bus 2, and bus 1 / bus 0 are read).
        sum(v.sum() for v in out.values()).backward()
        assert torch.isfinite(torch.view_as_real(V.grad)).all()
        assert torch.all(V.grad[:, 3] == 0)

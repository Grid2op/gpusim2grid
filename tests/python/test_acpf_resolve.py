# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""AcPfNrSession: lazy adjoint and ``resolve()``.

* Jᵀ (its structure, cuDSS ANALYSIS and FACTORIZATION) is built by the first
  ``solve_JT_dlpack``, never by the constructor.
* ``resolve(Vinit, Sbus)`` reruns the Newton loop on the session's own
  handles / ANALYSIS and must give what a fresh ``presolved_v=False`` session
  on the same inputs gives -- voltages, iteration count and adjoint -- also
  on the augmented systems (distributed slack, HVDC droop, VoltageControl),
  whose running state resolve() has to restart.
"""
import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="augmented systems need the lightsim2grid C++ bridge")


def _bare_session(d, Sbus, presolved_v=False, max_iter=20, tol=1e-10):
    from gpusim2grid._gpusim2grid import AcPfNrSession
    return AcPfNrSession(
        d["Ybus"], d["v_init"].copy(), Sbus,
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        max_iter, tol, presolved_v=presolved_v)


def _scaled_sbus(Sbus, factor):
    """Every injection scaled: a different operating point, same structure."""
    return np.ascontiguousarray(Sbus * factor, dtype=complex)


def _solve_JT(session, rhs):
    torch = pytest.importorskip("torch")
    from gpusim2grid._gpusim2grid import is_fp32
    dtype = torch.float32 if is_fp32 else torch.float64
    rhs_t = torch.as_tensor(rhs, dtype=dtype, device="cuda").contiguous()
    lam = torch.from_dlpack(session.solve_JT_dlpack(rhs_t.__dlpack__())).clone()
    return lam.cpu().numpy().astype(np.float64)


def _rhs(session):
    return np.random.default_rng(0).standard_normal(session.dim_J)


# ---------------------------------------------------------------------------
# Lazy adjoint
# ---------------------------------------------------------------------------

def test_adjoint_is_built_by_the_first_solve_JT(ieee14_base_case, solver_atol):
    d = ieee14_base_case
    sess = _bare_session(d, d["Sbus"])
    assert sess.timings.t_prepare_jt_ms == 0.0   # forward only: nothing paid

    rhs = _rhs(sess)
    lam1 = _solve_JT(sess, rhs)
    t_first = sess.timings.t_prepare_jt_ms
    assert t_first > 0.0

    lam2 = _solve_JT(sess, rhs)                  # factors reused as they are
    assert sess.timings.t_prepare_jt_ms == t_first
    # cuDSS's SOLVE is not bit-reproducible (the first solve on fresh factors
    # differs from the next ones in the last bits): compare to rounding
    np.testing.assert_allclose(lam1, lam2, rtol=0, atol=1e-3 * solver_atol)


def test_lazy_adjoint_is_the_transpose_solve(ieee14_base_case, solver_atol):
    """λ solves Jᵀλ = rhs for the J the session itself holds."""
    d = ieee14_base_case
    sess = _bare_session(d, d["Sbus"])
    indptr, indices, data = sess.get_J()
    from scipy.sparse import csr_matrix
    J = csr_matrix((data, indices, indptr), shape=(sess.dim_J, sess.dim_J))
    rhs = _rhs(sess)
    lam = _solve_JT(sess, rhs)
    np.testing.assert_allclose(J.T @ lam, rhs, atol=solver_atol)


# ---------------------------------------------------------------------------
# resolve(), bare system
# ---------------------------------------------------------------------------

def test_resolve_matches_a_fresh_session(ieee14_base_case, solver_atol):
    d = ieee14_base_case
    S2 = _scaled_sbus(d["Sbus"], 1.1)

    sess = _bare_session(d, d["Sbus"])
    t = sess.resolve(d["v_init"].copy(), S2)
    ref = _bare_session(d, S2)

    assert t.converged and ref.timings.converged
    assert t.nb_iter == ref.timings.nb_iter
    np.testing.assert_allclose(sess.get_v(), ref.get_v(), rtol=0, atol=1e-3 * solver_atol)
    # nothing one-time is paid again, and resolve's timings became `timings`
    assert t.t_analyze_ms == 0.0 and t.t_context_init_ms == 0.0
    assert sess.timings.nb_iter == t.nb_iter


def test_resolve_moves_the_adjoint(ieee14_base_case, solver_atol):
    """A solve_JT before resolve() must not leave the old factors in place."""
    d = ieee14_base_case
    S2 = _scaled_sbus(d["Sbus"], 1.1)
    sess = _bare_session(d, d["Sbus"])
    rhs = _rhs(sess)
    lam_before = _solve_JT(sess, rhs)

    sess.resolve(d["v_init"].copy(), S2)
    lam_after = _solve_JT(sess, rhs)
    lam_ref = _solve_JT(_bare_session(d, S2), rhs)

    np.testing.assert_allclose(lam_after, lam_ref, rtol=0, atol=1e-3 * solver_atol)
    assert np.max(np.abs(lam_after - lam_before)) > 1e3 * solver_atol


def test_resolve_keeps_sbus_by_default(ieee14_base_case, solver_atol):
    """From a presolved session: resolve(V) without Sbus re-solves the same
    case, back to lightsim2grid's voltages; also from another start."""
    d = ieee14_base_case
    sess = _bare_session(d, d["Sbus"], presolved_v=False)
    S2 = _scaled_sbus(d["Sbus"], 1.1)
    sess.resolve(d["v_init"].copy(), S2)          # elsewhere first
    t = sess.resolve(d["v_init"].copy(), d["Sbus"].copy())
    assert t.converged
    np.testing.assert_allclose(sess.get_v(), d["v_ref"], atol=10 * solver_atol)

    flat = np.abs(d["v_ref"]).astype(complex)    # set-points kept, angles 0
    t = sess.resolve(flat)                        # Sbus kept
    assert t.converged
    np.testing.assert_allclose(sess.get_v(), d["v_ref"], atol=10 * solver_atol)


def test_resolve_dlpack_view_follows(ieee14_base_case):
    torch = pytest.importorskip("torch")
    d = ieee14_base_case
    sess = _bare_session(d, d["Sbus"])
    view = torch.from_dlpack(sess.v_dlpack())
    sess.resolve(d["v_init"].copy(), _scaled_sbus(d["Sbus"], 1.1))
    np.testing.assert_array_equal(view.cpu().numpy(), sess.get_v())


def test_resolve_rejects_wrong_sizes(ieee14_base_case):
    d = ieee14_base_case
    sess = _bare_session(d, d["Sbus"])
    with pytest.raises(RuntimeError, match="Vinit"):
        sess.resolve(np.ones(d["n_bus"] + 1, dtype=complex))
    with pytest.raises(RuntimeError, match="Sbus"):
        sess.resolve(d["v_init"].copy(), np.ones(d["n_bus"] - 1, dtype=complex))


def test_resolve_honours_max_iter(ieee14_base_case):
    d = ieee14_base_case
    sess = _bare_session(d, d["Sbus"])
    t = sess.resolve(np.abs(d["v_ref"]).astype(complex), max_iter=1)
    assert t.nb_iter == 1 and not t.converged


# ---------------------------------------------------------------------------
# resolve(), augmented systems (bridge)
# ---------------------------------------------------------------------------

def _augmented_grids():
    from test_augmented_features import (
        _solved_multislack_grid, _solved_hvdc_droop_grid, _solved_shared_bus_vc_grid)
    return {
        "multislack": _solved_multislack_grid,
        "hvdc_droop": _solved_hvdc_droop_grid,
        "shared_bus_vc": _solved_shared_bus_vc_grid,
    }


@needs_bridge
@pytest.mark.parametrize("kind", ["multislack", "hvdc_droop", "shared_bus_vc"])
def test_resolve_augmented_matches_a_fresh_session(kind, solver_atol):
    from gpusim2grid import AcPfGPU
    from gpusim2grid._gpusim2grid import _make_acpf_session_from_lsgrid_with_sbus

    grid = _augmented_grids()[kind]()
    V0 = grid.get_V_solver()
    S = grid.get_Sbus_solver()
    S2 = _scaled_sbus(S, 1.05)
    max_iter, tol = 30, 1e-10

    ac = AcPfGPU(grid, scaling_max_voltage_change=False)   # presolved at V0
    assert ac.session.dim_J > len(grid.get_pv()) + 2 * len(grid.get_pq())

    # 1. the grid's own case from elsewhere: back to lightsim2grid's solution
    #    (the slack / controller state restarted, not left at ground truth)
    t = ac.resolve(np.abs(V0).astype(complex), max_iter=max_iter, tol=tol)
    assert t.converged
    np.testing.assert_allclose(ac.solve(), V0, atol=10 * solver_atol)

    # 2. a new Sbus: what a fresh bridge session on it gives
    t = ac.resolve(V0, S2, max_iter=max_iter, tol=tol)
    ref = _make_acpf_session_from_lsgrid_with_sbus(grid, S2, max_iter, tol)
    assert t.converged and ref.timings.converged
    assert t.nb_iter == ref.timings.nb_iter
    np.testing.assert_allclose(ac.solve(), ref.get_v(), rtol=0, atol=1e-3 * solver_atol)
    rhs = _rhs(ref)
    np.testing.assert_allclose(_solve_JT(ac.session, rhs), _solve_JT(ref, rhs),
                               rtol=0, atol=1e-3 * solver_atol)


@pytest.mark.parametrize("use_bridge", [
    pytest.param(True, marks=needs_bridge), False])
def test_ac_pf_reuses_the_session(ieee14_base_case, solver_atol, use_bridge):
    """AcPfGPU.ac_pf runs on the instance's own session (no new cuDSS ANALYSIS)
    and, like lightsim2grid's ac_pf, lands on the grid's solution from a flat
    start (the set-points are written into |V| at the PV / slack buses)."""
    from gpusim2grid import AcPfGPU
    grid = ieee14_base_case["grid"]
    ac = AcPfGPU(grid, use_bridge=use_bridge)
    session = ac.session
    V = ac.ac_pf(np.ones(len(grid.get_V_solver()), dtype=complex), 20, 1e-10)
    assert ac.session is session
    assert ac.timings.t_analyze_ms == 0.0
    np.testing.assert_allclose(V, grid.get_V_solver(), atol=10 * solver_atol)
    assert ac.ac_pf(np.ones(len(V), dtype=complex), 1, 1e-10).shape == (0,)


def test_ac_pf_tuple_mode(ieee14_base_case, solver_atol):
    """No lightsim2grid grid needed any more: the set-points are the tuple's."""
    from gpusim2grid import AcPfGPU
    d = ieee14_base_case
    ac = AcPfGPU((d["Ybus"], d["v_ref"].copy(), d["Sbus"], d["slack"],
                  d["slack_weights"], d["pv"], d["pq"]))
    V = ac.ac_pf(np.ones(d["n_bus"], dtype=complex), 20, 1e-10)
    np.testing.assert_allclose(V, d["v_ref"], atol=10 * solver_atol)

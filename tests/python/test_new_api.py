"""Tests for the lightsim2grid-companion API: ContingencyAnalysisGPU,
InjectionSweepGPU, AcPfGPU.

These exercise the grid-object constructors that reuse lightsim2grid's CPU
base-case solve. Reference values come from lightsim2grid (ContingencyAnalysisCPP
and the AC reference voltage) via the shared conftest fixtures.
"""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu


# ---------------------------------------------------------------------------
# ContingencyAnalysisGPU
# ---------------------------------------------------------------------------

@requires_gpu
def test_contingency_gpu_matches_reference(
        ieee14_grid, ieee14_cpu_n1_reference, solver_atol):
    """ContingencyAnalysisGPU(grid) voltages match the CPU N-1 reference."""
    from gpusim2grid import ContingencyAnalysisGPU

    V_ref, n_lines, n_trafos = ieee14_cpu_n1_reference
    n_ctg = n_lines + n_trafos

    ca = ContingencyAnalysisGPU(ieee14_grid, init_from_n_powerflow=True,
                                nb_iter=10)
    ca.add_contingencies_by_branch_id([[c] for c in range(n_ctg)])
    ca.compute(batch_size=64)

    n_bus = ca.n_bus
    V_gpu = ca.V_results.to_numpy().reshape(n_ctg, n_bus)
    res = ca.last_residuals()

    # lightsim2grid uses a double-bus topology; compare the first n_bus.
    V_ref_main = V_ref[:, :n_bus]

    for c in range(n_ctg):
        if not np.isfinite(res[c]):
            continue  # disconnecting contingency — skipped (NaN) by design
        np.testing.assert_allclose(
            np.abs(V_gpu[c]), np.abs(V_ref_main[c]),
            atol=10 * solver_atol,
            err_msg=f"voltage magnitude mismatch for contingency {c}")


@requires_gpu
def test_init_from_n_powerflow_true_vs_false(ieee14_grid, solver_atol):
    """Both seeding modes converge to the same contingency voltages."""
    from gpusim2grid import ContingencyAnalysisGPU

    ctgs = [[0], [1], [5]]

    ca_true = ContingencyAnalysisGPU(ieee14_grid, init_from_n_powerflow=True,
                                     nb_iter=10)
    ca_true.add_contingencies_by_branch_id(ctgs)
    ca_true.compute(batch_size=8)
    V_true = ca_true.V_results.to_numpy()

    ca_false = ContingencyAnalysisGPU(ieee14_grid, init_from_n_powerflow=False,
                                      nb_iter=10, max_iter_base=10)
    ca_false.add_contingencies_by_branch_id(ctgs)
    ca_false.compute(batch_size=8)
    V_false = ca_false.V_results.to_numpy()

    np.testing.assert_allclose(V_true, V_false, atol=10 * solver_atol)


@requires_gpu
def test_contingency_last_residuals_before_compute_raises(ieee14_grid):
    from gpusim2grid import ContingencyAnalysisGPU

    ca = ContingencyAnalysisGPU(ieee14_grid)
    with pytest.raises(RuntimeError):
        ca.last_residuals()


@requires_gpu
def test_zero_copy_bridge_available():
    """The C++ zero-copy lightsim2grid bridge is compiled in."""
    from gpusim2grid import _gpusim2grid as cpp
    assert getattr(cpp, "have_ls2g_bridge", False), (
        "bridge not compiled — build was not linked against lightsim2grid_core")


@requires_gpu
def test_bridge_vs_python_extraction_agree(ieee14_grid, solver_atol):
    """The C++ zero-copy bridge and the Python extraction path agree."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid import _gpusim2grid as cpp

    if not getattr(cpp, "have_ls2g_bridge", False):
        pytest.skip("bridge not compiled")

    ctgs = [[0], [1], [2], [5], [0, 1]]

    ca_bridge = ContingencyAnalysisGPU(ieee14_grid, nb_iter=10, use_bridge=True)
    ca_bridge.add_contingencies_by_branch_id(ctgs)
    ca_bridge.compute(batch_size=8)
    V_bridge = ca_bridge.V_results.to_numpy()

    ca_py = ContingencyAnalysisGPU(ieee14_grid, nb_iter=10, use_bridge=False)
    ca_py.add_contingencies_by_branch_id(ctgs)
    ca_py.compute(batch_size=8)
    V_py = ca_py.V_results.to_numpy()

    np.testing.assert_allclose(V_bridge, V_py, atol=10 * solver_atol)


# ---------------------------------------------------------------------------
# InjectionSweepGPU
# ---------------------------------------------------------------------------

@requires_gpu
def test_injection_gpu_base_case_matches_reference(
        ieee14_grid, ieee14_base_case, solver_atol):
    """A single scenario equal to the base case reproduces the AC reference V."""
    from gpusim2grid import InjectionSweepGPU

    grid = ieee14_grid
    n_bus = ieee14_base_case["n_bus"]
    sn_mva = grid.get_sn_mva()

    # Base-case injections in physical units (P + jQ = Sbus * sn_mva).
    Sbus = ieee14_base_case["Sbus"]
    p_mw = (Sbus.real * sn_mva).reshape(1, n_bus)
    q_mvar = (Sbus.imag * sn_mva).reshape(1, n_bus)

    sweep = InjectionSweepGPU(grid, nb_iter=10)
    sweep.set_injections(p_mw, q_mvar, sn_mva)
    sweep.compute(batch_size=4)

    V = sweep.V_results.to_numpy().reshape(1, n_bus)
    res = sweep.last_residuals()

    assert res[0] < 10 * solver_atol
    np.testing.assert_allclose(
        np.abs(V[0]), np.abs(ieee14_base_case["v_ref"]), atol=10 * solver_atol)


# ---------------------------------------------------------------------------
# AcPfGPU
# ---------------------------------------------------------------------------

@requires_gpu
def test_acpf_gpu_matches_reference(ieee14_grid, ieee14_base_case, solver_atol):
    from gpusim2grid import AcPfGPU

    ac = AcPfGPU(ieee14_grid)
    V = ac.solve()
    np.testing.assert_allclose(
        np.abs(V), np.abs(ieee14_base_case["v_ref"]), atol=10 * solver_atol)


# ---------------------------------------------------------------------------
# Precision validation (no GPU work needed beyond the compiled flag)
# ---------------------------------------------------------------------------

@requires_gpu
def test_precision_mismatch_raises(ieee14_grid):
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid._gpusim2grid import is_fp32

    wrong = "fp64" if is_fp32 else "fp32"
    with pytest.raises(ValueError):
        ContingencyAnalysisGPU(ieee14_grid, precision=wrong)


@requires_gpu
def test_precision_matching_ok(ieee14_grid):
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid._gpusim2grid import is_fp32

    right = "fp32" if is_fp32 else "fp64"
    ca = ContingencyAnalysisGPU(ieee14_grid, precision=right)  # must not raise
    assert ca.n_bus > 0


# ---------------------------------------------------------------------------
# DLPack round-trip (torch-guarded)
# ---------------------------------------------------------------------------

@requires_gpu
def test_dlpack_round_trip(ieee14_grid):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")

    from gpusim2grid import ContingencyAnalysisGPU

    ca = ContingencyAnalysisGPU(ieee14_grid, nb_iter=4)
    ca.add_contingencies_by_branch_id([[0], [1]])
    capsule = ca.compute(batch_size=2)

    t = torch.from_dlpack(capsule)
    assert tuple(t.shape) == (2, ca.n_bus)
    assert t.is_cuda


# ---------------------------------------------------------------------------
# Presolved-V fast path (init_from_n_powerflow skips the GPU NR loop entirely)
# ---------------------------------------------------------------------------

@requires_gpu
def test_acpf_gpu_presolved_v_bit_identical(ieee14_grid, ieee14_base_case):
    """AcPfGPU(init_from_n_powerflow=True) never perturbs V: bit-identical to
    lightsim2grid's own converged voltage."""
    from gpusim2grid import AcPfGPU

    grid = ieee14_grid
    V_cpu = grid.get_V_solver()

    ac = AcPfGPU(grid, init_from_n_powerflow=True)
    V = ac.solve()

    assert np.array_equal(V, V_cpu), "presolved V must be bit-identical to V0"
    assert ac.timings.nb_iter == 0
    assert ac.timings.converged is True


@requires_gpu
def test_acpf_gpu_presolved_v_false_matches_reference(
        ieee14_grid, ieee14_base_case, solver_atol):
    """init_from_n_powerflow=False still converges to the same voltage (it just
    doesn't skip the GPU NR loop)."""
    from gpusim2grid import AcPfGPU

    ac = AcPfGPU(ieee14_grid, init_from_n_powerflow=False, max_iter=10, tol=1e-8)
    V = ac.solve()
    np.testing.assert_allclose(
        np.abs(V), np.abs(ieee14_base_case["v_ref"]), atol=10 * solver_atol)


def _case14_grid_and_poor_seed():
    """Fresh (not the shared session-scoped fixture -- this test mutates the
    grid's algo config and solver-side V) case14 grid, its AC reference
    voltage from a flat start, and a deliberately rough seed (magnitude 0.85,
    angles perturbed +/-0.4 rad) far enough from the solution that an
    undamped GPU Newton step genuinely fails to converge on it (verified:
    still hasn't converged after 200 iterations, error growing rather than
    shrinking -- not just "needs more iterations")."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    grid = init_from_pandapower(pn.case14())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n_bus = grid.get_bus_vn_kv().shape[0]

    v_flat = np.ones(n_bus, dtype=complex)
    v_ref = grid.ac_pf(v_flat.copy(), 20, 1e-10)
    assert v_ref.shape[0] == n_bus, "reference AC solve diverged"

    rng = np.random.RandomState(0)
    v_poor = (0.85 * np.exp(1j * rng.uniform(-0.4, 0.4, n_bus))).astype(complex)
    return grid, v_ref, v_poor


def _seed_v_solver(grid, v_seed, tol):
    """Set get_V_solver() to v_seed untouched (0 CPU Newton iterations), so
    the GPU bridge (which always reads V0 from get_V_solver()) sees the exact
    rough seed instead of a CPU-converged answer. See AcPfGPU.ac_pf's own doc
    for why this trick works."""
    grid.ac_pf(v_seed.copy(), 0, tol)


@requires_gpu
def test_acpf_gpu_step_scaling_recovers_convergence(solver_atol):
    """scaling_max_voltage_change=True (MaxVoltageChangeScalingPolicy, mirrors
    lightsim2grid's own damping) lets the GPU's own iterative NR loop
    (init_from_n_powerflow=False) converge to the correct answer from a rough
    seed where the undamped full-Newton-step GPU loop genuinely fails --
    exactly the failure mode observed on real RTE grids with grouped
    VoltageControl (see the vc_shared_bus_qcol_bug / DC-seed sweep
    investigation): an undamped step can converge onto a different root, or
    not converge at all, when started far from the solution."""
    from gpusim2grid import AcPfGPU

    grid, v_ref, v_poor = _case14_grid_and_poor_seed()

    _seed_v_solver(grid, v_poor, 1e-10)
    ac_unscaled = AcPfGPU(grid, init_from_n_powerflow=False, max_iter=50, tol=1e-10,
                          scaling_max_voltage_change=False)
    V_unscaled = ac_unscaled.solve()
    assert not ac_unscaled.timings.converged, "test seed should defeat the undamped GPU loop"
    assert not np.allclose(np.abs(V_unscaled), np.abs(v_ref), atol=1e-3)

    _seed_v_solver(grid, v_poor, 1e-10)
    ac_scaled = AcPfGPU(grid, init_from_n_powerflow=False, max_iter=50, tol=1e-10,
                        scaling_max_voltage_change=True, max_dVa=0.02, max_dVm=0.01)
    V_scaled = ac_scaled.solve()
    assert ac_scaled.timings.converged
    np.testing.assert_allclose(
        np.abs(V_scaled), np.abs(v_ref), atol=10 * solver_atol)


@requires_gpu
def test_acpf_gpu_step_scaling_inherits_grid_config(solver_atol):
    """scaling_max_voltage_change=None (the default) mirrors whatever the
    lightsim2grid grid's OWN get_ac_algo_config() is set to -- opt-in by
    inheritance, not a separate gpusim2grid-level default. Explicitly forcing
    it False overrides that inherited config back off, even on the exact same
    grid/seed."""
    from lightsim2grid.lightsim2grid_cpp import ScalingPolicyType
    from gpusim2grid import AcPfGPU

    grid, v_ref, v_poor = _case14_grid_and_poor_seed()

    cfg = grid.get_ac_algo_config()
    ip = list(cfg.int_params)
    ip[0] = int(ScalingPolicyType.MaxVoltageChange)
    cfg.int_params = ip
    rp = list(cfg.real_params)
    rp[0] = 0.02  # max_dVa
    rp[1] = 0.01  # max_dVm
    cfg.real_params = rp
    grid.set_ac_algo_config(cfg)

    _seed_v_solver(grid, v_poor, 1e-10)
    ac_default = AcPfGPU(grid, init_from_n_powerflow=False, max_iter=100, tol=1e-10)
    V_default = ac_default.solve()
    assert ac_default.timings.converged, "should inherit the grid's own MaxVoltageChange config"
    np.testing.assert_allclose(
        np.abs(V_default), np.abs(v_ref), atol=10 * solver_atol)

    _seed_v_solver(grid, v_poor, 1e-10)
    ac_forced_off = AcPfGPU(grid, init_from_n_powerflow=False, max_iter=100, tol=1e-10,
                            scaling_max_voltage_change=False)
    ac_forced_off.solve()
    assert not ac_forced_off.timings.converged, (
        "explicit override should take priority over the grid's own config")


@requires_gpu
def test_contingency_gpu_step_scaling_recovers_convergence():
    """Same step-scaling mechanism as test_acpf_gpu_step_scaling_recovers_
    convergence, but through ContingencyAnalysisGPU's BATCH driver -- each
    contingency in the batch gets its own alpha from its own max|dtheta|/
    max|dvm|, not one shared across the chunk (see BatchPfDriver's own doc).
    Without scaling, the batch NR loop from the rough seed blows up (residuals
    in the tens to hundreds); with it, every contingency converges to machine
    precision."""
    from gpusim2grid import ContingencyAnalysisGPU

    grid, _, v_poor = _case14_grid_and_poor_seed()
    n_lines = min(5, len(grid.get_lines()))

    _seed_v_solver(grid, v_poor, 1e-10)
    ca_unscaled = ContingencyAnalysisGPU(
        grid, init_from_n_powerflow=False, nb_iter=50, max_iter_base=1,
        scaling_max_voltage_change=False)
    ca_unscaled.add_contingencies_by_branch_id([[c] for c in range(n_lines)])
    ca_unscaled.compute(batch_size=8)
    assert np.all(ca_unscaled.last_residuals() > 1.0), (
        "test seed should defeat the undamped batch NR loop")

    _seed_v_solver(grid, v_poor, 1e-10)
    ca_scaled = ContingencyAnalysisGPU(
        grid, init_from_n_powerflow=False, nb_iter=100, max_iter_base=1,
        scaling_max_voltage_change=True, max_dVa=0.02, max_dVm=0.01)
    ca_scaled.add_contingencies_by_branch_id([[c] for c in range(n_lines)])
    ca_scaled.compute(batch_size=8)
    np.testing.assert_array_less(ca_scaled.last_residuals(), 1e-8)


@requires_gpu
def test_injection_sweep_gpu_step_scaling_recovers_convergence():
    """Same as test_contingency_gpu_step_scaling_recovers_convergence, through
    InjectionSweepGPU's batch driver (each scenario gets its own alpha)."""
    from gpusim2grid import InjectionSweepGPU

    grid, _, v_poor = _case14_grid_and_poor_seed()
    n_bus = grid.get_bus_vn_kv().shape[0]
    sn = grid.get_sn_mva()
    Sbus = grid.get_Sbus_solver()
    p_mw = np.tile(Sbus.real * sn, (3, 1))
    q_mvar = np.tile(Sbus.imag * sn, (3, 1))

    _seed_v_solver(grid, v_poor, 1e-10)
    is_unscaled = InjectionSweepGPU(
        grid, init_from_n_powerflow=False, nb_iter=50, max_iter_base=1,
        scaling_max_voltage_change=False)
    is_unscaled.set_injections(p_mw, q_mvar, sn)
    is_unscaled.compute(batch_size=4)
    assert np.all(is_unscaled.last_residuals() > 1.0), (
        "test seed should defeat the undamped batch NR loop")

    _seed_v_solver(grid, v_poor, 1e-10)
    is_scaled = InjectionSweepGPU(
        grid, init_from_n_powerflow=False, nb_iter=100, max_iter_base=1,
        scaling_max_voltage_change=True, max_dVa=0.02, max_dVm=0.01)
    is_scaled.set_injections(p_mw, q_mvar, sn)
    is_scaled.compute(batch_size=4)
    np.testing.assert_array_less(is_scaled.last_residuals(), 1e-8)


@requires_gpu
def test_acpf_gpu_presolved_v_raises_when_not_converged():
    """init_from_n_powerflow=True must fail loudly if V0 is not actually
    converged for the given Ybus/Sbus/ledger (stale grid, wrong tol_base, ...)."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    from gpusim2grid import AcPfGPU

    grid = init_from_pandapower(pn.case14())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n_bus = grid.get_bus_vn_kv().shape[0]
    # Force an early stop (1 iteration, huge tolerance) so V0 is not converged.
    grid.ac_pf(np.ones(n_bus, dtype=complex), 1, 1e10)

    with pytest.raises(RuntimeError):
        AcPfGPU(grid, init_from_n_powerflow=True)


@requires_gpu
def test_contingency_gpu_presolved_v_base_matches_cpu(ieee14_grid, ieee14_base_case):
    """The base case tiled across the contingency batch is bit-identical to V0
    (not perturbed by a needless NR correction), via the DLPack base-voltage
    export."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")

    from gpusim2grid import ContingencyAnalysisGPU

    grid = ieee14_grid
    V_cpu = grid.get_V_solver()

    ca = ContingencyAnalysisGPU(grid, init_from_n_powerflow=True, nb_iter=4)
    capsule = ca.solver.v_base_dlpack()
    V_base = torch.from_dlpack(capsule).cpu().numpy()

    np.testing.assert_array_equal(V_base, V_cpu)


@requires_gpu
def test_acpf_nr_session_presolved_v_low_level(ieee14_base_case):
    """The low-level (non-bridge, raw-array) AcPfNrSession also supports
    presolved_v, for parity between the bridge and Python-extraction paths."""
    from gpusim2grid._gpusim2grid import AcPfNrSession

    d = ieee14_base_case
    sess = AcPfNrSession(
        d["Ybus"], d["v_ref"], d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        10, 1e-8, -1, presolved_v=True)

    assert np.array_equal(sess.get_v(), d["v_ref"])
    assert sess.timings.nb_iter == 0
    assert sess.timings.converged is True


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

@requires_gpu
def test_backward_compat_solver_imports():
    """The original low-level wrappers remain importable and constructible."""
    from gpusim2grid.contingency_analysis import _ContingencyAnalysisSolver
    from gpusim2grid.injection_sweep import _InjectionSweepSolver
    from gpusim2grid.acpf_nr import AcPfNrSession  # noqa: F401

    assert _ContingencyAnalysisSolver is not None
    assert _InjectionSweepSolver is not None

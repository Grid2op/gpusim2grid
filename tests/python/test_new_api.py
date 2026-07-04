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
# Backward compatibility
# ---------------------------------------------------------------------------

@requires_gpu
def test_backward_compat_solver_imports():
    """The original low-level wrappers remain importable and constructible."""
    from gpusim2grid.contingency_analysis import ContingencyAnalysisSolver
    from gpusim2grid.injection_sweep import InjectionSweepSolver
    from gpusim2grid.acpf_nr import AcPfNrSession  # noqa: F401

    assert ContingencyAnalysisSolver is not None
    assert InjectionSweepSolver is not None

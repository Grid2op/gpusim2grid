# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``outer_loops_backend``: ``'host'`` (default, today's path) or ``'device'`` (CUDA kernels), and
the feature census that refuses ``'device'`` with ``NotImplementedError`` for the whole batch
while a feature is not ported (docs/outer_loops_on_device.md, §4.1-4.2).

The census and the refusals run before any GPU work: only the GPU-run tests need a device."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _outer_device_helpers import MAX_IT, TOL_MVA, case14_outer, needs_outer_loops

from conftest import requires_gpu

pytestmark = [needs_outer_loops]


@pytest.fixture(scope="module")
def grid() -> Any:
    return case14_outer()


def _facades() -> list[Any]:
    from gpusim2grid import ContingencyAnalysisGPU, InjectionSweepGPU, ScenarioSweepGPU

    return [ContingencyAnalysisGPU, ScenarioSweepGPU, InjectionSweepGPU]


# ---- the census -----------------------------------------------------------------------------


def test_census_lists_every_feature_with_its_count(grid: Any):
    from gpusim2grid import outer_device_census

    census = outer_device_census(grid)
    assert census, "the census lists every feature, those the grid lacks with count 0"
    for name, count, supported in census:
        assert isinstance(name, str) and isinstance(count, int) and isinstance(supported, bool)
        assert count >= 0
    counts: dict[str, int] = {name: count for name, count, _ in census}
    # NROuter_KLU's default loops on case14
    assert counts["loop DistributedSlack"] == 1
    assert counts["loop ReactiveLimits"] == 1
    assert counts["loop PhaseControl"] == 0
    assert counts["loop TransformerVoltageControl"] == 0
    assert counts["loop ShuntVoltageControl"] == 0
    # its generators hold their own buses: local controller buses, no voltage-control group
    assert counts["ReactiveLimits local controller buses"] > 0
    assert counts["ReactiveLimits voltage-control group buses (remote regulators)"] == 0
    assert counts["DistributedSlack units"] == len(grid.get_generators())
    assert counts["resumed rows (init_from_n_powerflow)"] == 0


def test_census_sees_the_resume_point(grid: Any):
    from gpusim2grid import outer_device_census

    census = outer_device_census(grid, init_from_n_powerflow=True)
    counts: dict[str, int] = {name: count for name, count, _ in census}
    assert counts["resumed rows (init_from_n_powerflow)"] == 1


# ---- the option -----------------------------------------------------------------------------


@pytest.mark.parametrize("facade", range(3))
def test_unknown_backend_is_refused(grid: Any, facade: int):
    with pytest.raises(ValueError, match="outer_loops_backend"):
        _facades()[facade](grid, outer_loops=True, outer_loops_backend="gpu")


@pytest.mark.parametrize("facade", range(3))
def test_device_needs_the_outer_loops(grid: Any, facade: int):
    with pytest.raises(ValueError, match="needs outer_loops=True"):
        _facades()[facade](grid, outer_loops_backend="device")


@pytest.fixture(scope="module")
def grid_all_loops() -> Any:
    """case14 with every loop lightsim2grid runs: the four default ones, PhaseControl,
    TransformerVoltageControl and ShuntVoltageControl (none with an element to act on: stable at once)"""
    from lightsim2grid.algorithm import PhaseControl, ShuntVoltageControl, TransformerVoltageControl

    g = case14_outer()
    for lp in (PhaseControl(), TransformerVoltageControl(), ShuntVoltageControl()):
        g.add_outer_loop(lp)
    g.change_algorithm("NROuter_KLU")
    assert g.ac_pf(np.full(g.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    return g


def test_census_marks_what_the_device_runs(grid: Any, grid_all_loops: Any):
    """every feature the census lists is one the device runs"""
    from gpusim2grid import outer_device_census

    for g in (grid, grid_all_loops):
        census = outer_device_census(g)
        assert all(ok for _, _, ok in census), [name for name, _, ok in census if not ok]
    counts = {name: count for name, count, _ in outer_device_census(grid_all_loops)}
    for loop in ("PhaseControl", "TransformerVoltageControl", "ShuntVoltageControl"):
        assert counts[f"loop {loop}"] == 1


@pytest.mark.parametrize("facade", range(3))
def test_device_refuses_the_whole_batch_naming_what_is_missing(
    grid: Any, facade: int, monkeypatch: pytest.MonkeyPatch
):
    """a feature the census marks unsupported refuses the whole batch at construction, naming it
    (none is left today: the census is patched to report one)"""
    import gpusim2grid._outer_loops as ol

    real = ol._cpp._outer_device_census

    def census(g: Any, resume: bool) -> list[Any]:
        return [*real(g, resume), ("a feature to come", 2, False)]

    monkeypatch.setattr(ol._cpp, "_outer_device_census", census)
    with pytest.raises(NotImplementedError) as err:
        _facades()[facade](grid, outer_loops=True, outer_loops_backend="device")
    msg = str(err.value)
    assert "a feature to come (2)" in msg
    assert "outer_loops_backend='host'" in msg
    # what the device runs is not listed
    assert "loop ReactiveLimits" not in msg


@pytest.mark.parametrize("facade", range(3))
def test_device_runs_every_loop(grid_all_loops: Any, facade: int):
    _facades()[facade](grid_all_loops, outer_loops=True, outer_loops_backend="device")


@requires_gpu
@pytest.mark.parametrize("facade", range(3))
def test_host_is_the_default(grid: Any, facade: int):
    obj = _facades()[facade](grid, outer_loops=True)
    assert obj.outer_loops_backend == "host"


@requires_gpu
def test_host_backend_is_todays_path(grid: Any):
    from gpusim2grid import ContingencyAnalysisGPU

    n_rows: int = 8

    def run(**kw: Any) -> tuple[np.ndarray, np.ndarray]:
        tol: float = TOL_MVA / grid.get_sn_mva()
        ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, tol=tol, **kw)
        ca.add_contingencies_by_branch_id([[i] for i in range(n_rows)])
        ca.compute(batch_size=3)
        return np.asarray(ca.get_row_status()), ca.V_results.to_numpy().reshape(n_rows, -1)

    st_default, v_default = run()
    st_host, v_host = run(outer_loops_backend="host")
    np.testing.assert_array_equal(st_host, st_default)
    converged = st_default == 0
    assert converged.sum() >= n_rows - 1
    # two sessions reproduce to rounding (batched GPU reductions), never bit for bit; a row
    # out of its iteration budget keeps whatever its last iterate was
    np.testing.assert_allclose(v_host[converged], v_default[converged], rtol=0, atol=1e-12, equal_nan=True)

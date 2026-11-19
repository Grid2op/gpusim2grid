# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""What a row starts from on ``outer_loops_backend='device'``: the rows copied on the device from a
template (a row changing nothing, loaded once through the host model) and the ones the host model
loads must start, and end, as on ``'host'``.

- a row whose only change is a load it trips is not plain on a TransformerVoltageControl grid: the
  loop freezes a bus at ``mis_bus - load Q``, the row's own load;
- the results of copied rows report the template's targets, not those of the first row the host
  model happened to load (a generator contingency first: a guard, the targets of such rows equal
  the template's today)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _outer_device_helpers import MAX_IT, TOL_MVA, case14_outer, needs_outer_loops

from conftest import requires_gpu

pytestmark = [needs_outer_loops, requires_gpu]

VTOL: float = 1e-9


def _same_rows(host: Any, dev: Any, n_rows: int, gen_tol: float = 1e-6) -> None:
    """status, iterations, decisions, every generator's target and the converged voltages"""
    st_h, st_d = np.asarray(host.get_row_status()), np.asarray(dev.get_row_status())
    np.testing.assert_array_equal(st_d, st_h)
    conv = st_h == 0
    v_h = host.V_results.to_numpy().reshape(n_rows, -1)
    v_d = dev.V_results.to_numpy().reshape(n_rows, -1)
    np.testing.assert_allclose(v_d[conv], v_h[conv], rtol=0, atol=VTOL, equal_nan=True)
    for k, (rh, rd) in enumerate(zip(host.get_outer_results(), dev.get_outer_results(), strict=True)):
        where = f"row {k}"
        assert (rd.status, rd.nb_outer_iterations, rd.nb_nr_iterations) == (
            rh.status,
            rh.nb_outer_iterations,
            rh.nb_nr_iterations,
        ), where
        assert list(rd.decisions) == list(rh.decisions), where
        assert sorted(rd.ratio_tap) == sorted(rh.ratio_tap), where
        h, d = dict(rh.gen_target_p), dict(rd.gen_target_p)
        assert sorted(d) == sorted(h), f"{where}: generators with a target"
        for g, mw in h.items():
            assert d[g] == pytest.approx(mw, rel=0, abs=gen_tol), f"{where}, generator {g}"


@pytest.mark.parametrize("init_from_n", [False, True])
def test_rows_tripping_a_load_on_transformer_voltage_control(init_from_n: bool):
    """TransformerVoltageControl freezes its low voltage generator buses at ``mis_bus - load Q``
    with the row's own loads (``device_tvc_image``'s ``tv_load``): a row tripping a load must load
    through the host model, not be copied from a template holding the grid's loads"""
    import test_outer_model_cpu as cpu

    from gpusim2grid import ContingencyAnalysisGPU

    grid = cpu._tvc_grid(
        "T4-9-1",
        12.6,
        0.1,
        reactive_limits=True,
        use_initial_tap_position=True,
        max_controlled_nominal_voltage=120.0,
    )
    assert grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, 1e-10).size > 0
    n_load = len(grid.get_loads())
    # one load per row: the first row loaded through the host is the template the next are copied from
    ctgs = [((), (), (load,), ()) for load in range(n_load)]

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=1e-10 / grid.get_sn_mva(),
            init_from_n_powerflow=init_from_n,
            handle_disconnected_grid=True,
            keep_outer_trace=True,
            outer_loops_backend=backend,
        )
        ca.tail_max_groups, ca.tail_group_size = 4, 1
        ca.add_contingencies(ctgs)
        ca.compute(batch_size=4)
        return ca

    host, dev = run("host"), run("device")
    assert any(r.ratio_tap for r in host.get_outer_results()), "TransformerVoltageControl moved no tap"
    _same_rows(host, dev, len(ctgs))


@pytest.mark.parametrize("distributed", [True, False])
@pytest.mark.parametrize("init_from_n", [False, True])
def test_copied_rows_report_the_template_targets(distributed: bool, init_from_n: bool):
    """the first batch is generator contingencies only (each loaded through the host model with
    targets of its own), the rows after it change nothing (copied on the device from the first of
    them): every row's targets are its own on both backends"""
    from gpusim2grid import ContingencyAnalysisGPU

    grid = case14_outer(distributed=distributed)
    n_gen = len(grid.get_generators())
    batch = 4
    # generators 1.. (not the slack): batch of them first, then plain rows (branch trips that island nothing)
    gen_rows = [((), (), (), (1 + g % (n_gen - 1),)) for g in range(batch)]
    plain_rows = [((b,), (), (), ()) for b in (0, 2, 3, 4, 5, 6)]
    ctgs = gen_rows + plain_rows

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            init_from_n_powerflow=init_from_n,
            handle_disconnected_grid=True,
            keep_outer_trace=True,
            outer_loops_backend=backend,
        )
        ca.add_contingencies(ctgs)
        ca.compute(batch_size=batch)
        return ca

    host, dev = run("host"), run("device")
    _same_rows(host, dev, len(ctgs))

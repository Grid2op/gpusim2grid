# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
test_warmup.py — gpusim2grid.warmup() pays CUDA/cuSPARSE/cuDSS init up front.

The contract is behavioural, not numeric: after warmup(), a freshly built
session must report a near-zero t_context_init_ms, because there is no
library/context initialization left to do. The absolute cost of the *first*
warm-up is not asserted -- it depends entirely on whether some earlier test in
the same process already touched CUDA.
"""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu


def test_warmup_returns_elapsed_ms_and_is_idempotent():
    from gpusim2grid import warmup

    first = warmup()
    assert isinstance(first, float)
    assert first >= 0.0

    # A second call short-circuits: the device is already warm, so it must not
    # repeat the (not free) cuSPARSE/cuDSS exercise the first call ran.
    second = warmup()
    assert second < 1.0


def test_warmup_rejects_out_of_range_device():
    from gpusim2grid import warmup

    with pytest.raises(RuntimeError, match="out of range"):
        warmup(device=9999)


def test_context_init_is_negligible_after_warmup(ieee14_base_case):
    """The whole point: no library init is charged to a post-warmup session."""
    from gpusim2grid import warmup
    from gpusim2grid._gpusim2grid import AcPfNrSession

    warmup()

    d = ieee14_base_case
    sess = AcPfNrSession(d["Ybus"], d["v_init"].copy(), d["Sbus"],
                         d["slack"], d["slack_weights"], d["pv"], d["pq"],
                         10, 1e-8, -1)
    sess.get_v()

    t = sess.timings
    assert t.t_context_init_ms < 5.0, (
        f"t_context_init_ms={t.t_context_init_ms:.3f}ms after warmup() -- "
        "library/context init should already be paid")
    # And it stays a strict sub-phase of t_init_ms, so t_host_to_device_ms
    # (t_init - build_J - context_init) cannot go negative.
    assert t.t_context_init_ms <= t.t_init_ms + 1e-9
    assert t.t_host_to_device_ms >= 0.0


def test_context_init_is_reported_separately_from_gpu_compute(
        ieee14_grid, ieee14_base_case):
    """t_context_init_ms is its own bucket, never folded into gpu_compute."""
    from gpusim2grid import warmup, ContingencyAnalysisGPU
    from conftest import branch_data_arrays

    warmup()

    d = ieee14_base_case
    branch_data, _, _ = branch_data_arrays(ieee14_grid)
    ca = ContingencyAnalysisGPU(
        (d["Ybus"], d["v_init"].copy(), d["Sbus"], d["slack"],
         d["slack_weights"], d["pv"], d["pq"]),
        nb_iter=3, init_from_n_powerflow=False)
    ca.set_branch_data(*branch_data)
    ca.add_contingencies_by_branch_id([[0], [1]])
    ca.compute(batch_size=2)

    t = ca.timings
    dd = t.to_dict()
    assert dd["context_init"]["total"] == pytest.approx(t.t_context_init_ms, abs=1e-9)
    assert t.t_gpu_compute_ms == pytest.approx(
        t.t_base_case_solve_only_ms + t.t_analysis_ms + t.t_chunks_total_wall_ms,
        abs=1e-9)
    assert np.isfinite(t.t_grand_total_ms)

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The rows' own branch blocks (``ContingencyAnalysisSession.set_row_branch_blocks``) are written
into the flow buffers of the rows the last run solved: a block of a row (or a branch) outside them
is refused with an error, never written past the device buffers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gpusim2grid._gpusim2grid import have_ls2g_bridge

from _topo_helpers import case14, make_ca
from conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.skipif(not have_ls2g_bridge, reason="needs the lightsim2grid bridge")]

BATCH: int = 4


def _block_args(rows: list[int], branches: list[int]) -> tuple[np.ndarray, ...]:
    n: int = len(rows)
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(branches, dtype=np.int32),
        np.tile(np.asarray([0, 1], dtype=np.int32), n),
        np.tile(np.asarray([1 - 10j, -1 + 10j, -1 + 10j, 1 - 10j], dtype=np.complex128), n),
        np.ones(2 * n),
    )


def _solved_session() -> Any:
    ca = make_ca(case14(), {}, batch_size=BATCH)
    ca.add_contingencies_by_branch_id([[3], [10]])
    ca.compute(batch_size=BATCH)
    return ca._inner._s


def test_a_block_of_a_row_the_last_run_did_not_solve_is_refused():
    """build_contingencies with more rows after the run: the flows are those of the run's rows, a
    block on a row past them would write past the flow buffers"""
    s = _solved_session()
    s.build_contingencies([[3], [10], [], [4], [5]])
    s.set_row_branch_blocks(*_block_args([4], [0]))
    with pytest.raises(RuntimeError, match="row 4"):
        s.compute_flows()


def test_a_block_of_a_branch_past_the_flow_buffers_is_refused():
    """the branch id is checked against the branches the flows have: branch data set after the
    blocks with fewer branches leaves the block's branch out of the flow buffers"""
    from gpusim2grid._ls2g_utils import extract_branch_data

    grid = case14()
    s = _solved_session()
    args = list(extract_branch_data(grid)[0])
    n_br: int = len(args[0])
    s.set_row_branch_blocks(*_block_args([1], [n_br - 1]))
    s.compute_flows()  # the last branch is one of them
    s.set_branch_data(*[np.asarray(a)[: n_br - 1] for a in args[:6]], *args[6:])
    with pytest.raises(RuntimeError, match=f"branch {n_br - 1}"):
        s.compute_flows()

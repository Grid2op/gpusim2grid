# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Host-side pieces of the batch solvers, checked through their test hooks: no GPU
needed, so these run on the GPU-less CI legs too.

- ``move_reference`` (the automatic reference slack) keeps lightsim2grid's ordering
  constraints valid: they name Vm columns, controller unknowns and custom rows, never
  the theta column / P rows it trades, and the moved ledger's are lightsim2grid's own
  for that reference. What ``linear_solver='own_lu'`` (which never pivots) and
  ``reordering_alg='ls2g_constrained'`` order with.
- ``slot_upload_ranges``: the contiguous slot ranges the outer loops' per-slot controls
  are uploaded by (``BatchPfDriver::_oc_write_controls``).
"""

import numpy as np
import pytest

from gpusim2grid import _gpusim2grid as _cpp

pytestmark = pytest.mark.skipif(not getattr(_cpp, "have_ls2g_bridge", False),
                                reason="needs the lightsim2grid C++ bridge")


# ------------------------------------------- ordering constraints, reference moved
def _held_grid_two_slacks(reference=None):
    """lightsim2grid's fixture for its held controllers (test_hold_frozen_regulators), held:
    the group of bus 2 has an active and a held controller, whose rows are switched by
    value -- the case lightsim2grid declares ordering constraints for, here on a plain
    Newton. Plus a generator on bus 3 sharing the slack with gen 0, so that the angle
    reference can move; `reference` forces it (lightsim2grid's own ledger for it)."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    from test_hold_frozen_regulators import LINES, STEP_UP, GENS, CAN_BE_PV
    gens = [list(g) for g in GENS] + [[3, 5., 1.02, 0., True, -500., 500., 3]]
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(8, 1, np.full(8, 138.), 0, 0)
    x = np.array([0.03 if k in STEP_UP else 0.08 for k in range(len(LINES))])
    grid.init_powerlines(np.full(len(LINES), 0.01), x, np.full(len(LINES), 0.02j),
                         np.array([a for a, _ in LINES]), np.array([b for _, b in LINES]))
    grid.init_loads(np.array([60., 50., 40.]), np.array([25., 20., 15.]), np.array([1, 2, 3]))
    grid.init_generators_full(np.array([g[1] for g in gens]), np.array([g[2] for g in gens]),
                              np.array([g[3] for g in gens]), [g[4] for g in gens],
                              np.array([g[5] for g in gens]), np.array([g[6] for g in gens]),
                              np.array([g[0] for g in gens]))
    for k, g in enumerate(gens):
        if g[7] != g[0]:
            grid.set_gen_regulated_bus(k, g[7])
    grid.set_gen_can_be_pv(np.array(list(CAN_BE_PV) + [False]))
    grid.add_gen_slackbus(0, 0.6)
    grid.add_gen_slackbus(len(gens) - 1, 0.4)
    grid.set_hold_frozen_regulators(True)
    if reference is not None:
        grid.set_reference_slack_bus(reference)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.full(8, 1.0 + 0j), 30, 1e-12).shape[0] > 0
    return grid


@pytest.mark.skipif(not getattr(_cpp, "have_ls2g_hold_frozen", False),
                    reason="needs a lightsim2grid with set_hold_frozen_regulators")
def test_moved_reference_keeps_lightsim2grids_ordering_constraints():
    """The ledger move_reference builds has lightsim2grid's own ordering constraints for
    that reference (and its skeleton and maps, as in
    test_moved_reference_is_lightsim2grids_own_ledger), on a grid that declares some: none
    names an index whose unknown or equation the move changes."""
    grid = _held_grid_two_slacks()
    base = _cpp._ledger_skeleton(grid)
    assert base["ord_constraints_read"] == 1
    named = set(base["ord_follow"]) | set(base["ord_lead"])
    assert len(base["ord_in_order"]) > 0 and named

    s2me = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    me2s = np.asarray(grid.id_me_to_ac_solver(), dtype=int)
    other = 3 if int(s2me[base["reference_bus"]]) == 0 else 0   # the other slack bus
    target = int(me2s[other])
    moved = _cpp._ledger_skeleton(grid, target)
    native = _cpp._ledger_skeleton(_held_grid_two_slacks(reference=other))
    assert native["reference_bus"] == moved["reference_bus"] == target != base["reference_bus"]
    for key in native:
        a, b = moved[key], native[key]
        if isinstance(a, int):
            assert a == b, key
        else:
            assert list(a) == list(b), key
    # what the move changes: the theta column handed over and the two P rows traded
    changed = {base["theta_col_of_bus"][target], base["p_row_of_bus"][target],
               base["p_row_of_bus"][base["reference_bus"]]}
    assert not (named & changed)


# ------------------------------------------------------------- slot upload ranges
@pytest.mark.parametrize("slots, expected", [
    ([], []),
    ([7], [(7, 7)]),
    ([7, 7, 7], [(7, 7)]),
    ([5, 1, 2, 3], [(1, 5)]),                        # unsorted, the gap at 4 bridged
    ([0, 9], [(0, 9)]),                              # 8 unlisted slots between: one range
    ([0, 10], [(0, 0), (10, 10)]),                   # 9: two
    ([40, 1, 41, 2, 100], [(1, 2), (40, 41), (100, 100)]),
])
def test_slot_upload_ranges(slots, expected):
    assert [tuple(r) for r in _cpp._slot_upload_ranges(slots)] == expected


def test_slot_upload_ranges_cover_the_list_in_few_ranges():
    """Every listed slot is in exactly one range, the ranges are increasing and disjoint,
    and a scattered list falls back to the one range spanning it."""
    rng = np.random.default_rng(0)
    for n_slots, n_listed in ((64, 5), (4096, 40), (4096, 2000), (10, 10)):
        slots = rng.choice(n_slots, size=n_listed, replace=False).tolist()
        ranges = [tuple(r) for r in _cpp._slot_upload_ranges(slots)]
        assert len(ranges) <= 32
        assert all(a <= b for a, b in ranges)
        assert all(ranges[k][1] < ranges[k + 1][0] for k in range(len(ranges) - 1))
        for s in slots:
            assert sum(a <= s <= b for a, b in ranges) == 1
        assert ranges[0][0] == min(slots) and ranges[-1][1] == max(slots)
    scattered = list(range(0, 4000, 20))   # 200 slots, 19 apart: more than 32 ranges
    assert [tuple(r) for r in _cpp._slot_upload_ranges(scattered)] == [(0, 3980)]
    assert [tuple(r) for r in _cpp._slot_upload_ranges(scattered, max_gap=19)] == [(0, 3980)]
    assert len(_cpp._slot_upload_ranges(scattered, max_gap=0, max_ranges=1000)) == 200

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``outer_loops=True`` on the batch facades: lightsim2grid's ``NROuter_*``
outer loops run per row (docs/dev_notes/continuous_batching.md, Part B; the
C++ side is ``outer/``). What the three facades check alike."""

from . import _gpusim2grid as _cpp


def _check_outer_loops_args(grid, use_bridge, scheduling, nb_iter_per_round, **unsupported):
    """Raise ValueError when the outer loops cannot run with these arguments.
    ``unsupported``: name -> value of each option they do not support yet (a
    truthy value raises).

    ``scheduling="continuous"`` and ``nb_iter_per_round=1`` are requirements by
    design, not missing features (neither is planned to change): a row the loops
    change is re-solved in its own slot while the other slots go on, which a
    chunk (every row the same ``nb_iter`` iterations) cannot do; and each inner
    solve must stop at the iteration lightsim2grid's does, so that the loops
    decide on the same state. The sessions check both again at every run, the
    two being mutable."""
    if isinstance(grid, (tuple, list)):
        raise ValueError("outer_loops=True needs a lightsim2grid grid (its loops), not an array tuple")
    if use_bridge is False or not getattr(_cpp, "have_ls2g_bridge", False):
        raise ValueError("outer_loops=True needs the compiled lightsim2grid bridge")
    if not getattr(_cpp, "have_ls2g_outer_loops", False):
        raise ValueError("outer_loops=True: gpusim2grid was built against a lightsim2grid "
                         "without outer loops (NROuter_*)")
    if scheduling != "continuous":
        raise ValueError("outer_loops=True runs on scheduling='continuous' only, by design (a row "
                         "the loops change is re-solved in its own slot while the others go on, "
                         "which a chunk cannot do); 'chunked' is not supported and is not planned to be")
    if nb_iter_per_round != 1:
        raise ValueError("outer_loops=True needs nb_iter_per_round=1, by design (each inner solve "
                         "must stop at the iteration lightsim2grid's does, so that the loops decide "
                         "on the same state); other values are not supported and are not planned to be")
    bad = [name for name, value in unsupported.items() if value]
    if bad:
        raise ValueError(f"outer_loops=True does not support {', '.join(bad)} yet")

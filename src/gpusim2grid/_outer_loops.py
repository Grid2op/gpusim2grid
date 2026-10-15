# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``outer_loops=True`` on the batch facades: lightsim2grid's ``NROuter_*``
outer loops run per row (docs/dev_notes/continuous_batching.md, Part B; the
C++ side is ``outer/``). What the three facades check alike."""

from . import _gpusim2grid as _cpp


def _check_outer_loops_args(grid, use_bridge, scheduling, **unsupported):
    """Raise ValueError when the outer loops cannot run with these arguments.
    ``unsupported``: name -> value of each option they do not support yet (a
    truthy value raises)."""
    if isinstance(grid, (tuple, list)):
        raise ValueError("outer_loops=True needs a lightsim2grid grid (its loops), not an array tuple")
    if use_bridge is False or not getattr(_cpp, "have_ls2g_bridge", False):
        raise ValueError("outer_loops=True needs the compiled lightsim2grid bridge")
    if not getattr(_cpp, "have_ls2g_outer_loops", False):
        raise ValueError("outer_loops=True: gpusim2grid was built against a lightsim2grid "
                         "without outer loops (NROuter_*)")
    if scheduling != "continuous":
        raise ValueError("outer_loops=True runs on scheduling='continuous' only")
    bad = [name for name, value in unsupported.items() if value]
    if bad:
        raise ValueError(f"outer_loops=True does not support {', '.join(bad)} yet")

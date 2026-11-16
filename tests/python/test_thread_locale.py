# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""gpusim2grid must leave the calling thread's locale as it found it.

cuDSS JIT-compiles kernels through NVRTC on a process' first ``cudssExecute``, and
``nvrtcCompileProgram`` switches the calling thread to the "C" locale (``uselocale``) without
switching it back. Python then reads the preferred encoding as ASCII, and every later ``open()``
without an explicit encoding fails on a non-ASCII file (lightsim2grid's ``init_from_matpower``
on the PGLib pegase cases, whose comments carry accented names).

Each check runs in a fresh interpreter: NVRTC compiles only once per process, and
``warmup()`` returns at once when it already ran.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from conftest import requires_gpu

_PRELUDE = """
import locale, warnings
warnings.filterwarnings("ignore")
locale.setlocale(locale.LC_ALL, "")
before = locale.getpreferredencoding(False)
"""

_EPILOGUE = """
after = locale.getpreferredencoding(False)
assert after == before, f"preferred encoding {before!r} -> {after!r}"
print("OK", before)
"""


def _run(body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(_PRELUDE) + textwrap.dedent(body) + textwrap.dedent(_EPILOGUE)
    env = dict(os.environ, LC_ALL="C.UTF-8", LANG="C.UTF-8", PYTHONUTF8="0")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)


@requires_gpu
def test_warmup_keeps_the_thread_locale():
    proc = _run("""
        import gpusim2grid
        gpusim2grid.warmup()
    """)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("OK UTF-8"), proc.stdout


@requires_gpu
def test_first_session_keeps_the_thread_locale():
    """No warmup(): the first session's own cuDSS calls are where NVRTC compiles."""
    proc = _run("""
        import numpy as np
        import pandapower.networks as pn
        from lightsim2grid.network import init_from_pandapower
        from gpusim2grid import ContingencyAnalysisGPU
        grid = init_from_pandapower(pn.case14())
        grid.ac_pf(np.ones(len(grid.get_bus_vn_kv()), dtype=complex), 10, 1e-8)
        ca = ContingencyAnalysisGPU(grid, nb_iter=4)
        ca.add_contingencies_by_branch_id([[0], [1]])
        ca.compute(batch_size=2)
    """)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("OK UTF-8"), proc.stdout

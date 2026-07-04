# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .._gpusim2grid import AcPfTimings as AcPfTimings_GPU
import numpy as np


def pretty_timings_ac_gpu(timings: AcPfTimings_GPU, nb_solve: int = 1, batch_size : int = 1, detailed:int=0) -> str:
    status: str = "CONVERGED" if timings.converged else "NOT CONVERGED"
    lines: list[str] = [
        f"Status               : {status}",
        f"Iterations           : {timings.nb_iter}",
        f"Total time           : {timings.t_total.wall_ms:.3f} ms, {1000./timings.t_total.wall_ms:.2e} pf/s",
    ]
    if detailed >= 1 and timings.nb_iter > 0:
        t_solve = timings.t_solve.wall_ms
        lines += [
            f"  Mismatch (sum)    : {np.sum(timings.t_mismatch.wall_ms):.3f} ms",
            f"  Jacobian (sum)    : {timings.t_fill_J.wall_ms:.3f} ms ",
            f"  Linear (re)factorize (sum)    : {timings.t_first_factorize.wall_ms + timings.t_refactorize.wall_ms:.3f} ms ",
            f"  Linear solve (sum): {t_solve:.3f} ms"]
    return "\n".join(lines)
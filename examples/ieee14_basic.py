# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
ieee14_basic.py — end-to-end AC power flow on the IEEE 14-bus case.

Solves the base case on the GPU with the single-system Newton-Raphson solver
and compares the result against the lightsim2grid KLU reference.

Run:
    python examples/ieee14_basic.py
"""
import numpy as np

from _common import load_case, print_acpf_timings
from gpusim2grid import AcPfGPU
from gpusim2grid.compilation_options import is_fp32


def main():
    print(f"gpusim2grid precision: {'FP32' if is_fp32 else 'FP64'}")
    d = load_case("case14")

    ac = AcPfGPU(d["grid"], max_iter=10, tol=1e-6)
    v_out = ac.solve()

    # Compare |V| against the CPU reference. Angles are compared relative to the
    # slack bus to remove the global phase offset.
    err_vmag = np.abs(np.abs(v_out) - np.abs(d["v_ref"])).max()
    ref_ang = np.angle(d["v_ref"]) - np.angle(d["v_ref"][d["slack"][0]])
    gpu_ang = np.angle(v_out) - np.angle(v_out[d["slack"][0]])
    err_ang = np.abs(gpu_ang - ref_ang).max()

    timings = ac.timings
    print(f"converged           : {timings.converged}")
    print(f"NR iterations       : {timings.nb_iter}")
    print(f"max |V| error vs CPU: {err_vmag:.2e} pu")
    print(f"max angle error     : {err_ang:.2e} rad")
    print()
    print_acpf_timings(timings)


if __name__ == "__main__":
    main()

"""
ieee14_basic.py — end-to-end AC power flow on the IEEE 14-bus case.

Solves the base case on the GPU with the single-system Newton-Raphson solver
and compares the result against the lightsim2grid KLU reference.

Run:
    python examples/ieee14_basic.py
"""
import numpy as np

from _common import load_case
from gpusim2grid.acpf_nr import acpf_nr_gpu
from gpusim2grid.compilation_options import is_fp32


def main():
    print(f"gpusim2grid precision: {'FP32' if is_fp32 else 'FP64'}")
    d = load_case("case14")

    # Output voltage vector, written in place by the solver.
    v_out = np.zeros(d["n_bus"], dtype=complex)

    timings = acpf_nr_gpu(
        v_out,
        d["Ybus"],
        d["v_init"].copy(),     # warm-start (DC solution)
        d["Sbus"],
        d["slack"],
        d["slack_weights"],
        d["pv"],
        d["pq"],
        max_iter=10,
        tol=1e-6,
        nb_solve=1,
    )

    # Compare |V| against the CPU reference. Angles are compared relative to the
    # slack bus to remove the global phase offset.
    err_vmag = np.abs(np.abs(v_out) - np.abs(d["v_ref"])).max()
    ref_ang = np.angle(d["v_ref"]) - np.angle(d["v_ref"][d["slack"][0]])
    gpu_ang = np.angle(v_out) - np.angle(v_out[d["slack"][0]])
    err_ang = np.abs(gpu_ang - ref_ang).max()

    print(f"converged           : {timings.converged}")
    print(f"NR iterations       : {timings.nb_iter}")
    print(f"max |V| error vs CPU: {err_vmag:.2e} pu")
    print(f"max angle error     : {err_ang:.2e} rad")
    print(f"total solve time    : {timings.t_total.wall_ms:.3f} ms")


if __name__ == "__main__":
    main()

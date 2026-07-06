"""
distributed_slack.py — augmented AC power flow with distributed slack.

When seeded from a lightsim2grid grid through the C++ bridge (the default path
for the ``AcPfGPU`` facade), gpusim2grid solves the *same augmented Newton-Raphson
system lightsim2grid poses* rather than the bare ``[pvpq | pq]`` system. The
in-Jacobian power-system controls are reproduced bit-for-bit:

  * distributed slack (this example),
  * HVDC angle-droop,
  * SVC (static var compensator),
  * remote generator voltage control.

Here we add a second ext_grid to the IEEE 14-bus case so the active-power
slack is genuinely shared (distributed) across two weighted slack buses, solve
on the CPU with lightsim2grid as the reference, then solve the same augmented
system on the GPU and compare.

Run:
    python examples/distributed_slack.py
"""
import warnings

import numpy as np

from gpusim2grid import AcPfGPU
from gpusim2grid._gpusim2grid import have_ls2g_bridge
from gpusim2grid.compilation_options import is_fp32


def solved_multislack_grid():
    """IEEE 14-bus with a 2nd weighted slack → genuine distributed slack (N=2)."""
    import pandapower as pp
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        # Second slack on bus 3; both slacks absorb the mismatch by weight.
        pp.create_ext_grid(net, bus=3, vm_pu=1.0, va_degree=0.0, slack_weight=1.0)
        net.ext_grid.loc[0, "slack_weight"] = 1.0
        pp.runpp(net, distributed_slack=True)

        grid = init_from_pandapower(net)
        grid.change_algorithm(AlgorithmType.NR_KLU)   # MultiSlackNRSystem
        n_bus = grid.get_bus_vn_kv().shape[0]
        v0 = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
        grid.ac_pf(v0.copy(), 30, 1e-10)              # CPU reference solve
    return grid


def main():
    print(f"gpusim2grid precision: {'FP32' if is_fp32 else 'FP64'}")
    if not have_ls2g_bridge:
        print("This build has no lightsim2grid C++ bridge; the augmented "
              "(distributed-slack) path is unavailable. Rebuild against "
              "lightsim2grid_core to enable it.")
        return

    grid = solved_multislack_grid()
    n_slack = grid.get_slack_ids_solver().shape[0]
    V_ref = grid.get_V_solver()

    # Bridge path is auto-detected; AcPfGPU solves the augmented system.
    ac = AcPfGPU(grid, max_iter=40, tol=1e-10)
    V_gpu = ac.solve()

    # The augmented Jacobian adds the non-reference slack angle column(s) plus
    # the slack_absorbed column over the bare n_pv + 2*n_pq system.
    n_pv = grid.get_pv().shape[0]
    n_pq = grid.get_pq().shape[0]
    err_v = np.abs(V_gpu - V_ref).max()

    print(f"slack buses (distributed): {n_slack}")
    print(f"bare J dim (n_pv+2*n_pq) : {n_pv + 2 * n_pq}")
    print(f"augmented J dim          : {ac.session.dim_J}")
    print(f"max |V| error vs CPU     : {err_v:.2e} pu")


if __name__ == "__main__":
    main()

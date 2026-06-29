# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

def compute_branch_flows_cpu(V, branch_from, branch_to, yff, yft, ytf, ytt, bus_vn_kv, sn_mva):
    """Compute branch current magnitudes (in amperes) from a converged voltage vector.

    Uses the π-model admittance parameters (same formula as the GPU kernel):
        I_or = yff * V[from] + yft * V[to]   (origin  / from-bus terminal)
        I_ex = ytf * V[from] + ytt * V[to]   (extremity / to-bus terminal)

    The per-unit magnitude is converted to amperes using the from-bus voltage base:
        I_base_A = sn_mva * 1e6 / (sqrt(3) * bus_vn_kv[from] * 1e3)

    Parameters
    ----------
    V          : (n_bus,) complex  — converged bus voltages in pu
    branch_from: (n_branches,) int — from-bus index for each branch
    branch_to  : (n_branches,) int — to-bus index for each branch
    yff        : (n_branches,) complex — yac_11 (self-admittance at from-bus)
    yft        : (n_branches,) complex — yac_12 (mutual admittance from→to)
    ytf        : (n_branches,) complex — yac_21 (mutual admittance to→from)
    ytt        : (n_branches,) complex — yac_22 (self-admittance at to-bus)
    bus_vn_kv  : (n_bus,) float   — nominal voltage in kV per bus
    sn_mva     : float             — system apparent-power base in MVA

    Returns
    -------
    or_amps : (n_branches,) float — current magnitude in A at origin terminal
    ex_amps : (n_branches,) float — current magnitude in A at extremity terminal
    """
    Vi = V[branch_from]
    Vj = V[branch_to]
    I_or = yff * Vi + yft * Vj
    I_ex = ytf * Vi + ytt * Vj
    base_current_A = sn_mva * 1e6 / (np.sqrt(3.) * bus_vn_kv[branch_from] * 1e3)
    return np.abs(I_or) * base_current_A, np.abs(I_ex) * base_current_A
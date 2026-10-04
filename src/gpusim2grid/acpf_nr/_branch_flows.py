# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

def compute_branch_flows_cpu(V, branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff, bus_vn_kv, sn_mva):
    """Compute branch current magnitudes (in amperes) from a converged voltage vector.

    Uses the π-model admittance parameters (same formula as the GPU kernel):
        I_or = yff_eff * V[from] + yft_eff * V[to]   (origin  / from-bus terminal)
        I_ex = ytf_eff * V[from] + ytt_eff * V[to]   (extremity / to-bus terminal)

    The per-unit magnitude of each terminal current is converted to amperes
    with the nominal voltage of the bus that terminal sits on (as lightsim2grid
    does; the two bases differ on a transformer or on any branch joining two
    voltage levels):
        I_base_or_A = sn_mva * 1e6 / (sqrt(3) * bus_vn_kv[from] * 1e3)
        I_base_ex_A = sn_mva * 1e6 / (sqrt(3) * bus_vn_kv[to]   * 1e3)

    A branch endpoint can be ``-1``: a side that lightsim2grid Kron-reduced away
    (isolated bus, half-open line with ``keep_half_open_lines``), which has no
    voltage in the solved system. Same convention as the GPU kernels
    (``compute_branch_flows_kernel``, ``check_limit_violations_kernel``): that
    side is treated as V=0, its terminal current is reported as 0 A (the
    terminal does not exist), and its amp base falls back to the other
    endpoint's nominal voltage (it only has to be finite). Without this guard, numpy silently reads
    ``V[-1]`` / ``bus_vn_kv[-1]`` (the last bus) and reports a wrong, possibly
    limit-violating, current on the live side.

    Parameters
    ----------
    V          : (n_bus,) complex  — converged bus voltages in pu
    branch_from: (n_branches,) int — from-bus index for each branch (-1: no such terminal)
    branch_to  : (n_branches,) int — to-bus index for each branch (-1: no such terminal)
    yff_eff        : (n_branches,) complex — yac_eff_11 (self-admittance at from-bus)
    yft_eff        : (n_branches,) complex — yac_eff_12 (mutual admittance from→to)
    ytf_eff        : (n_branches,) complex — yac_eff_21 (mutual admittance to→from)
    ytt_eff        : (n_branches,) complex — yac_eff_22 (self-admittance at to-bus)
    bus_vn_kv  : (n_bus,) float   — nominal voltage in kV per bus
    sn_mva     : float             — system apparent-power base in MVA

    Returns
    -------
    or_amps : (n_branches,) float — current magnitude in A at origin terminal
    ex_amps : (n_branches,) float — current magnitude in A at extremity terminal
    """
    branch_from = np.asarray(branch_from)
    branch_to = np.asarray(branch_to)
    has_from = branch_from >= 0
    has_to = branch_to >= 0
    Vi = np.where(has_from, V[np.where(has_from, branch_from, 0)], 0.)
    Vj = np.where(has_to, V[np.where(has_to, branch_to, 0)], 0.)
    I_or = np.where(has_from, yff_eff * Vi + yft_eff * Vj, 0.)
    I_ex = np.where(has_to, ytf_eff * Vi + ytt_eff * Vj, 0.)
    # amp base of each terminal: its own bus' nominal voltage, or the other
    # end's when that side does not exist (a branch cannot have both ends
    # reduced away and still be listed; 0 A if it somehow is)
    def _base(vn_bus):
        has_vn = vn_bus >= 0
        vn_kv = np.where(has_vn, bus_vn_kv[np.where(has_vn, vn_bus, 0)], np.inf)
        return sn_mva * 1e6 / (np.sqrt(3.) * vn_kv * 1e3)
    base_or_A = _base(np.where(has_from, branch_from, branch_to))
    base_ex_A = _base(np.where(has_to, branch_to, branch_from))
    return np.abs(I_or) * base_or_A, np.abs(I_ex) * base_ex_A
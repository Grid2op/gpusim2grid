# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
compute_flows — differentiable branch-flow computation in pure PyTorch.

π-model branch equations:
    I_or = yff * V[from] + yft * V[to]
    I_ex = ytf * V[from] + ytt * V[to]

    S_or = V[from] * conj(I_or)      (complex apparent power, per-unit)
    S_ex = V[to]   * conj(I_ex)

    base_A = sn_mva * 1e6 / (sqrt(3) * vn_kv[from] * 1e3)

All operations are natively differentiable via PyTorch autograd.
The function body is framework-neutral: a future JAX variant only needs to swap
the V indexing at the call site.
"""
import math

import torch
from torch import Tensor


def compute_flows(
    V: Tensor,           # complex [n_bus] on GPU
    yff: Tensor,         # complex [n_branches]
    yft: Tensor,         # complex [n_branches]
    ytf: Tensor,         # complex [n_branches]
    ytt: Tensor,         # complex [n_branches]
    branch_from: Tensor, # int64  [n_branches]
    branch_to: Tensor,   # int64  [n_branches]
    bus_vn_kv: Tensor,   # float  [n_bus] — nominal voltage kV per bus
    sn_mva: float,
) -> dict:
    """
    Compute branch power flows and ampere flows from converged voltages.

    Returns a dict with keys:

    - ``p_or_mw``, ``q_or_mvar`` — MW/MVAr at the origin terminal
    - ``p_ex_mw``, ``q_ex_mvar`` — MW/MVAr at the extremity terminal
    - ``i_or_a``, ``i_ex_a`` — ampere flows at each terminal

    All values are real tensors of shape [n_branches].
    """
    Vi = V[branch_from]  # complex [n_branches]
    Vj = V[branch_to]    # complex [n_branches]

    I_or = yff * Vi + yft * Vj  # origin terminal current
    I_ex = ytf * Vi + ytt * Vj  # extremity terminal current

    S_or = Vi * I_or.conj()     # complex apparent power (pu), origin
    S_ex = Vj * I_ex.conj()     # complex apparent power (pu), extremity

    base_A = sn_mva * 1e6 / (math.sqrt(3.0) * bus_vn_kv[branch_from] * 1e3)

    return {
        "p_or_mw":   S_or.real * sn_mva,
        "q_or_mvar": S_or.imag * sn_mva,
        "p_ex_mw":   S_ex.real * sn_mva,
        "q_ex_mvar": S_ex.imag * sn_mva,
        "i_or_a":    I_or.abs() * base_A,
        "i_ex_a":    I_ex.abs() * base_A,
    }
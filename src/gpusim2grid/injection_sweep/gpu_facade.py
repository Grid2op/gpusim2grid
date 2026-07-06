# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU injection sweep driven directly from a lightsim2grid grid.

``InjectionSweepGPU`` reuses the CPU base-case solve, keeps the same Ybus, and
re-solves the network on the GPU for many (P, Q) injection profiles in parallel.
It is a thin facade over :class:`_InjectionSweepSolver`.
"""

from . import _InjectionSweepSolver, _normalize_device
from .._ls2g_utils import (
    extract_grid_arrays,
    extract_branch_data,
    grid_from_pandapower,
    _validate_precision,
)
from .. import _gpusim2grid as _cpp

__all__ = ["InjectionSweepGPU"]


def _have_bridge():
    return getattr(_cpp, "have_ls2g_bridge", False)


class InjectionSweepGPU:
    """Batch injection sweep on the GPU, seeded from a CPU base-case solve.

    By default (``use_bridge=None`` auto-detects the compiled lightsim2grid
    bridge) every scenario is solved on the **same augmented system lightsim2grid
    poses** — distributed slack, HVDC angle-droop, SVC, and remote generator
    voltage control are carried through the Jacobian as Sbus varies. When the
    bridge is unavailable (or ``use_bridge=False``) the Python-array fallback
    solves only the bare ``[pvpq | pq]`` system.

    Parameters
    ----------
    grid : lightsim2grid GridModel / LSGrid, or tuple
        Either the grid whose base case is solved on the CPU, or an explicit
        ``(Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq)`` array tuple
        for callers without a lightsim2grid grid. In the latter case,
        ``use_bridge`` must not be True and branch data (if needed for
        :meth:`compute_flows`) must be supplied via :meth:`set_branch_data`.
    init_from_n_powerflow : bool, default True
        Seed each scenario with the CPU-converged base-case voltage V0,
        skipping the GPU base-case NR loop (see :class:`ContingencyAnalysisGPU`).
    precision : {"fp64", "fp32", None}, default "fp64"
        Validated against the compiled extension.
    nb_iter : int, default 4
        Newton-Raphson iterations per scenario in the batch phase.
    max_iter_base, tol_base : int, float
        Base-case CPU solve / GPU fallback settings.
    device : None | int | "cuda" | "cuda:N"
        Target CUDA device.
    auto_branch_data : bool, default True
        Extract and store π-model branch admittances so :meth:`compute_flows`
        works without a manual ``set_branch_data`` call.

    Examples
    --------
    >>> grid = init_from_pandapower(net)
    >>> grid.ac_pf(Vinit, max_iter=10, tol=1e-8)
    >>> sweep = InjectionSweepGPU(grid, nb_iter=4)
    >>> sweep.set_injections(p_mw, q_mvar, sn_mva)   # (n_scen, n_bus)
    >>> V_batch = sweep.compute(batch_size=512)      # DLPack (n_scen, n_bus)
    >>> residuals = sweep.last_residuals()

    Explicit-array path (no lightsim2grid grid available):

    >>> sweep = InjectionSweepGPU(
    ...     (Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq), nb_iter=4)
    >>> sweep.set_injections(p_mw, q_mvar, sn_mva)
    >>> V_batch = sweep.compute(batch_size=512)
    """

    def __init__(self, grid, *, init_from_n_powerflow=True, precision="fp64",
                 nb_iter=4, max_iter_base=10, tol_base=1e-8, device=None,
                 auto_branch_data=True, use_bridge=None):
        _validate_precision(precision)

        if isinstance(grid, (tuple, list)):
            # Explicit-array mode: no grid to seed/extract branch data from.
            if use_bridge:
                raise ValueError(
                    "use_bridge=True requires `grid` to be a lightsim2grid "
                    "grid object, not an explicit-array tuple.")
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq = grid
            self._inner = _InjectionSweepSolver(
                Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
                batch_size=100, nb_iter=nb_iter,
                max_iter_base=max_iter_base, tol_base=tol_base, device=device,
                presolved_v=bool(init_from_n_powerflow))
            # No grid to auto-extract branch data from: call set_branch_data()
            # manually if compute_flows() is needed.
        else:
            if use_bridge is None:
                use_bridge = _have_bridge()

            if use_bridge:
                session = _cpp._make_is_session_from_lsgrid(
                    grid, bool(init_from_n_powerflow), 100, int(nb_iter),
                    int(max_iter_base), float(tol_base), _normalize_device(device),
                    bool(auto_branch_data))
                self._inner = _InjectionSweepSolver._wrap_session(
                    session, max_iter_base=max_iter_base, tol_base=tol_base)
            else:
                d = extract_grid_arrays(grid, max_iter=max_iter_base, tol=tol_base)
                vinit = d["v_converged"] if init_from_n_powerflow else d["v_init"]

                self._inner = _InjectionSweepSolver(
                    d["Ybus"], vinit, d["Sbus"],
                    d["slack"], d["slack_weights"], d["pv"], d["pq"],
                    batch_size=100, nb_iter=nb_iter,
                    max_iter_base=max_iter_base, tol_base=tol_base, device=device,
                    presolved_v=init_from_n_powerflow)

                if auto_branch_data:
                    branch_args, _, _ = extract_branch_data(grid)
                    self._inner.set_branch_data(*branch_args)

        self._init_from_n_powerflow = bool(init_from_n_powerflow)
        self._last_residuals = None

    # ------------------------------------------------------------------ spec
    def set_branch_data(self, branch_from, branch_to, yff, yft, ytf, ytt,
                        bus_vn_kv, sn_mva):
        """Store π-model branch admittances (explicit-array mode only).

        Grid mode extracts this automatically (unless ``auto_branch_data``
        was set to False at construction); only needed when ``grid`` was an
        explicit-array tuple. Required before :meth:`compute_flows`.
        """
        self._inner.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                                    bus_vn_kv, sn_mva)

    def set_injections(self, p_mw, q_mvar, sn_mva):
        """Store the (n_scenarios, n_bus) MW / MVAr injection arrays."""
        self._inner.set_injections(p_mw, q_mvar, sn_mva)

    def compute(self, batch_size=512):
        """Solve every scenario; return DLPack (n_scenarios, n_bus) complex.

        The capsule aliases live GPU memory; clone before the next ``compute()``
        for a snapshot.  Residuals are cached for :meth:`last_residuals`.
        """
        self._inner.batch_size = int(batch_size)
        self._inner.run()
        self._last_residuals = self._inner.residuals
        return self._inner.v_results_dlpack()

    def last_residuals(self):
        """``‖F‖∞`` per scenario from the most recent :meth:`compute`."""
        if self._last_residuals is None:
            raise RuntimeError("Call compute() before last_residuals().")
        return self._last_residuals.to_numpy()

    def compute_flows(self):
        """Compute branch currents (``or_amps`` / ``ex_amps``) after compute()."""
        self._inner.compute_flows()

    # ----------------------------------------------------------- pass-through
    @property
    def or_amps(self):
        return self._inner.or_amps

    @property
    def ex_amps(self):
        return self._inner.ex_amps

    @property
    def V_results(self):
        return self._inner.V_results

    @property
    def strategy(self):
        return self._inner.strategy

    @strategy.setter
    def strategy(self, value):
        self._inner.strategy = value

    @property
    def timings(self):
        return self._inner.timings

    @property
    def n_scenarios(self):
        return self._inner.n_scenarios

    @property
    def n_branches(self):
        return self._inner.n_branches

    @property
    def n_bus(self):
        return self._inner.n_bus

    @property
    def solver(self):
        """The underlying :class:`_InjectionSweepSolver` (escape hatch)."""
        return self._inner

    # --------------------------------------------------------------- factory
    @classmethod
    def from_pandapower(cls, net, solver_type="KLU", **kwargs):
        grid = grid_from_pandapower(net, solver_type=solver_type)
        return cls(grid, **kwargs)

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU contingency analysis driven directly from a lightsim2grid grid.

``ContingencyAnalysisGPU`` is the GPU sibling of lightsim2grid's
``ContingencyAnalysis(grid)``: it takes the solved grid object, reuses the
CPU base-case (N) power flow, and runs the batched contingencies on the GPU.
It is a thin, batch-oriented facade over :class:`ContingencyAnalysisSolver`.
"""

import numpy as np

from .contingency_analysis import ContingencyAnalysisSolver, _normalize_device
from ._ls2g_utils import (
    extract_grid_arrays,
    extract_branch_data,
    grid_from_pandapower,
    _validate_precision,
)
from . import _gpusim2grid as _cpp

__all__ = ["ContingencyAnalysisGPU", "optimize_reference_slack"]


def _have_bridge():
    return getattr(_cpp, "have_ls2g_bridge", False)


def optimize_reference_slack(grid, contingency_branch_ids, *, Vinit=None,
                             max_iter=30, tol=1e-10):
    """Choose the angle-reference slack that minimises skipped contingencies, set
    it on the grid, and re-solve the base AC power flow (CPU, in lightsim2grid).

    For ``handle_disconnected_grid`` the reference slack is fixed for the whole
    GPU batch, and any contingency that strands it is skipped (NaN). lightsim2grid
    can pick the slack stranded by the *fewest* of the given contingencies and
    re-solve the base case with it as the angle reference; the GPU companion then
    inherits that reference (read off the solved grid) and skips as few split
    contingencies as possible. Call this **before** building a
    :class:`ContingencyAnalysisGPU` (bridge / multi-slack path).

    Requires a lightsim2grid whose ``ContingencyAnalysisCPP`` exposes
    ``pick_reference_slack`` and whose grid exposes ``set_reference_slack_bus``.

    Parameters
    ----------
    grid : lightsim2grid LSGrid
        Grid to re-solve in place (its slack ordering is updated).
    contingency_branch_ids : list[list[int]]
        Branch-removal contingencies (lines-then-trafos), as passed to
        :meth:`ContingencyAnalysisGPU.add_contingencies_by_branch_id`.
    Vinit : (n_bus,) complex, optional
        Base-case warm start; defaults to a flat 1.0 start.
    max_iter, tol : int, float
        Base-case AC solve settings.

    Returns
    -------
    int
        The chosen reference bus id (gridmodel numbering), or -1 if the grid has
        no slack to choose from.
    """
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    ca = ContingencyAnalysisCPP(grid)
    for ids in contingency_branch_ids:
        ca.add_nk([int(i) for i in ids])
    ref = ca.pick_reference_slack()
    if ref is not None and ref >= 0:
        grid.set_reference_slack_bus(int(ref))
        n_bus = grid.get_bus_vn_kv().shape[0]
        v = Vinit if Vinit is not None else np.ones(n_bus, dtype=complex)
        grid.ac_pf(v, int(max_iter), float(tol))
    return ref


class ContingencyAnalysisGPU:
    """Batch N-k contingency analysis on the GPU, seeded from a CPU solve.

    By default (``use_bridge=None`` auto-detects the compiled lightsim2grid
    bridge) every contingency is solved on the **same augmented system
    lightsim2grid poses** — distributed slack, HVDC angle-droop, SVC, and remote
    generator voltage control are carried through the Jacobian under the Ybus
    patch. When the bridge is unavailable (or ``use_bridge=False``) the
    Python-array fallback solves only the bare ``[pvpq | pq]`` system.

    Parameters
    ----------
    grid : lightsim2grid GridModel / LSGrid
        The grid whose base case has been (or will be) solved on the CPU.
        lightsim2grid owns the physics; gpusim2grid only runs the GPU batch.
    init_from_n_powerflow : bool, default True
        Seed every contingency with the CPU-converged base-case voltage V0
        (mirrors lightsim2grid's flag of the same name).  This skips the GPU
        base-case Newton-Raphson loop: at V0 the base mismatch is already below
        tolerance, so a single GPU step fills J at the right operating point and
        factorizes once.  When ``False``, the GPU runs ``max_iter_base`` NR
        iterations from a DC warm-start.
    precision : {"fp64", "fp32", None}, default "fp64"
        Validated against the compiled extension; precision is a build-time
        choice.  ``None`` accepts whatever was compiled.
    nb_iter : int, default 4
        Newton-Raphson iterations per contingency in the batch phase.
    max_iter_base, tol_base : int, float
        Base-case CPU solve / GPU fallback settings.
    device : None | int | "cuda" | "cuda:N"
        Target CUDA device.

    Examples
    --------
    >>> grid = init_from_pandapower(net)
    >>> grid.ac_pf(Vinit, max_iter=10, tol=1e-8)
    >>> ca = ContingencyAnalysisGPU(grid, init_from_n_powerflow=True, nb_iter=4)
    >>> ca.add_contingencies_by_branch_id([[12], [40], [12, 40]])
    >>> V_batch = ca.compute(batch_size=512)   # DLPack (n_ctg, n_bus) complex
    >>> residuals = ca.last_residuals()        # ‖F‖∞ per contingency
    """

    def __init__(self, grid, *, init_from_n_powerflow=True, precision="fp64",
                 nb_iter=4, max_iter_base=10, tol_base=1e-8, device=None,
                 use_bridge=None, handle_disconnected_grid=False):
        _validate_precision(precision)

        if use_bridge is None:
            use_bridge = _have_bridge()

        if use_bridge:
            # Zero-copy: extract everything in C++ off the solved LSGrid
            # (no scipy CSR marshalling, branch data set automatically).
            session = _cpp._make_ca_session_from_lsgrid(
                grid, bool(init_from_n_powerflow), 100, int(nb_iter),
                int(max_iter_base), float(tol_base), _normalize_device(device))
            self._inner = ContingencyAnalysisSolver._wrap_session(
                session, max_iter_base=1 if init_from_n_powerflow else max_iter_base,
                tol_base=tol_base)
            self._n_branches = self._inner._s.n_branches
        else:
            d = extract_grid_arrays(grid, max_iter=max_iter_base, tol=tol_base)
            # Seed from the CPU base-case solution and let the GPU do a single
            # (no-op) NR step, or run the full GPU base solve from the DC start.
            vinit = d["v_converged"] if init_from_n_powerflow else d["v_init"]
            base_iters = 1 if init_from_n_powerflow else max_iter_base

            self._inner = ContingencyAnalysisSolver(
                d["Ybus"], vinit, d["Sbus"],
                d["slack"], d["slack_weights"], d["pv"], d["pq"],
                batch_size=100, nb_iter=nb_iter,
                max_iter_base=base_iters, tol_base=tol_base, device=device)

            # Branch admittances come straight from lightsim2grid (never
            # recomputed) so branch-removal contingencies become exact Ybus
            # patches.
            branch_args, _, _ = extract_branch_data(grid)
            self._inner.set_branch_data(*branch_args)
            self._n_branches = len(branch_args[0])

        # Solve the largest connected component of a split grid (masking the rest
        # as NaN) instead of skipping such contingencies. Works on both the bridge
        # and the array path (mutable property on the underlying session).
        self._inner.handle_disconnected_grid = bool(handle_disconnected_grid)

        self._nb_iter = int(nb_iter)
        self._init_from_n_powerflow = bool(init_from_n_powerflow)
        self._last_residuals = None

    # ------------------------------------------------------------------ spec
    def add_contingencies_by_branch_id(self, branch_ids_per_ctg):
        """Define contingencies as branch removals.

        Parameters
        ----------
        branch_ids_per_ctg : list[list[int]]
            One inner list per contingency, holding the branch indices to trip.
            Indices are 0-based, lines first then trafos (``c < n_lines`` is
            line ``c``; ``c >= n_lines`` is trafo ``c - n_lines``).
        """
        self._inner.build_contingencies(branch_ids_per_ctg)

    def compute(self, batch_size=512):
        """Solve every contingency and return the batched voltages.

        Returns a DLPack capsule of shape ``(n_contingencies, n_bus)``,
        complex, aliasing live GPU memory.  Pass to ``torch.from_dlpack`` /
        ``jax.dlpack.from_dlpack``; clone before the next ``compute()`` for a
        snapshot.  Residuals are cached for :meth:`last_residuals`.
        """
        self._inner.batch_size = int(batch_size)
        self._inner.run()
        self._last_residuals = self._inner.residuals
        return self._inner.v_results_dlpack()

    def last_residuals(self):
        """``‖F‖∞`` per contingency from the most recent :meth:`compute`."""
        if self._last_residuals is None:
            raise RuntimeError("Call compute() before last_residuals().")
        return self._last_residuals.to_numpy()

    def compute_flows(self):
        """Compute branch currents (``or_amps`` / ``ex_amps``) after compute()."""
        self._inner.compute_flows()

    # ----------------------------------------------------------- pass-through
    @property
    def or_amps(self):
        """DeviceBuffer: (n_ctg * n_branches,) origin terminal amps."""
        return self._inner.or_amps

    @property
    def ex_amps(self):
        """DeviceBuffer: (n_ctg * n_branches,) extremity terminal amps."""
        return self._inner.ex_amps

    @property
    def V_results(self):
        """DeviceBuffer: (n_ctg * n_bus,) complex voltages (lazy D->H)."""
        return self._inner.V_results

    @property
    def strategy(self):
        """Linear-solve strategy (str). Takes effect on the next compute()."""
        return self._inner.strategy

    @strategy.setter
    def strategy(self, value):
        self._inner.strategy = value

    @property
    def timings(self):
        """BatchTimings from the most recent compute() / compute_flows()."""
        return self._inner.timings

    @property
    def n_branches(self):
        return self._n_branches

    @property
    def n_contingencies(self):
        return self._inner._s.n_contingencies

    @property
    def n_bus(self):
        return self._inner._s.n_bus

    @property
    def solver(self):
        """The underlying :class:`ContingencyAnalysisSolver` (escape hatch)."""
        return self._inner

    # --------------------------------------------------------------- factory
    @classmethod
    def from_pandapower(cls, net, solver_type="KLU", **kwargs):
        """Build directly from a pandapower network (convenience).

        Converts to a lightsim2grid grid, solves the base case on the CPU, then
        constructs the GPU analysis.
        """
        grid = grid_from_pandapower(net, solver_type=solver_type)
        return cls(grid, **kwargs)

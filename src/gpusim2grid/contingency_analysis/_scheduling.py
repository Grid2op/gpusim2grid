# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Scheduling of the batched solves: chunked (every row runs ``nb_iter``
Newton iterations) or continuous (a row leaves as soon as it is done and its
slot is refilled from the queue), and the per-row outcome of a run.

Workload-agnostic like ``_limit_violations.py``: the three batch engines and
facades share it (``SchedulingEngineMixin`` expects ``self._s``, the compiled
session; ``SchedulingFacadeMixin`` expects ``self._inner``, the engine). See
docs/dev_notes/continuous_batching.md.
"""

import warnings
from enum import IntEnum

import numpy as np

from .._gpusim2grid import BatchScheduling as _BatchScheduling, is_fp32 as _is_fp32

_SCHEDULING_MAP = {
    "chunked":    _BatchScheduling.Chunked,
    "continuous": _BatchScheduling.Continuous,
}
_SCHEDULING_NAME = {v: k for k, v in _SCHEDULING_MAP.items()}

# strategies the continuous schedule refuses (see check_continuous_scheduling
# in batch_pf_driver.cuh, which raises at run() whatever the order of the
# settings)
_CONTINUOUS_REFUSED_STRATEGIES = ("direct_iter0_only", "direct_refactor_every_n")


class RowStatus(IntEnum):
    """Outcome of one row of a batched solve (``get_row_status()``); mirrors
    the C++ ``RowStatus`` (contingency/row_status.hpp)."""
    #: ``||F||inf < tol``
    CONVERGED = 0
    #: the ``nb_iter`` budget used up, residual finite but ``>= tol``
    MAX_ITER = 1
    #: residual not finite
    DIVERGED = 2
    #: dropped by the pre-check (would disconnect the grid, strands the angle
    #: reference, ...), never solved
    NOT_SIMULATED = 3


def default_tol():
    """Per unit: 1e-8 in double precision; 1e-3 in an FP32 build, whose
    residual floor is around 1e-4 already on case118 (with a tighter tol the
    rows near it would end MAX_ITER)."""
    return 1e-3 if _is_fp32 else 1e-8


def resolve_scheduling(value):
    """'chunked' / 'continuous' (or the BatchScheduling enum) -> enum."""
    if isinstance(value, _BatchScheduling):
        return value
    if value not in _SCHEDULING_MAP:
        raise ValueError(f"Unknown scheduling {value!r}. Choose from: {list(_SCHEDULING_MAP)}")
    return _SCHEDULING_MAP[value]


class SchedulingEngineMixin:
    """For the engine classes (``_ContingencyAnalysisSolver`` & co): expects
    ``self._s`` to be the compiled session and ``self.strategy`` its strategy
    name."""

    def _init_scheduling(self, scheduling="chunked", nb_iter_per_round=1, tol=None):
        self.nb_iter_per_round = nb_iter_per_round
        self.tol = tol
        self.scheduling = scheduling

    @property
    def scheduling(self):
        """``'chunked'`` (default: ``ceil(n_rows / batch_size)`` chunks, every
        row runs exactly ``nb_iter`` iterations) or ``'continuous'``
        (``batch_size`` slots; every ``nb_iter_per_round`` iterations a row
        that converged (``residual < tol``), diverged or used its ``nb_iter``
        budget leaves and its slot is refilled at once). Continuous refuses
        the ``'direct_iter0_only'`` / ``'direct_refactor_every_n'`` strategies
        and the batched adjoint. Takes effect on the next run()."""
        return _SCHEDULING_NAME[self._s.scheduling]

    @scheduling.setter
    def scheduling(self, value):
        sched = resolve_scheduling(value)
        if (sched == _BatchScheduling.Continuous
                and getattr(self, "strategy", None) in _CONTINUOUS_REFUSED_STRATEGIES):
            raise ValueError(
                f"scheduling='continuous' does not support strategy {self.strategy!r}")
        self._s.scheduling = sched

    @property
    def nb_iter_per_round(self):
        """Continuous scheduling: Newton iterations every slot runs between two
        convergence checks (k >= 1; a row then runs a multiple of k
        iterations, at least k). Takes effect on the next run()."""
        return self._s.nb_iter_per_round

    @nb_iter_per_round.setter
    def nb_iter_per_round(self, value):
        value = int(value)
        if value < 1:
            raise ValueError("nb_iter_per_round must be >= 1")
        self._s.nb_iter_per_round = value

    @property
    def tol(self):
        """A row has converged when ``||F||inf < tol``, the residual in per unit
        (what ``residuals`` / ``last_residuals()`` report and ``violation_tol``
        compares): decides when it leaves (continuous) and its
        :class:`RowStatus` (both schedules). lightsim2grid's comparison, but
        not its unit: its ``tol`` is in MVA (``||F||inf < tol / sn_mva``).
        ``None`` restores the default (1e-8, 1e-3 in an FP32 build). Takes
        effect on the next run()."""
        return self._s.tol

    @tol.setter
    def tol(self, value):
        value = default_tol() if value is None else float(value)
        if not (value > 0.0 and np.isfinite(value)):
            raise ValueError("tol must be a positive finite number")
        self._s.tol = value

    def get_row_iterations(self):
        """(n_rows,) int ndarray: Newton iterations each row ran in the last
        run() (``nb_iter`` for every simulated row of a chunked run; a
        multiple of ``nb_iter_per_round`` in a continuous one; 0 for a row
        never simulated)."""
        return np.asarray(self._s.get_row_iterations(), dtype=np.int64)

    def get_row_status(self):
        """(n_rows,) int ndarray of :class:`RowStatus` codes of the last
        run()."""
        return np.asarray(self._s.get_row_status(), dtype=np.int64)

    def _warn_tol_above_violation_tol(self):
        vt = getattr(self, "violation_tol", None)
        if (vt is not None and self.scheduling == "continuous"
                and getattr(self, "compute_limit_violations", False) and self.tol > vt):
            warnings.warn(
                f"tol ({self.tol:g}) > violation_tol ({vt:g}): a row that leaves "
                "CONVERGED with a residual above violation_tol still gets a DIVERGENCE "
                "record", RuntimeWarning, stacklevel=3)


class SchedulingFacadeMixin:
    """For the ``*GPU`` facades: expects ``self._inner`` (an engine carrying
    :class:`SchedulingEngineMixin`)."""

    @property
    def scheduling(self):
        """See :attr:`SchedulingEngineMixin.scheduling`."""
        return self._inner.scheduling

    @scheduling.setter
    def scheduling(self, value):
        self._inner.scheduling = value

    @property
    def nb_iter_per_round(self):
        """See :attr:`SchedulingEngineMixin.nb_iter_per_round`."""
        return self._inner.nb_iter_per_round

    @nb_iter_per_round.setter
    def nb_iter_per_round(self, value):
        self._inner.nb_iter_per_round = value

    @property
    def tol(self):
        """See :attr:`SchedulingEngineMixin.tol`."""
        return self._inner.tol

    @tol.setter
    def tol(self, value):
        self._inner.tol = value

    def get_row_iterations(self):
        """(n_rows,) int ndarray: Newton iterations each row ran in the last
        compute() (see :meth:`SchedulingEngineMixin.get_row_iterations`)."""
        return self._inner.get_row_iterations()

    def get_row_status(self):
        """(n_rows,) int ndarray of :class:`RowStatus` codes of the last
        compute()."""
        return self._inner.get_row_status()

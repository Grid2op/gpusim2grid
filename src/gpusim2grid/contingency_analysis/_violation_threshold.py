# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``violation_threshold``: lightsim2grid's knob of the same name and semantics, shared by
``ContingencyAnalysisGPU`` and ``ScenarioSweepGPU``.

Every checked limit moves towards a healthy anchor::

    effective = t * limit + (1 - t) * anchor

anchor 0 for currents, the bus nominal voltage clamped into ``[vmin, vmax]`` for both voltage
bounds (so the two bounds stay independent and never cross). The kernels check the effective
limits; the records keep the configured one (``restore_limits``)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from ._limit_violations import LimitViolation, LimitViolationType, ViolationElementType

__all__ = [
    "LimitsSession",
    "ThresholdHost",
    "ViolationThresholdFacadeMixin",
    "check_threshold",
    "effective_limits",
    "restore_limits",
]

#: (bus vmin kV, bus vmax kV, branch limit side 1 kA, side 2 kA, n_lines)
Limits = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]


class LimitsSession(Protocol):
    """What the mixin needs of a facade's engine (``_inner``): taking limits."""

    def set_limits(
        self,
        bus_vmin_kv: np.ndarray,
        bus_vmax_kv: np.ndarray,
        branch_limit_a1_ka: np.ndarray,
        branch_limit_a2_ka: np.ndarray,
        n_lines: int,
    ) -> None: ...


def check_threshold(value: Any) -> float:
    """``value`` as a float in ``]0, 1]``, else ``ValueError`` (lightsim2grid's messages)."""
    if isinstance(value, (str, bytes)):
        raise ValueError("The `violation_threshold` attribute must be a real number.")
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ValueError("The `violation_threshold` attribute must be a real number.") from None
    if not 0.0 < t <= 1.0:
        raise ValueError(f"The `violation_threshold` attribute must be in the range ]0., 1.] (got {t}).")
    return t


def effective_limits(
    threshold: float,
    bus_vn_kv: np.ndarray,
    bus_vmin_kv: np.ndarray,
    bus_vmax_kv: np.ndarray,
    limit_a1_ka: np.ndarray,
    limit_a2_ka: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The limits the checks use at ``threshold`` (NaN = not configured stays NaN). A bus with
    ``vmin > vmax`` raises ``RuntimeError``, as lightsim2grid."""
    t = float(threshold)
    vmin = np.asarray(bus_vmin_kv, dtype=float)
    vmax = np.asarray(bus_vmax_kv, dtype=float)
    vn = np.asarray(bus_vn_kv, dtype=float)
    bad = np.isfinite(vmin) & np.isfinite(vmax) & (vmin > vmax)
    if bad.any():
        raise RuntimeError(
            f"inconsistent voltage limits (vmin > vmax) at buses {np.flatnonzero(bad).tolist()}"
        )
    anchor = vn.copy()
    anchor = np.where(np.isfinite(vmin), np.maximum(anchor, vmin), anchor)
    anchor = np.where(np.isfinite(vmax), np.minimum(anchor, vmax), anchor)
    eff_vmin = t * vmin + (1.0 - t) * anchor
    eff_vmax = t * vmax + (1.0 - t) * anchor
    eff_a1 = t * np.asarray(limit_a1_ka, dtype=float)
    eff_a2 = t * np.asarray(limit_a2_ka, dtype=float)
    return eff_vmin, eff_vmax, eff_a1, eff_a2


def restore_limits(
    rows: Sequence[Sequence[LimitViolation]],
    bus_vmin_kv: np.ndarray,
    bus_vmax_kv: np.ndarray,
    limit_a1_ka: np.ndarray,
    limit_a2_ka: np.ndarray,
    n_lines: int,
) -> list[list[LimitViolation]]:
    """``rows`` with the configured limit in every voltage / current record (the kernels wrote
    the effective one)."""

    def fix(v: LimitViolation) -> LimitViolation:
        vt = v.violation_type
        if vt == LimitViolationType.LOW_VOLTAGE:
            return dataclasses.replace(v, limit=float(bus_vmin_kv[v.element_id]))
        if vt == LimitViolationType.HIGH_VOLTAGE:
            return dataclasses.replace(v, limit=float(bus_vmax_kv[v.element_id]))
        if vt == LimitViolationType.CURRENT:
            branch = v.element_id + (n_lines if v.element_type == ViolationElementType.TRAFO else 0)
            lim = limit_a1_ka if v.side == 1 else limit_a2_ka
            return dataclasses.replace(v, limit=float(lim[branch]))
        return v

    return [[fix(v) for v in row] for row in rows]


class ThresholdHost(Protocol):
    """What ``ViolationThresholdFacadeMixin`` needs of its facade (read-only properties, so a
    facade's more specific engine type fits)."""

    @property
    def _inner(self) -> LimitsSession: ...
    @property
    def _grid(self) -> Any: ...

    _violation_threshold: float
    _threshold_pushed: tuple[float, int]

    def _extract_limits_arrays(self) -> Limits: ...
    def _configured_limits(self) -> Limits: ...
    def _restore_limits(self, rows: Sequence[Sequence[LimitViolation]]) -> list[list[LimitViolation]]: ...


class ViolationThresholdFacadeMixin:
    """``violation_threshold`` on a batch facade (``ContingencyAnalysisGPU``,
    ``ScenarioSweepGPU``). The facade calls ``_push_effective_limits()`` at the start of
    ``compute()`` and ``_restore_limits`` on the records it returns; ``get_violations`` raises
    while ``_violations_valid`` is False.

    What the host facade must provide is the :class:`ThresholdHost` protocol: the methods that
    need it are typed ``self: ThresholdHost``, so the type checker verifies each facade at the
    call site without narrowing the facade's own attributes. ``_limits_args`` (limits given
    explicitly) and ``_limits_gen`` (bumped when they change) are optional."""

    _violation_threshold: float = 1.0
    _threshold_pushed: tuple[float, int] = (1.0, -1)
    _violations_valid: bool = True

    @property
    def violation_threshold(self) -> float:
        """float in ``]0, 1]``, default 1.0: lightsim2grid's knob of the same name and
        semantics -- every checked limit moves towards a healthy anchor,
        ``t * limit + (1 - t) * anchor`` (0 for currents, the bus nominal voltage clamped into
        ``[vmin, vmax]`` for voltages); records keep the configured limit. Lowering it
        invalidates the computed violations, raising it does not. Next compute()."""
        return self._violation_threshold

    @violation_threshold.setter
    def violation_threshold(self, value: Any) -> None:
        t = check_threshold(value)
        if t < self._violation_threshold:
            self._violations_valid = False
        self._violation_threshold = t

    def _configured_limits(self: ThresholdHost) -> Limits:
        """(vmin, vmax, limit_a1, limit_a2, n_lines): the limits given, else the grid's."""
        given = getattr(self, "_limits_args", None)
        if given is not None:
            return given
        return self._extract_limits_arrays()

    def _push_effective_limits(self: ThresholdHost) -> None:
        """Give the session the limits at ``violation_threshold`` when it or the configured
        limits changed since the last push (the configured ones back at 1.0)."""
        t = self._violation_threshold
        key = (t, int(getattr(self, "_limits_gen", 0)))
        if key == self._threshold_pushed or (t == 1.0 and self._threshold_pushed[0] == 1.0):
            return
        from .._ls2g_utils import extract_branch_data

        vmin, vmax, a1, a2, n_lines = self._configured_limits()
        if t != 1.0:
            branch_args, _, _ = extract_branch_data(self._grid)
            vmin, vmax, a1, a2 = effective_limits(t, branch_args[6], vmin, vmax, a1, a2)
        self._inner.set_limits(vmin, vmax, a1, a2, int(n_lines))
        self._threshold_pushed = key

    def _restore_limits(
        self: ThresholdHost, rows: Sequence[Sequence[LimitViolation]]
    ) -> list[list[LimitViolation]]:
        if self._violation_threshold == 1.0:
            return [list(r) for r in rows]
        vmin, vmax, a1, a2, n_lines = self._configured_limits()
        return restore_limits(rows, vmin, vmax, a1, a2, int(n_lines))

    def _threshold_violations_n(
        self: ThresholdHost,
        compute_n: Any,
        bus_vn_kv: np.ndarray,
        limits: tuple[Any, Any, Any, Any],
        n_branch: int,
    ) -> list[LimitViolation]:
        """The "n" check at the threshold: ``compute_n(vmin, vmax, a1, a2)`` with the effective
        limits (``None`` = not configured stays ``None``), records restored."""
        t = self._violation_threshold
        if t == 1.0:
            return compute_n(*limits)
        nan_b = np.full(len(bus_vn_kv), np.nan)
        nan_l = np.full(n_branch, np.nan)
        filled = [
            nan_b if limits[0] is None else limits[0],
            nan_b if limits[1] is None else limits[1],
            nan_l if limits[2] is None else limits[2],
            nan_l if limits[3] is None else limits[3],
        ]
        eff = effective_limits(t, bus_vn_kv, *filled)
        out = compute_n(*(None if c is None else e for c, e in zip(limits, eff)))
        return self._restore_limits([out])[0]

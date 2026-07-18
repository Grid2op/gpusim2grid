# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared grid setup utilities for benchmark scripts."""

import numpy as np
import warnings
import grid2op
import pandapower.networks as pn
from lightsim2grid import LightSimBackend
from lightsim2grid.gridmodel import init_from_pandapower
from lightsim2grid.solver import SolverType

_GRID2OP_TEST_ENVS = set(grid2op.list_available_test_env())


def load_grid(grid_name):
    """Load a grid by name; return (grid, v_init, v_res, n_sub, v_init_ca).

    Parameters
    ----------
    grid_name : str
        Either a grid2op test environment name (e.g. ``"l2rpn_idf_2023"``)
        or a pandapower network function name (e.g. ``"case6515rte"``).

    Returns
    -------
    grid : lightsim2grid GridModel
    v_init : complex ndarray, shape (n_sub,)
        DC-initialised voltage vector (n_sub buses).
    v_res : complex ndarray, shape (n_sub,)
        AC power-flow solution used as reference.
    n_sub : int
        Number of buses in the physical grid.
    v_init_ca : complex ndarray
        Initial voltage vector for contingency analysis.
        For grid2op grids (doubled-bus model) this is ``[v_init, v_init]``;
        for pandapower grids it is a copy of ``v_init``.

    Raises
    ------
    ValueError
        If ``grid_name`` is neither a known grid2op test environment nor
        an attribute of ``pandapower.networks``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if grid_name in _GRID2OP_TEST_ENVS:
            return _load_grid2op(grid_name)
        if hasattr(pn, grid_name):
            return _load_pandapower(grid_name)
    raise ValueError(
        f"{grid_name!r} is not a known grid2op test environment "
        f"(see grid2op.list_available_test_env()) "
        f"nor a pandapower.networks attribute."
    )


def _load_grid2op(env_name):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = grid2op.make(env_name, backend=LightSimBackend(), test=True)
        env.reset(seed=0, options={"time serie id": 0})
    grid = env.backend._grid
    n_sub = type(env).n_sub
    v_init = env.backend._debug_Vdc[:n_sub].copy()
    v_res = env.backend.V[:n_sub].copy()
    v_init_ca = np.concat([v_init, v_init])
    vn_kv = env.backend._grid.get_bus_vn_kv()[:n_sub]
    return grid, v_init, v_res, n_sub, v_init_ca, vn_kv


def _load_pandapower(case_name):
    grid = init_from_pandapower(getattr(pn, case_name)())
    grid.change_solver(SolverType.KLU)
    v_init_dc = np.ones(shape=grid.get_bus_vn_kv().shape[0], dtype=complex)
    v_init = grid.dc_pf(v_init_dc, 1, 1)
    v_res = grid.ac_pf(v_init, 10, 1e-8)
    n_sub = grid.get_bus_vn_kv().shape[0]
    v_init_ca = 1. * v_init
    vn_kv =  grid.get_bus_vn_kv().copy()
    return grid, v_init, v_res, n_sub, v_init_ca, vn_kv

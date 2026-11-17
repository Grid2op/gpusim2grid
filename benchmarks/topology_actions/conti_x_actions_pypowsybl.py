# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Contingency x remedial action screen of a pypowsybl network, in one compute(), optionally
compared with reference results (docs/dev_notes/topology_actions_plan.md § 8).

Inputs: a network file pypowsybl reads, a JSON list of contingency branch ids, and actions /
operator strategies in pypowsybl's JSON names (``{"actions": [...], "operator_strategies":
[{"id", "contingency_id", "action_ids"}, ...]}``; SWITCH and PHASE_TAP_CHANGER_POSITION).

- The grid: lightsim2grid's pypowsybl loader with OpenLoadFlow's loading rules (as
  ``benchmarks/outer_loops.py``), every branch's current limits set to pypowsybl's permanent
  ones (both sides), the outer loops on every row.
- Switch actions are translated outside the solver: each distinct action set is applied to a
  pypowsybl variant, and the new bus *partition* of every touched voltage level (pypowsybl
  renames buses, ids are not comparable) becomes an ``Action.from_partition``. A phase tap
  position becomes the transformer's ratio / shift, read off a copy of the grid after
  lightsim2grid's ``change_trafo_phase_tap``.
- One ``compute()`` solves the contingencies and every strategy, a contingency releasing its
  strategies as it finishes (per-row results stay on the device), each resuming from its own
  contingency's converged state -- voltages and outer-loop state -- unless ``--no-warm-start``.

Output: one CSV row per situation (contingency_id, operator_strategy_id, kind, converged,
reason, max_rho, branch_id); with ``--reference`` (a CSV with contingency_id,
operator_strategy_id, status, max_rho, branch_id), a comparison per action kind.

    CUDA_VISIBLE_DEVICES=0 tools/run_limited.sh -m 64 -c 8 -- python \\
        benchmarks/topology_actions/conti_x_actions_pypowsybl.py NETWORK CONTINGENCIES.json \\
        ACTIONS.json --out OUT_DIR [--reference REF.csv]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gpusim2grid import ContingencyAnalysisGPU

# absolute: the editor stubs cover the compiled module, not the package's relative re-export
from gpusim2grid._gpusim2grid import warmup
from gpusim2grid.actions import Action, ElementRef, substation_elements

if TYPE_CHECKING:
    import pandas as pd
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    from pypowsybl.network.impl.network import Network

#: pypowsybl element kind -> (getter, [(bus column, its voltage level column)])
_PP_KINDS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "line": ("get_lines", [("bus1_id", "voltage_level1_id"), ("bus2_id", "voltage_level2_id")]),
    "2wt": (
        "get_2_windings_transformers",
        [("bus1_id", "voltage_level1_id"), ("bus2_id", "voltage_level2_id")],
    ),
    "load": ("get_loads", [("bus_id", "voltage_level_id")]),
    "gen": ("get_generators", [("bus_id", "voltage_level_id")]),
    "shunt": ("get_shunt_compensators", [("bus_id", "voltage_level_id")]),
}


# ------------------------------------------------------------------------------------- grid
def load_grid(net: Network) -> LSGrid:
    """lightsim2grid's grid of ``net`` (OpenLoadFlow's loading rules, NROuter_KLU, solved), every
    branch limited at pypowsybl's permanent current limits."""
    from lightsim2grid.lightsim2grid_cpp import ScalingPolicyType
    from lightsim2grid.network.from_pypowsybl import OlfLoadingParameters
    from lightsim2grid.network.from_pypowsybl import init as init_from_pypowsybl
    from lightsim2grid.network.from_pypowsybl._aux_add_slack import _default_distributed_slack

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        slack = _default_distributed_slack(net, net.get_generators())
        if not slack:
            raise RuntimeError("no distributed slack participant in the network")
        gen_slack_id = next(iter(slack))
        grid = init_from_pypowsybl(
            net,
            gen_slack_id=gen_slack_id,
            sort_index=False,
            buses_for_sub=False,
            keep_half_open_lines=True,
            fuse_zero_impedance_branches=True,
            init_vm_pu=1.0,
            # annotated bool by lightsim2grid, documented (and used) as bool | OlfLoadingParameters
            olf_rules=OlfLoadingParameters(reactive_limits=True, svc_voltage_monitoring=True),  # pyright: ignore[reportArgumentType]
        )
    grid.change_algorithm("NROuter_KLU")
    grid.set_keep_vinit_at_group_controlled_buses(True)
    cfg = grid.get_ac_algo_config()
    ip, rp = list(cfg.int_params), list(cfg.real_params)
    ip[0] = int(ScalingPolicyType.MaxVoltageChange)
    rp[0], rp[1] = 1.0, 0.4
    cfg.int_params, cfg.real_params = ip, rp
    grid.set_ac_algo_config(cfg)
    _set_permanent_limits(net, grid)
    flat = np.full(grid.total_bus(), 1.0 + 0j)
    grid.set_dc_distribute_slack_on_can_participate(True)
    v_dc = grid.dc_pf(flat, 50, 1e-8)
    v0 = np.exp(1j * np.angle(v_dc)) if v_dc.shape[0] > 0 else flat
    if grid.ac_pf(v0, 30, 1e-6).shape[0] == 0:
        raise RuntimeError("the base case diverged")
    return grid


def _set_permanent_limits(net: Network, grid: LSGrid) -> None:
    """Every line / transformer side's limit = pypowsybl's permanent CURRENT limit (kA)."""
    lim = net.get_operational_limits(all_attributes=True).reset_index()
    lim = lim[(lim["type"] == "CURRENT") & (lim["acceptable_duration"] == -1)]
    ids, sides, amps = lim["element_id"], lim["side"], lim["value"]
    perm: dict[tuple[str, str], float] = {
        (str(e), str(s)): float(v) * 1e-3 for e, s, v in zip(ids, sides, amps)
    }
    for container, set1, set2 in (
        (grid.get_lines(), grid.set_line_current_limit_side1, grid.set_line_current_limit_side2),
        (grid.get_trafos(), grid.set_trafo_current_limit_side1, grid.set_trafo_current_limit_side2),
    ):
        names = [el.name for el in container]
        set1(np.array([perm.get((n, "ONE"), np.nan) for n in names]))
        set2(np.array([perm.get((n, "TWO"), np.nan) for n in names]))


def branch_names(grid: LSGrid) -> list[str]:
    """Branch names in lightsim2grid's numbering: lines, then transformers."""
    return [ln.name for ln in grid.get_lines()] + [t.name for t in grid.get_trafos()]


def _ref_by_pp_end(grid: LSGrid) -> dict[tuple[str, str], ElementRef]:
    """(pypowsybl id, bus column) -> lightsim2grid element end."""
    out: dict[tuple[str, str], ElementRef] = {}
    for ln in grid.get_lines():
        out[(ln.name, "bus1_id")] = ("line_bus1", ln.id)
        out[(ln.name, "bus2_id")] = ("line_bus2", ln.id)
    for t in grid.get_trafos():
        out[(t.name, "bus1_id")] = ("trafo_bus1", t.id)
        out[(t.name, "bus2_id")] = ("trafo_bus2", t.id)
    for field, container in (
        ("load_bus", grid.get_loads()),
        ("gen_bus", grid.get_generators()),
        ("shunt_bus", grid.get_shunts()),
    ):
        for el in container:
            out[(el.name, "bus_id")] = (field, el.id)
    return out


# ---------------------------------------------------------------------------- translation
def _end_voltage_levels(net: Network) -> dict[tuple[str, str], str]:
    """(pypowsybl id, bus column) -> its voltage level (fixed, whatever the switches)."""
    out: dict[tuple[str, str], str] = {}
    for getter, cols in _PP_KINDS.values():
        df = getattr(net, getter)(attributes=[vl for _, vl in cols])
        for col, vl in cols:
            for eid, v in df[vl].items():
                out[(str(eid), col)] = str(v)
    return out


def _bus_view(net: Network) -> dict[tuple[str, str], str | None]:
    """(pypowsybl id, bus column) -> its bus id in the bus view (None: disconnected)."""
    out: dict[tuple[str, str], str | None] = {}
    for getter, cols in _PP_KINDS.values():
        df = getattr(net, getter)(attributes=[c for c, _ in cols])
        for col, _ in cols:
            for eid, bus in df[col].items():
                out[(str(eid), col)] = bus if isinstance(bus, str) and bus else None
    return out


def translate(
    net: Network, grid: LSGrid, actions: dict[str, dict[str, Any]], combos: list[tuple[str, ...]]
) -> tuple[list[Action | None], list[str], list[str]]:
    """One ``Action`` per action set (``None`` when it cannot be expressed: reason given), and its
    kind: noop / disconnect / move / pst / mixed."""
    by_end = _ref_by_pp_end(grid)
    pp_of_ref: dict[ElementRef, tuple[str, str]] = {ref: end for end, ref in by_end.items()}
    trafo_id: dict[str, int] = {t.name: t.id for t in grid.get_trafos()}
    sub_of: dict[str, int] = {name: s for s, name in enumerate(grid.get_substation_names())}
    vl_of_end = _end_voltage_levels(net)
    sw_open = net.get_switches(attributes=["open"])["open"]
    base = _bus_view(net)
    variant = net.get_working_variant_id()
    net.clone_variant(variant, "g2g_translate", True)
    out: list[Action | None] = []
    kinds: list[str] = []
    reasons: list[str] = []
    for combo in combos:
        parts: list[Action] = []
        kind_set: set[str] = set()
        why = ""
        switches = [actions[a] for a in combo if actions[a]["type"] == "SWITCH"]
        for a in combo:
            act = actions[a]
            if act["type"] == "PHASE_TAP_CHANGER_POSITION":
                if act.get("is_relative"):
                    why = why or f"{a}: relative tap positions are not translated"
                    continue
                t = trafo_id.get(act["transformer_id"])
                if t is None:
                    why = why or f"{a}: transformer not in the grid"
                    continue
                changed = copy.deepcopy(grid)
                changed.change_trafo_phase_tap(t, int(act["tap_position"]))
                tr = changed.get_trafos()[t]
                parts.append(
                    Action(trafo_shift_deg={t: float(np.rad2deg(tr.shift_rad))}, trafo_ratio={t: tr.ratio})
                )
                kind_set.add("pst")
            elif act["type"] != "SWITCH":
                why = why or f"{a}: action type {act['type']} not translated"
        if switches:
            net.set_working_variant("g2g_translate")
            try:
                for sw in switches:
                    net.update_switches(id=sw["switch_id"], open=bool(sw["open"]))
                after = _bus_view(net)
            except Exception as exc:  # an element of the action not in this network
                after, why = {}, why or f"switch: {exc}"
            finally:
                for sw in switches:
                    if sw["switch_id"] in sw_open.index:
                        net.update_switches(id=sw["switch_id"], open=bool(sw_open[sw["switch_id"]]))
                net.set_working_variant(variant)
            touched = {end for end in base if after and base[end] != after.get(end)}
            # a voltage level whose partition the switches change: any element end that moved
            vls: set[str] = {vl_of_end[e] for e in touched}
            for vl in sorted(vls):
                sub = sub_of.get(vl)
                if sub is None:
                    why = why or f"voltage level {vl} not a lightsim2grid substation"
                    continue
                groups: dict[str, list[ElementRef]] = defaultdict(list)
                off: list[ElementRef] = []
                for ref in substation_elements(grid, sub):
                    end = pp_of_ref.get(ref)
                    if end is None:  # not a pypowsybl element we read: keeps its bus
                        groups[f"ls2g:{ref}"].append(ref)
                    elif after.get(end) is None:
                        off.append(ref)
                    else:
                        groups[str(after[end])].append(ref)
                try:
                    act = Action.from_partition(grid, sub, list(groups.values()), off)
                except ValueError as exc:
                    why = why or f"voltage level {vl}: {exc}"
                    continue
                parts.append(act)
                moved = any(b != -1 for _, _, b in act.bus_items())
                kind_set.add("move" if moved else "disconnect" if not act.is_empty() else "noop")
        try:
            action = Action.combine(parts) if parts else Action()
        except ValueError as exc:
            action, why = None, why or str(exc)
        if action is not None and action.is_empty():
            kind_set = {"noop"}
        kinds.append(
            "mixed" if len(kind_set - {"noop"}) > 1 else (sorted(kind_set - {"noop"}) or ["noop"])[0]
        )
        out.append(None if why else action)
        reasons.append(why)
    net.remove_variant("g2g_translate")
    return out, kinds, reasons


# ------------------------------------------------------------------------------------ run
def run(args: argparse.Namespace) -> pd.DataFrame:
    import pandas as pd
    import pypowsybl as pp

    t0 = time.perf_counter()
    net = pp.network.load(args.network)  # pyright: ignore[reportPrivateImportUsage] (pypowsybl's export)
    grid = load_grid(net)
    t_load = time.perf_counter() - t0
    names = branch_names(grid)
    index = {n: i for i, n in enumerate(names)}
    ctg_ids: list[str] = [c for c in json.loads(Path(args.contingencies).read_text()) if c in index]
    spec = json.loads(Path(args.actions).read_text())
    actions = {a["id"]: a for a in spec["actions"]}
    strategies = [s for s in spec["operator_strategies"] if s["contingency_id"] in set(ctg_ids)]
    combos = sorted({tuple(s["action_ids"]) for s in strategies})
    t0 = time.perf_counter()
    translated, kinds, why_not = translate(net, grid, actions, combos)
    t_translate = time.perf_counter() - t0
    print(
        f"load {t_load:.1f} s; {len(ctg_ids)} contingencies, {len(strategies)} strategies, "
        f"{len(combos)} action sets translated in {t_translate:.1f} s: {dict(Counter(kinds))}, "
        f"{sum(1 for a in translated if a is None)} not expressible"
    )

    warmup()
    kw: dict[str, Any] = {"linear_solver": "own_lu"} if args.settings == "perf" else {}
    ca = ContingencyAnalysisGPU(
        grid,
        outer_loops=True,
        nb_iter=30,
        init_from_n_powerflow=False,
        handle_disconnected_grid=True,
        tol=1e-6 / grid.get_sn_mva(),
        ir_n_steps=1,
        strategy_warm_start=not args.no_warm_start,
        **kw,
    )
    if args.settings == "perf":
        ca.tail_max_groups, ca.tail_group_size = args.batch, 1
    ca.add_contingencies_by_branch_id([[index[c]] for c in ctg_ids])
    expressible = [i for i, a in enumerate(translated) if a is not None]
    combo_action = {combos[i]: k for k, i in enumerate(expressible)}
    chosen: list[Action] = [a for a in translated if a is not None]
    ca.set_actions(chosen)
    ctg_pos = {c: k for k, c in enumerate(ctg_ids)}
    rows = [s for s in strategies if tuple(s["action_ids"]) in combo_action]
    ca.add_strategies(
        np.array([ctg_pos[s["contingency_id"]] for s in rows]),
        np.array([combo_action[tuple(s["action_ids"])] for s in rows]),
    )

    out: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    ca.compute(batch_size=args.batch)
    t_compute = time.perf_counter() - t0
    res, reasons = ca.last_residuals(), ca.get_batch_unsupported_reasons()
    rho, br = ca.get_max_loading()  # per row: the contingencies, then the strategies
    n_ctg: int = len(ctg_ids)
    for k, c in enumerate(ctg_ids):
        out.append(
            dict(
                contingency_id=c,
                operator_strategy_id="",
                kind="",
                converged=bool(res[k] <= 1e-6),
                reason="",
                max_rho=rho[k],
                branch_id=names[br[k]] if br[k] >= 0 else "",
            )
        )
    for k, s in enumerate(rows, start=n_ctg):
        out.append(
            dict(
                contingency_id=s["contingency_id"],
                operator_strategy_id=s["id"],
                kind=kinds[combos.index(tuple(s["action_ids"]))],
                converged=bool(res[k] <= 1e-6),
                reason=reasons[k],
                max_rho=rho[k],
                branch_id=names[br[k]] if br[k] >= 0 else "",
            )
        )
    t_results = time.perf_counter() - t0 - t_compute
    for s in strategies:
        k = combos.index(tuple(s["action_ids"]))
        if translated[k] is None:
            out.append(
                dict(
                    contingency_id=s["contingency_id"],
                    operator_strategy_id=s["id"],
                    kind=kinds[k],
                    converged=False,
                    reason=why_not[k],
                    max_rho=np.nan,
                    branch_id="",
                )
            )
    print(f"compute {t_compute:.1f} s ({n_ctg} contingencies, {len(rows)} strategies)")
    print(f"results {t_results:.1f} s")
    return pd.DataFrame(out)


def compare(df: pd.DataFrame, reference: str) -> None:
    """Per action kind: situations both sides solved, status agreement, |max rho| difference,
    same most loaded branch."""
    import pandas as pd

    ref = pd.read_csv(reference, keep_default_na=False)
    ref["max_rho"] = pd.to_numeric(ref["max_rho"], errors="coerce")
    m = df.merge(ref, on=["contingency_id", "operator_strategy_id"], suffixes=("", "_ref"))
    m["kind"] = m["kind"].replace("", "contingency")
    print(f"\ncompared with {reference}: {len(m)} situations in both")
    print(
        f"{'kind':<12}{'rows':>8}{'unsupported':>13}{'status =':>10}{'|drho| p50':>12}{'p99':>10}{'max':>10}"
        f"{'same branch':>13}"
    )
    for kind, g in m.groupby("kind"):
        run = g[g["reason"] == ""]
        both = run[run["converged"] & (run["status"] == "CONVERGED")]
        d = np.abs(both["max_rho"] - both["max_rho_ref"])
        same = (both["branch_id"] == both["branch_id_ref"]).mean() if len(both) else np.nan
        agree = (run["converged"] == (run["status"] == "CONVERGED")).mean() if len(run) else np.nan
        print(
            f"{kind:<12}{len(g):>8}{len(g) - len(run):>13}{agree:>10.4f}"
            f"{d.quantile(0.5) if len(d) else np.nan:>12.2e}{d.quantile(0.99) if len(d) else np.nan:>10.2e}"
            f"{d.max() if len(d) else np.nan:>10.2e}{same:>13.4f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("network")
    p.add_argument("contingencies", help="JSON list of branch ids")
    p.add_argument("actions", help="pypowsybl actions / operator_strategies JSON")
    p.add_argument("--out", required=True, help="output directory (results.csv)")
    p.add_argument(
        "--reference", help="CSV: contingency_id, operator_strategy_id, status, max_rho, branch_id"
    )
    p.add_argument(
        "--settings",
        choices=["perf", "defaults"],
        default="perf",
        help="perf: own LU, the tail from the start; defaults: cuDSS, the default tail",
    )
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="strategies from the base case (default: each strategy resumes from its contingency's "
        "converged state, outer loops included, as OpenLoadFlow's security analysis)",
    )
    args = p.parse_args()
    df = run(args)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / "results.csv", index=False)
    if args.reference:
        compare(df, args.reference)


if __name__ == "__main__":
    main()

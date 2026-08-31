# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import warnings
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from matpowercaseframes import CaseFrames
from lightsim2grid.network import init_from_matpower
from scenario_utils import (
    get_args,
)
from get_init_method import get_init_method


class KCLChecker(object):
    """The dedicated class, implemented independantly of all the benchmarked solvers that check
    whether a given complex voltages meet the KCL for a given matpower test case (read thanks to matpowercaseframes).
    
    This has been implemented solely based on the matpower manual to check the equation 1) of the paper.
    """
    def __init__(self, path, tol=1e-9):
        self.tol = tol
        self.caseframes = CaseFrames(path)
        self.caseframes.bus["id"] = np.arange(self.caseframes.bus.shape[0])
        self.nb_bus = self.caseframes.bus.shape[0]
        
        # fill shunt and ybus data
        self.shunt_bus = None
        self.shunt_data = None
        self.branch_fbus = None
        self.branch_tbus = None
        self.branch_data = None  # nbranches x 4 (yff, yft, ytf, ytt)
        self.branch_status = None 
        self.base_ybus = self.build_ybus()
        
        # fill sbus data
        self.load_bus = None
        self.gen_bus = None
        self.base_sbus = self.build_sbus()
        
        # compute indices
        self.base_ref, self.base_pv, self.base_pq = self.build_bus_idx()
        
    def build_ybus(self, tol=1e-9):
        """build ybus from the matpower .m casefile."""
        baseMVA = self.caseframes.baseMVA
        bus = self.caseframes.bus
        branch = self.caseframes.branch

        # handle shunts (part of the diagonal of Ybus matrix)
        has_shunts = (bus["GS"].abs() > tol) | (bus["BS"].abs() > tol)

        data = []
        row_ind = []
        col_ind = []
        for indx, el in bus.loc[has_shunts, ["id", "GS", "BS"]].iterrows():
            data.append((el.GS + 1j * el.BS) / baseMVA)
            row_ind.append(int(el.id))
            col_ind.append(int(el.id))
        self.shunt_bus = np.array(row_ind, dtype=int)
        self.shunt_data = np.array(data, dtype=complex)

        # handle branches (lines or transformers)
        branch_fbus = []
        branch_tbus = []
        branch_data = []
        branch_status = []
        for indx, el in branch.iterrows():
            # extract data
            ys = 1. / (el.BR_R + 1j * el.BR_X)
            hs = 0.5j * el.BR_B
            tap = el.TAP if abs(el.TAP) > tol else 1.
            shift = np.deg2rad(el.SHIFT)
            eitheta_shift = np.exp(1j * shift)
            emitheta_shift = np.exp(-1j * shift)
            fbus_id = int(bus.loc[bus["BUS_I"] == el.F_BUS, "id"].to_numpy()[0])
            tbus_id = int(bus.loc[bus["BUS_I"] == el.T_BUS, "id"].to_numpy()[0])

            # compute the physics
            yff = (ys + hs) / (tap * tap)
            yft = -ys / tap * eitheta_shift
            ytf = -ys / tap * emitheta_shift
            ytt = ys + hs

            # save the data
            branch_fbus.append(fbus_id)
            branch_tbus.append(tbus_id)
            branch_data.append((yff, yft, ytf, ytt))
            branch_status.append(el.BR_STATUS)

        self.branch_fbus = np.array(branch_fbus, dtype=int)
        self.branch_tbus = np.array(branch_tbus, dtype=int)
        self.branch_status = np.array(branch_status, dtype=bool)
        self.branch_data = np.array(branch_data, dtype=complex).reshape(branch.shape[0], 4)

        return self._assemble_ybus(self.branch_status)

    def _assemble_ybus(self, branch_status):
        """Ybus for an arbitrary per-branch on/off mask (same column order as
        self.caseframes.branch, i.e. the matpower branch table row order),
        reusing the per-branch (fbus, tbus, yff, yft, ytf, ytt) already
        extracted by build_ybus(). `branch_status` is truthy/1 for a
        connected branch, falsy/0 for an outaged one -- lets a caller build a
        Ybus for a contingency (eg an N-1/N-2 scenario) without touching the
        case file's own BR_STATUS.
        """
        bus = self.caseframes.bus
        data = list(self.shunt_data)
        row_ind = list(self.shunt_bus)
        col_ind = list(self.shunt_bus)
        for l_id in np.flatnonzero(np.asarray(branch_status).astype(bool)):
            fbus_id = self.branch_fbus[l_id]
            tbus_id = self.branch_tbus[l_id]
            yff, yft, ytf, ytt = self.branch_data[l_id]
            data += [yff, yft, ytf, ytt]
            row_ind += [fbus_id, fbus_id, tbus_id, tbus_id]
            col_ind += [fbus_id, tbus_id, fbus_id, tbus_id]
        return csr_matrix((data, (row_ind, col_ind)), shape=(bus.shape[0], bus.shape[0]))

    def build_sbus(self, tol=1e-9):
        """build sbus from matpower file."""
        baseMVA = self.caseframes.baseMVA
        bus = self.caseframes.bus
        gen = self.caseframes.gen
        sbus = -(bus.PD + 1j * bus.QD).to_numpy() / baseMVA
        self.load_bus = bus.loc[(np.abs(bus["PD"]) > tol) | (np.abs(bus["QD"]) > tol), "id"].to_numpy()
        
        gen_buses = []
        for _, el in gen.loc[gen["GEN_STATUS"] != 0].iterrows():
            gen_bus = bus.loc[bus["BUS_I"] == el.GEN_BUS, "id"].to_numpy()[0]
            sbus[gen_bus] += el.PG / baseMVA
            gen_buses.append(gen_bus)
        self.gen_bus = np.array(gen_buses, dtype=int)
        return sbus

    def build_bus_idx(self, tol=1e-9):
        """build ref (called "slack" in the paper), pv and pq indices from matpower file."""
        bus = self.caseframes.bus
        gen = self.caseframes.gen
        online = (gen["GEN_STATUS"] != 0).to_numpy()
        has_online_gen = bus["BUS_I"].isin(gen.loc[online, "GEN_BUS"])
        is_ref = bus["BUS_TYPE"] == 3
        is_pv = ~is_ref & has_online_gen

        ref = bus.loc[is_ref, "id"].to_numpy()
        pv = bus.loc[is_pv, "id"].to_numpy()
        pq = bus.loc[~is_ref & ~is_pv, "id"].to_numpy()

        # for check_kcl_scenario(..., gen_status=...): self.pv_gen_cols[bus_id] is the list
        # of (raw gen-table, 0-based -- same column order as gen_p/gen_status) columns of
        # the originally-online generator(s) that conferred PV status on that bus. If ALL of
        # them go offline in a given scenario, that bus is no longer really
        # voltage-controlled and should be checked as PQ for that scenario instead.
        bus_id_of_gen = bus.set_index("BUS_I").reindex(gen["GEN_BUS"])["id"].to_numpy()
        self.pv_gen_cols = {int(b): [] for b in pv}
        for j, (bus_id, is_online) in enumerate(zip(bus_id_of_gen, online)):
            if is_online and int(bus_id) in self.pv_gen_cols:
                self.pv_gen_cols[int(bus_id)].append(j)

        return ref, pv, pq
    
    def _build_sbus(self, load_p, load_q, gen_p, gen_status=None):
        """Row-wise Sbus (n_scenario, nb_bus) from per-scenario load_p/load_q/gen_p,
        shared by check_kcl_injection() and check_kcl_scenario(). gen_p spans *all*
        generators (online and offline, like lsgrid.get_gen_res_full()), not just the
        online ones used in self.gen_bus / base_sbus.

        Pass `gen_status` (shape (n_scenario, n_gen), same raw gen-table column order as
        gen_p, 0/falsy for offline) to ignore (treat as 0) an offline generator's own
        gen_p entry rather than trust it -- confirmed empirically that pfdelta's own
        "pg_sol" field is NOT reliably 0 for an offline unit in every split (0 for its
        "nose"/test rows, but NaN for some of its "normal"/non-nose rows), so trusting it
        directly would silently poison Sbus -- and therefore every downstream KCL
        check, even the real-power one -- with NaN for that scenario.
        """
        baseMVA = self.caseframes.baseMVA
        bus = self.caseframes.bus
        gen = self.caseframes.gen
        n_scenario = load_p.shape[0]
        all_gen_bus = bus.set_index("BUS_I").loc[gen["GEN_BUS"], "id"].to_numpy()
        gen_p_eff = gen_p if gen_status is None else np.where(np.asarray(gen_status) != 0, gen_p, 0.)

        sbus = np.zeros((n_scenario, self.nb_bus), dtype=complex)
        sbus[:, self.load_bus] -= (load_p + 1j * load_q) / baseMVA
        scen_idx = np.repeat(np.arange(n_scenario), all_gen_bus.shape[0])
        bus_idx = np.tile(all_gen_bus, n_scenario)
        np.add.at(sbus, (scen_idx, bus_idx), (gen_p_eff / baseMVA).ravel())
        return sbus

    def check_kcl_injection(self, load_p, load_q, gen_p, v):
        """Per-bus KCL violation magnitude |dS_i| = sqrt(dp_i^2 + dq_i^2), one column per
        non-slack bus (pvpq, sorted) -- see check_kcl_scenario()'s docstring for the full
        rationale (this is its fixed-topology, no-`gen_status` special case).
        """
        sbus = self._build_sbus(load_p, load_q, gen_p)

        s_calc = v * np.conj(self.base_ybus @ v.T).T
        mismatch = s_calc - sbus

        pvpq = np.sort(np.concat((self.base_pv, self.base_pq)))
        base_pq_set = set(self.base_pq.tolist())
        is_pq = np.array([int(b) in base_pq_set for b in pvpq])
        delta_p = mismatch[:, pvpq].real
        delta_q = np.where(is_pq[None, :], mismatch[:, pvpq].imag, 0.)
        return np.sqrt(delta_p**2 + delta_q**2)

    def check_kcl_scenario(self, load_p, load_q, gen_p, v, branch_status, gen_status=None):
        """Like check_kcl_injection(), but each row also carries its own branch on/off
        mask (`branch_status`, shape (n_scenario, n_branch), same column order as
        self.caseframes.branch / topology.json's branch_order) instead of assuming
        self.base_ybus for every row -- eg for N-1/N-2 contingency scenarios where
        (almost) every row has a distinct topology. Builds one Ybus per distinct
        branch_status pattern via _assemble_ybus() rather than one per row, since
        scenarios sharing a topology are common even when injections differ.

        Returns the per-bus KCL violation magnitude |dS_i| = sqrt(dp_i^2 + dq_i^2), one
        column per non-slack bus (pvpq, sorted) -- the same per-bus combination PF-Delta's
        own eq. (5) uses (|dS_i| there), reduced HERE to what's actually meaningful for a
        classical AC-PF solver rather than an ML model: dp_i (real-power balance) is
        checked at every non-slack bus regardless of generator status, since P is always
        either a genuine specified input (load) or an enforced setpoint (generator), even
        at a PV bus. dq_i (reactive-power balance) is only meaningful at a bus with NO
        online, voltage-controlling generator -- at a PV bus, Q is not an independent
        input at all: it is *derived* from V by the exact same network equations this
        method itself uses to compute dS_i (the "physical solver only computes V and
        substitutes to get the rest" argument) -- so comparing the solver's re-derivation
        against itself would be tautologically zero for ANY v (even a wrong one), telling
        us nothing (confirmed empirically: skipping this bypass_lossy_trafo_conversion()
        gives a genuine ~0.2 p.u. mean / ~100 p.u. max KCL violation on case3120sp/
        case3375wp via the dp_i check alone -- there was never a need for a PV-bus dq_i
        term to catch that). dq_i is therefore forced to 0 at a PV bus, same effect as
        masking it out of the sqrt entirely, but expressed as a per-bus combination instead
        of two separately-averaged flat vectors -- this is what makes the dp_i and dq_i at
        the SAME pq bus combine correctly via sqrt(.^2+.^2) instead of being treated as two
        independent entries. The slack bus is excluded from `pvpq` entirely, for the same
        reason as PV-bus dq_i: both P and Q there are derived from V, not inputs.

        Pass `gen_status` (shape (n_scenario, n_gen), same raw gen-table column order as
        gen_p, 0/falsy for offline) to dynamically move a base_pv bus into the dq_i check
        for exactly the scenarios where its every originally-online controlling generator
        (self.pv_gen_cols, see build_bus_idx()) is offline: such a bus has no online
        generator left, so Q there really is now an ordinary, independently meaningful
        (load-only) specified quantity, not one derived from V -- the same reasoning as
        base_pq, just evaluated per scenario instead of fixed at construction time. Without
        `gen_status`, such a bus is always treated as PV -- still fully correct as long as
        no row has a generator outage (eg ls_pfdelta.py/gpu_pfdelta.py's own gen-outage-free
        rows; see ls_pfdelta.py's load_pfdelta_rows(drop_gen_outage=...)).
        """
        sbus = self._build_sbus(load_p, load_q, gen_p, gen_status=gen_status)

        s_calc = np.zeros_like(sbus)
        uniq, inv = np.unique(np.asarray(branch_status), axis=0, return_inverse=True)
        for grp in range(uniq.shape[0]):
            rows = inv == grp
            ybus = self._assemble_ybus(uniq[grp])
            s_calc[rows] = v[rows] * np.conj(ybus @ v[rows].T).T
        mismatch = s_calc - sbus

        pvpq = np.sort(np.concat((self.base_pv, self.base_pq)))
        delta_p = mismatch[:, pvpq].real
        base_pq_set = set(self.base_pq.tolist())

        if gen_status is None:
            is_pq = np.array([int(b) in base_pq_set for b in pvpq])
            delta_q = np.where(is_pq[None, :], mismatch[:, pvpq].imag, 0.)
            return np.sqrt(delta_p**2 + delta_q**2)

        gen_status = np.asarray(gen_status)
        is_pq_row = np.empty((load_p.shape[0], pvpq.size), dtype=bool)
        for col, bus_id in enumerate(pvpq):
            bus_id = int(bus_id)
            if bus_id in base_pq_set:
                is_pq_row[:, col] = True
            else:
                ctrl = self.pv_gen_cols[bus_id]
                is_pq_row[:, col] = (gen_status[:, ctrl] == 0).all(axis=1)

        delta_q = np.where(is_pq_row, mismatch[:, pvpq].imag, 0.)
        return np.sqrt(delta_p**2 + delta_q**2)


def kcl_mismatch_l2(mismatch):
    """Per-scenario L2 norm sqrt(sum(mis**2)) over buses -- a global, whole-network
    alternative to the per-bus mean/max PF-Delta's own eq. (5) reports (see
    check_kcl_injection()/check_kcl_scenario()'s docstrings for what each column already
    is: the per-bus KCL violation magnitude |dS_i|, always >= 0), for callers that want a
    single whole-network figure per scenario instead.
    """
    return np.sqrt(np.sum(np.square(mismatch), axis=1))


def summarize_kcl_mismatch(mismatch):
    """Summary stats (as plain python floats, ready to json.dump) of a
    check_kcl_injection()/check_kcl_scenario() result -- each column already is the
    per-bus KCL violation magnitude |dS_i| (see their own docstrings), so "mean"/"max"
    here reduce that directly: per-scenario mean/max (over buses) first, then aggregated
    across scenarios. This is PF-Delta's own eq. (5)-and-Table-3 metric (mean and maximum
    power mismatch across the dataset), adapted per check_kcl_scenario()'s docstring for
    what's actually meaningful to check for a classical AC-PF solver. kcl_mismatch_l2 is
    an additional, non-paper whole-network alternative -- see kcl_mismatch_l2()'s own
    docstring.
    """
    per_scenario_mean = mismatch.mean(axis=1)
    per_scenario_max = mismatch.max(axis=1)
    per_scenario_l2 = kcl_mismatch_l2(mismatch)
    return {
        "kcl_mismatch_mean": float(per_scenario_mean.mean()),
        "kcl_mismatch_std": float(per_scenario_mean.std()),
        "kcl_mismatch_max": float(per_scenario_max.max()),
        "kcl_mismatch_max_of_mean": float(per_scenario_mean.max()),
        "kcl_mismatch_p95_of_mean": float(np.percentile(per_scenario_mean, 95)),
        "kcl_mismatch_p99_of_mean": float(np.percentile(per_scenario_mean, 99)),
        "kcl_mismatch_p95_of_max": float(np.percentile(per_scenario_max, 95)),
        "kcl_mismatch_p99_of_max": float(np.percentile(per_scenario_max, 99)),
        "kcl_mismatch_l2_mean": float(per_scenario_l2.mean()),
        "kcl_mismatch_l2_std": float(per_scenario_l2.std()),
        "kcl_mismatch_l2_max": float(per_scenario_l2.max()),
        "kcl_mismatch_l2_p95": float(np.percentile(per_scenario_l2, 95)),
        "kcl_mismatch_l2_p99": float(np.percentile(per_scenario_l2, 99)),
    }


def kcl_checkpoint_sizes(n_scenario):
    """1-3-10 progression (100, 300, 1000, 3000, ...) capped at n_scenario,
    with n_scenario itself always appended last so the final checkpoint
    covers every solved scenario.
    """
    sizes = []
    base = 100
    while base < n_scenario:
        sizes.append(base)
        if 3 * base < n_scenario:
            sizes.append(3 * base)
        base *= 10
    sizes.append(n_scenario)
    return sizes


def summarize_kcl_mismatch_checkpoints(mismatch, checkpoint_sizes=None):
    """Same stats as summarize_kcl_mismatch(), computed on growing prefixes
    of the scenario axis (first `n` scenarios), keyed by `n`.
    """
    if checkpoint_sizes is None:
        checkpoint_sizes = kcl_checkpoint_sizes(mismatch.shape[0])
    return {n: summarize_kcl_mismatch(mismatch[:n]) for n in checkpoint_sizes}


def kcl_mismatch_per_scenario(mismatch, solved_mask, ndigits=6):
    """Per-scenario mean/max/L2 (over buses) of mismatch, ready to json.dump,
    index-aligned with the *full* (unfiltered) `solved_mask` -- unsolved
    scenarios get `None` rather than being dropped, so
    kcl_mismatch_mean_per_scenario[i] always refers to scenario i, same as
    converged_mask[i]. `mismatch` itself must already be filtered down to the
    solved scenarios only (check_kcl_injection's own convention throughout
    this codebase). Floats are rounded to `ndigits` significant figures --
    plenty for post-processing, and much lighter to store as plain json than
    full float64 precision. See summarize_kcl_mismatch()'s docstring for what `mismatch`'s
    columns already are (the per-bus KCL violation magnitude |dS_i|) and what "mean"/"max"
    mean here.
    """
    per_scenario_mean = mismatch.mean(axis=1)
    per_scenario_max = mismatch.max(axis=1)
    per_scenario_l2 = kcl_mismatch_l2(mismatch)

    def _round(x):
        return float(f"{x:.{ndigits}g}")

    n = solved_mask.shape[0]
    mean_per_scenario = [None] * n
    max_per_scenario = [None] * n
    l2_per_scenario = [None] * n
    for pos, idx in enumerate(np.flatnonzero(solved_mask)):
        mean_per_scenario[idx] = _round(per_scenario_mean[pos])
        max_per_scenario[idx] = _round(per_scenario_max[pos])
        l2_per_scenario[idx] = _round(per_scenario_l2[pos])

    return {
        "converged_mask": [bool(x) for x in solved_mask],
        "kcl_mismatch_mean_per_scenario": mean_per_scenario,
        "kcl_mismatch_max_per_scenario": max_per_scenario,
        "kcl_mismatch_l2_per_scenario": l2_per_scenario,
    }


if __name__ == "__main__":
    # sanity check of this function with a
    # powerflow
    # lightsim2grid is taken because it :
    # 1. is available directly in python
    # 2. has naturally the same order as the input matpower caseframe
    
    from ls_injection import (    
        BASE_EXPE_NAME,
        PF_ALGO,
        REF_PATH
    )
    args = get_args().parse_args()
    nb_pf_total = int(args.nb_pf)
    seed = int(args.seed)
    sample_data_meth = str(args.sample_data_meth)
    add_to_name = str(args.add_to_name)
    save_flows = args.save_flows
    save_voltages = args.save_voltages
    nb_thread = int(args.nb_threads)
    grids = [g.strip() for g in args.grids.split(",") if g.strip()]

    benchmark_results = {}
    nm_results = BASE_EXPE_NAME.format(sample_data_meth, nb_pf_total)
    tol = 1e-9
    for fn in tqdm(grids):
        tmp_res_dict = {}
        path = os.path.join(REF_PATH, fn)
        
        
        ####################
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lsgrid = init_from_matpower(path)
        cf = CaseFrames(path)

        init_method = get_init_method(path)
        if init_method == "flat":
            v_init = np.ones(lsgrid.total_bus(), dtype=complex)
        elif init_method == "dc":
            v_init_dc = np.ones(lsgrid.total_bus(), dtype=complex)
            v_init = lsgrid.dc_pf(v_init_dc, 10, 1e-6)
        elif init_method == "file":
            v_init = (cf.bus["VM"] * np.exp(1j * np.deg2rad(cf.bus["VA"]))).to_numpy()

        lsgrid.change_algorithm(PF_ALGO)
        res_1 = lsgrid.ac_pf(v_init, 10, 1e-6)
        assert res_1.shape[0] > 0, f"{fn}: base powerflow did not converge"
        ybus_ref = lsgrid.get_Ybus_solver()
        sbus_ref = lsgrid.get_Sbus_solver()
        ref_ref = lsgrid.get_slack_ids()
        pv_ref = lsgrid.get_pv()
        pq_ref = lsgrid.get_pq()
        ########################
        
        kcl_checker = KCLChecker(path)
        ybus = kcl_checker.base_ybus
        assert np.abs(ybus - ybus_ref).max() <= 1e-6
        
        sbus = kcl_checker.base_sbus
        assert np.abs(sbus - sbus_ref).max() <= 1e-6
        
        # ref, pv, pq = build_bus_idx(cf.bus, cf.gen, tol=tol)
        ref, pv, pq = kcl_checker.base_ref, kcl_checker.base_pv, kcl_checker.base_pq
        assert np.array_equal(ref, np.sort(ref_ref))
        assert np.array_equal(pv, np.sort(pv_ref))
        assert np.array_equal(pq, np.sort(pq_ref))

        load_p, load_q, *_ = lsgrid.get_loads_res_full()
        gen_p, *_ = lsgrid.get_gen_res_full()

        # should be >= tol: v_init does not satisfy KCL
        mismatch_init = kcl_checker.check_kcl_injection(
            load_p[None, :], load_q[None, :], gen_p[None, :], v_init[None, :])
        assert np.abs(mismatch_init).max() > tol

        # should all be < tol: the AC pf solution satisfies KCL
        mismatch_res = kcl_checker.check_kcl_injection(
            load_p[None, :], load_q[None, :], gen_p[None, :], res_1[None, :])
        assert np.abs(mismatch_res).max() <= 1e-6
        

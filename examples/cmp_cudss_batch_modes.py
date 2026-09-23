"""Compare the three cuDSS batch modes on one real (IIDM) snapshot.

N-1 on the first --n-ctg branches of the grid, solved in batches of --batch-size,
once per mode, each in its own subprocess (the mode is picked from environment
variables read when the batch driver is built):

  uniform      (default)                     CUDSS_CONFIG_UBATCH_SIZE
  block_diag   GPUSIM2GRID_USE_BLOCKDIAG=1   one block-diagonal matrix
  non_uniform  GPUSIM2GRID_USE_BATCH_MODE=1  cudssMatrixCreateBatchCsr

Requires a gpusim2grid build that contains these modes. Prints the raw per-phase
GPU timings (gpu_ms of the CUDA events; analysis / base case solve are
wall-clock) side by side, plus a correctness check against the uniform mode.

    python cmp_cudss_batch_modes.py --snapshot PATH.xiidm[.xz] \
        --n-ctg 300 --batch-size 300 [--nb-iter 4] [--modes uniform,block_diag] \
        [--out results.json]
"""
import argparse, json, os, subprocess, sys, tempfile, time, warnings
import numpy as np

MODES = {"uniform": {}, "block_diag": {"GPUSIM2GRID_USE_BLOCKDIAG": "1"},
         "non_uniform": {"GPUSIM2GRID_USE_BATCH_MODE": "1"}}
KCL_TOL = 1e-4


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True, help="IIDM snapshot loadable by pypowsybl")
    p.add_argument("--n-ctg", type=int, default=300, help="number of N-1 contingencies (first branches)")
    p.add_argument("--batch-size", type=int, default=None, help="batch size (default: --n-ctg, one batch)")
    p.add_argument("--nb-iter", type=int, default=4)
    p.add_argument("--reordering-alg", default="default")
    p.add_argument("--modes", default=",".join(MODES), help="comma-separated subset of " + ",".join(MODES))
    p.add_argument("--out", default=None, help="write all timings/results as JSON here")
    return p


def enable_max_voltage_change(model, max_dva=1.0, max_dvm=0.4):
    # NR step damping for the lightsim2grid base case (same as test_real_data.py)
    from lightsim2grid.lightsim2grid_cpp import ScalingPolicyType
    cfg = model.get_ac_algo_config()
    ip = list(cfg.int_params)
    ip[0] = int(ScalingPolicyType.MaxVoltageChange)
    cfg.int_params = ip
    rp = list(cfg.real_params)
    rp[0], rp[1] = max_dva, max_dvm
    cfg.real_params = rp
    model.set_ac_algo_config(cfg)


def child(args, out):
    import pypowsybl as pp
    import pypowsybl.loadflow as lf
    import torch
    from lightsim2grid.network.from_pypowsybl import init, bake_outer_loops
    from lightsim2grid.network.from_pypowsybl._olf_compare import iidm_bus_voltages
    import gpusim2grid
    from gpusim2grid import ContingencyAnalysisGPU

    net = pp.network.load(args.snapshot, {'iidm.die.with-extensions': "all"})
    lf.run_ac(net, parameters=None)
    bake_outer_loops(net)
    v_olf = iidm_bus_voltages(net)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init(net, gen_slack_id=None, sort_index=False, buses_for_sub=False,
                    keep_half_open_lines=True, fuse_zero_impedance_branches=True)
    enable_max_voltage_change(grid)
    v = v_olf["vm_pu"] * np.exp(1j * np.deg2rad(v_olf["va_deg"]))
    v[~np.isfinite(v)] = 1.
    v0 = np.ones(grid.get_bus_vn_kv().shape[0], dtype=complex)
    v0[grid._orig_to_ls] = v
    V = grid.ac_pf(v0, 300, 1e-6)
    assert V.shape[0] > 0, "lightsim2grid base case diverges"

    gpusim2grid.warmup()
    t0 = time.perf_counter()
    ca = ContingencyAnalysisGPU(grid, nb_iter=args.nb_iter, handle_disconnected_grid=True,
                                reordering_alg=args.reordering_alg,
                                compute_limit_violations=True)
    ca.add_contingencies_by_branch_id([[b] for b in range(args.n_ctg)])
    Vb = torch.from_dlpack(ca.compute(batch_size=args.batch_size)).cpu().numpy()
    wall = time.perf_counter() - t0
    np.savez(out, V=Vb, res=ca.last_residuals())
    t = ca.timings.to_dict()
    t["n_violations"] = int(sum(len(r) for r in ca.get_violations()))
    t["stopwatch_ms"] = 1e3 * wall
    t["n_bus_solver"] = int(grid.get_Ybus_solver().shape[0])
    json.dump(t, open(out + ".json", "w"), default=str)


def main():
    args = get_parser().parse_args()
    if args.batch_size is None:
        args.batch_size = args.n_ctg
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            sys.exit(f"unknown mode {m!r}; choose among {list(MODES)}")
    tmp = tempfile.mkdtemp(prefix="cudss_modes_")

    out = {}
    for m in modes:
        path = os.path.join(tmp, m)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GPUSIM2GRID_USE_")}
        env.update(MODES[m])
        cmd = [sys.executable, os.path.abspath(__file__), "--child", path] + sys.argv[1:]
        print(f"running {m} ...", flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode:
            last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"exit {r.returncode}"
            print(f"--- {m} FAILED: {last}")
            continue
        out[m] = (np.load(path + ".npz"), json.load(open(path + ".json")))
    if not out:
        sys.exit("every mode failed")

    first = next(iter(out.values()))[1]
    n_chunks = -(-args.n_ctg // args.batch_size)
    print(f"\nsnapshot {os.path.basename(args.snapshot)}  n_bus_solver={first['n_bus_solver']}  "
          f"{args.n_ctg} contingencies, batch {args.batch_size} ({n_chunks} batch(es)), "
          f"nb_iter {args.nb_iter}, reordering {args.reordering_alg}\n")

    # Raw per-phase GPU timings, one column per mode.
    names = list(out)
    keys = list(first["gpu_compute"].keys())
    w = max(len(k) for k in keys)
    print(f"{'GPU timings (ms)':{w}s} | " + " | ".join(f"{n:>12s}" for n in names))
    for k in keys:
        cells = []
        for n in names:
            v = out[n][1]["gpu_compute"].get(k)
            v = v["gpu_ms"] if isinstance(v, dict) else v
            cells.append(f"{v:12.2f}" if v is not None else f"{'-':>12s}")
        print(f"{k:{w}s} | " + " | ".join(cells))

    # Correctness summary.
    print()
    ref = out.get("uniform")
    summary = {}
    for n, (d, t) in out.items():
        res = d["res"]
        s = dict(nan=int(np.isnan(res).sum()),
                 above_tol=int((np.isfinite(res) & (res > KCL_TOL)).sum()),
                 median_residual=float(np.nanmedian(res)) if np.isfinite(res).any() else None,
                 n_violations=t["n_violations"], stopwatch_ms=t["stopwatch_ms"])
        line = (f"{n:12s} NaN {s['nan']}/{len(res)}  res>{KCL_TOL:g} {s['above_tol']}  "
                f"violations {s['n_violations']}  stopwatch {s['stopwatch_ms']:.1f} ms")
        if ref is not None and n != "uniform":
            a, b = ref[0]["V"], d["V"]
            both = np.isfinite(a) & np.isfinite(b)
            s["max_abs_dV_vs_uniform"] = float(np.abs(a[both] - b[both]).max()) if both.any() else None
            line += f"  max|dV| vs uniform {s['max_abs_dV_vs_uniform']:.1e}"
        print(line)
        summary[n] = s

    if args.out:
        json.dump({"args": vars(args), "summary": summary,
                   "timings": {n: t for n, (_, t) in out.items()}},
                  open(args.out, "w"), indent=1, default=str)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        path = sys.argv[2]
        a = get_parser().parse_args(sys.argv[3:])
        if a.batch_size is None:
            a.batch_size = a.n_ctg
        child(a, path)
    else:
        main()

"""Load sweep behind figures/saturation.png. Sweeps offered load (session
population) per policy, averages over seeds, writes results/load_sweep.json.
Parallel across cores; runs are independent. Run from anywhere:  python3 figures/sweep_load.py

Affinity saturates early, so it uses a lower load cap; the balance-family
policies use capacity better and need higher load to reveal their knee."""
import json, time, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim import Sim

TEMPORAL = dict(root_tokens=0, repo_trunk_tokens=0)
LOADS = {
    "aff":      [300, 600, 900, 1100, 1300, 1500],
    "cbio":     [400, 800, 1100, 1400, 1700],
    "bfio":     [400, 800, 1100, 1400, 1700],
    "prestage": [400, 800, 1100, 1400, 1700],
}
POLS  = ["aff", "bfio", "cbio", "prestage"]
SEEDS = [1]
T_END = 110.0
KEYS  = ("goodput", "idle", "ttft50", "ttft95", "blocked_s", "mem")


def one(args):
    pol, ns, seed = args
    r = Sim(policy=pol, n_sess=ns, seed=seed, T_end=T_END, **TEMPORAL).run()
    return pol, ns, {k: r[k] for k in KEYS}


if __name__ == "__main__":
    jobs = [(p, n, s) for p in POLS for n in LOADS[p] for s in SEEDS]
    t0, agg = time.time(), {}
    with ProcessPoolExecutor() as ex:
        for i, (pol, ns, m) in enumerate(ex.map(one, jobs), 1):
            agg.setdefault((pol, ns), []).append(m)
            print(f"  {i:3d}/{len(jobs)}  {pol:8s} n_sess={ns}", flush=True)

    out = {p: {k: [] for k in ("n_sess", *KEYS)} for p in POLS}
    for p in POLS:
        for n in LOADS[p]:
            ms = agg[(p, n)]
            out[p]["n_sess"].append(n)
            for k in KEYS:
                out[p][k].append(sum(d[k] for d in ms) / len(ms))
    dest = ROOT / "results" / "load_sweep.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}  ({time.time()-t0:.0f}s, {len(jobs)} runs)")

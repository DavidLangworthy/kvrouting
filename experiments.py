"""
Every table in the papers, reproducible. Run:  python3 experiments.py [name ...]
Full suite is ~20 min single-threaded. Each experiment is independent.
"""
import sys, json, time
from sim import Sim, fmt

TEMPORAL = dict(root_tokens=0, repo_trunk_tokens=0)                 # the-think-gap.md
SPATIAL  = dict(root_tokens=15_000, repo_trunk_tokens=20_000)       # price-of-a-cache-hit.md
OUT = {}


def E1_policies_temporal(T=150):
    """the-think-gap.md sec.4. Balance beats affinity by ~11%. Cache affinity is
    what manufactures the barrier idle."""
    print("\nE1  routing policies, temporal reuse (kappa == 1)")
    for p in ["rr", "jsq", "aff", "cache_lb", "bfio", "cbio"]:
        r = Sim(policy=p, seed=1, T_end=T, **TEMPORAL).run()
        print(fmt(p, r)); OUT[f"E1.{p}"] = r


def E2_cascade_temporal(T=150):
    """the-think-gap.md sec.2. Cascade is a NO-OP under temporal reuse: bit-identical."""
    print("\nE2  cascade on/off, temporal reuse  -> expect IDENTICAL")
    for c in [True, False]:
        r = Sim(policy="cbio", cascade=c, seed=1, T_end=T, **TEMPORAL).run()
        print(fmt(f"cascade={c}", r)); OUT[f"E2.cascade_{c}"] = r


def E3_cascade_spatial(T=150):
    """price-of-a-cache-hit.md sec.7.2. WITH cross-session sharing, cascade is a
    precondition: goodput ratio should equal kappa."""
    print("\nE3  cascade on/off, spatial (cross-session) sharing  -> expect ratio == kappa")
    for p in ["rr", "jsq"]:
        for c in [True, False]:
            r = Sim(policy=p, cascade=c, seed=1, T_end=T, **SPATIAL).run()
            print(fmt(f"{p} casc={c}", r)); OUT[f"E3.{p}.cascade_{c}"] = r


def E4_retention(T=120):
    """THE HEADLINE. pin / offload / discard. NOTE: retain='pin' can deadlock
    (HBM fills with idle KV and admission blocks) -- that IS the finding, but it
    makes the sim crawl. The closed form in analytics.retention() is the reliable
    version; this is the empirical check."""
    print("\nE4  retention mode (pin is slow by construction -- see docstring)")
    for ret in ["offload", "discard"]:
        r = Sim(policy="cbio", retain=ret, seed=1, T_end=T, **TEMPORAL).run()
        print(fmt(ret, r)); OUT[f"E4.{ret}"] = r
    print("     (for retain='pin' use analytics.retention() -- closed form, exact)")


def E5_theta(T=120):
    """the-think-gap.md sec.5. Price sweep. theta=0 -> pure affinity; theta=1 ->
    the physically-derived exchange rate; theta->inf -> pure balance."""
    print("\nE5  locality price sweep (CB-IO)")
    for th in [0.0, 0.1, 1.0, 10.0, 100.0]:
        r = Sim(policy="cbio", theta=th, seed=1, T_end=T, **TEMPORAL).run()
        print(fmt(f"theta={th}", r)); OUT[f"E5.theta_{th}"] = r


def E6_scale(T=110):
    """Does the routing GAIN scale with G? (It does not -- fractional waste is
    G-invariant. Only the exchange RATE scales, because cache cost is borne by one
    node regardless of pool size. We predicted otherwise and were wrong.)"""
    print("\nE6  pool size G (affinity vs priced)")
    for G, ns in [(8, 600), (16, 1200), (32, 2400)]:
        a = Sim(policy="cbio", theta=0.0, G=G, n_sess=ns, seed=1, T_end=T, **TEMPORAL).run()
        b = Sim(policy="cbio", theta=1.0, G=G, n_sess=ns, seed=1, T_end=T, **TEMPORAL).run()
        print(f"  G={G:>3}  affinity gp={a['goodput']:7.0f} idle={a['idle']:5.1%}  |  "
              f"priced gp={b['goodput']:7.0f} idle={b['idle']:5.1%}  |  "
              f"gain={100*(b['goodput']/a['goodput']-1):+5.1f}%")
        OUT[f"E6.G{G}.affinity"] = a; OUT[f"E6.G{G}.priced"] = b


def E7_w_fabric(T=120):
    """SENSITIVITY ON THE LOAD-BEARING CONSTANT. Everything in sec.5 of the paper
    depends on W_FABRIC=10. Sweep it. If the conclusion flips, say so."""
    print("\nE7  W_FABRIC sensitivity  (**the number to measure on real hardware**)")
    import hardware
    orig = hardware.W_FABRIC
    for w in [1.0, 3.0, 10.0, 30.0, 100.0]:
        hardware.W_FABRIC = w
        import importlib, sim as simmod
        importlib.reload(simmod)
        r = simmod.Sim(policy="cbio", seed=1, T_end=T, **TEMPORAL).run()
        print(simmod.fmt(f"W={w}", r)); OUT[f"E7.W_{w}"] = r
    hardware.W_FABRIC = orig


ALL = dict(E1=E1_policies_temporal, E2=E2_cascade_temporal, E3=E3_cascade_spatial,
           E4=E4_retention, E5=E5_theta, E6=E6_scale, E7=E7_w_fabric)

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    t = time.time()
    for n in names:
        ALL[n]()
    json.dump(OUT, open(f"results/{'_'.join(names)}.json", "w"), indent=1)
    print(f"\nwrote results/{'_'.join(names)}.json  ({time.time()-t:.0f}s)")

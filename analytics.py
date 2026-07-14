"""
Closed-form results. These do not need the simulator and are exact given hardware.py.
Three of the paper's five load-bearing numbers live here, not in the sim.
"""
from hardware import (ALPHA, M_TOK, T_SYNC, C_KV, B_PCIE, B_FABRIC,
                      prefill_s, fetch_pcie, fetch_ssd, fetch_fabric, barrier_cost, W_FABRIC)

FBAR, O = 107_000, 500      # steady-state mean footprint (tokens) and output length


def anchors():
    print("--- sanity anchors (if these are wrong, nothing downstream is worth reading) ---")
    A = M_TOK / FBAR
    step = ALPHA * M_TOK + T_SYNC
    print(f"  concurrent requests per node at full HBM : {A:6.1f}")
    print(f"  decode step time at full HBM             : {step*1e3:6.1f} ms")
    print(f"  per-stream token rate                    : {1/step:6.1f} tok/s")
    print(f"  turn service time (o={O})                 : {O*step:6.1f} s")
    print(f"  warm prefill of a 1.5k delta @ {FBAR//1000}k ctx  : {prefill_s(1500,FBAR):6.3f} node-s")
    print(f"     of which attention against the cache  : "
          f"{100*(prefill_s(1500,FBAR)-prefill_s(1500,0))/prefill_s(1500,FBAR):5.1f}%")


def exchange_rate():
    """T3. The cache cost is one-time and local. The barrier cost is recurring
    (every step) and global (all G-1 other ranks idle). So the ratio scales with G,
    and the threshold for 'take the miss' moves with pool size."""
    print(f"\n--- T3: exchange rate for a {FBAR//1000}k-token session, o={O}, W_FABRIC={W_FABRIC} ---")
    dram, fab, cold = fetch_pcie(FBAR), fetch_fabric(FBAR), prefill_s(FBAR)
    print(f"  cache costs (node-s):  local DRAM {dram:.3f} | migrate {fab:.3f} | cold recompute {cold:.3f}")
    print(f"\n{'G':>5} {'barrier':>9} {'/DRAM':>8} {'/migrate':>9} {'/recompute':>11}  verdict")
    for G in [4, 8, 16, 32, 64, 128]:
        b = barrier_cost(G, O, FBAR)
        v = "TAKE THE COLD MISS" if b > cold else "keep the hit"
        print(f"{G:>5} {b:>9.2f} {b/dram:>7.0f}x {b/fab:>8.0f}x {b/cold:>10.2f}x  {v}")


def retention(G=16):
    """T5 / the headline. A session's KV during the think gap: pin, offload, or discard.
    Pinning: idle KV is RESIDENT but not READ, so it does not slow the step -- it
    displaces active requests. Little's law closes the loop, and the fixed point is
    vicious: fewer active -> faster steps -> shorter service -> worse idle:active
    ratio -> fewer active still."""
    def solve(I, pin):
        A = 5.0
        for _ in range(500):
            step = ALPHA * A * FBAR + T_SYNC
            Ts   = O * step
            A_new = (M_TOK / FBAR) / (1 + I / Ts) if pin else M_TOK / FBAR
            A = 0.5 * A + 0.5 * A_new
        step = ALPHA * A * FBAR + T_SYNC; Ts = O * step
        return dict(A=A, step=step, Ts=Ts, tok_s=A / step, sessions=A * (1 + I / Ts))

    print(f"\n--- retention: what to do with a session's KV during the think gap ---")
    print(f"{'gap I':>7} {'':>4} {'active/node':>11} {'step':>8} {'tok/s/node':>11} "
          f"{'sessions/node':>14} {'penalty':>8}")
    for I in [1, 5, 15, 30, 60, 120]:
        p, f = solve(I, True), solve(I, False)
        print(f"{I:>5}s   pin {p['A']:>11.1f} {p['step']*1e3:>6.1f}ms {p['tok_s']:>11.0f} "
              f"{p['sessions']:>14.1f}")
        print(f"{'':>7}  off {f['A']:>11.1f} {f['step']*1e3:>6.1f}ms {f['tok_s']:>11.0f} "
              f"{f['sessions']:>14.1f} {f['sessions']/p['sessions']:>7.2f}x")
    rt = 2 * fetch_pcie(FBAR)
    print(f"\n  break-even think time I* = PCIe round trip = {rt*1e3:.0f} ms.")
    print(f"  Real agentic gaps are 1-300 s.  NEVER PIN.")
    print(f"  discard+recompute = {prefill_s(FBAR):.2f} node-s vs {fetch_pcie(FBAR):.3f} reload "
          f"= {prefill_s(FBAR)/fetch_pcie(FBAR):.0f}x worse.")


def offload_economics(G=16, goodput_on=23666, goodput_off=24156, pfu_on=0.51, pfu_off=26.53):
    """Cluster tokens/s per node, counting prefill nodes. Numbers from experiments.E6."""
    print(f"\n--- offload cluster economics (prefill pool = G/2 = {G//2} nodes) ---")
    for tag, gp, pfu in [("offload ON ", goodput_on, pfu_on), ("offload OFF", goodput_off, pfu_off)]:
        pf = pfu * (G // 2)
        print(f"  {tag}: {G} decode + {pf:6.1f} prefill = {G+pf:6.1f} nodes "
              f"-> {gp/(G+pf):7.0f} tok/s/node")


def prestage_economics(gap=15.0, turns_s_node=3.0):
    """T6 / open problem #2. Why think-gap pre-staging is possible at all.

    The affinity/balance tension is assumed to be decided at ADMISSION. But the KV of
    an idle session sits untouched for the whole think gap with nobody waiting on it.
    Moving it is fabric-bound and one-time; the question is whether it fits in the gap
    and inside the fabric budget the barrier can spare."""
    raw = FBAR * C_KV / B_FABRIC                  # raw migration time, no W_FABRIC (off critical path)
    charged = fetch_fabric(FBAR)                  # what the sim charges (W_FABRIC, collective contention)
    gb = FBAR * C_KV / 1e9
    print(f"\n--- T6: think-gap pre-staging feasibility ({FBAR//1000}k-token session) ---")
    print(f"  KV to move                : {gb:5.1f} GB")
    print(f"  raw fabric time           : {raw*1e3:5.0f} ms   (idle-fabric, off the barrier's path)")
    print(f"  fits in a {gap:.0f}s think gap : {gap/raw:5.0f}x over")
    print(f"  charged cost (W={W_FABRIC:.0f})       : {charged:5.3f} node-s   (if it collides with the barrier)")
    print(f"\n  fabric budget: at ~{turns_s_node:.0f} turns/s/node, one migration = {gb:.0f} GB")
    budget = B_FABRIC / (FBAR * C_KV)             # migrations/s/node at full fabric
    print(f"    full fabric would allow  {budget:5.1f} migrations/s/node")
    print(f"    but it shares the barrier's fabric -> spend only the slack: ~1-2/s/node")
    print(f"  the promise: admission sees a LOCAL reload ({fetch_pcie(FBAR)*1e3:.0f} ms, affinity's TTFT)")
    print(f"    on a balanced node (balance's throughput). Decision moved OUT of admission.")


if __name__ == "__main__":
    anchors(); exchange_rate(); retention(); offload_economics(); prestage_economics()

"""
Discrete-step simulator: barrier-synchronised DP decode pool + PD-disaggregated
prefill pool + prefix-tree KV reuse + node-local KV offload hierarchy.

The whole model rests on separating two quantities that prior work conflates:

    resident(g)  -- tokens of KV occupying node g's HBM.       Bounded by M.
    read(g)      -- tokens of KV node g must READ this step.   Sets the barrier.

They differ in two ways, and both differences are load-bearing:
  * Idle (between-turn) KV is RESIDENT but not READ. -> the retention question.
  * A cascade kernel makes concurrent requests sharing a prefix READ it once,
    even though they'd otherwise each read it in full. -> the cascade question.

Without that split you cannot state either result.
"""
import math, random, heapq
from dataclasses import dataclass
from hardware import (ALPHA, M_TOK, T_SYNC, C_KV, DRAM_TOK,
                      prefill_s, fetch_pcie, fetch_ssd, fetch_fabric, barrier_cost)

POLICIES = ["rr", "jsq", "aff", "cache_lb", "bfio", "cbio"]
RETAIN   = ["offload", "pin", "discard"]


def _zipf(n, a):
    w = [1 / (i + 1) ** a for i in range(n)]
    s = sum(w)
    return [x / s for x in w]


def _ln(rng, median, sigma):
    return median * math.exp(rng.gauss(0, sigma))


@dataclass
class Req:
    sid: int; repo: int; ctx: int; out: int
    g: int = -1; d: int = 0; arr: float = 0.0


@dataclass
class Sess:
    sid: int; repo: int; ctx: int
    turns: int = 0
    home: int = -1                 # node whose HBM/DRAM/SSD holds this session's KV
    tier: str = "none"             # hbm | dram | ssd | none


class Sim:
    """
    sharing:
      root_tokens / repo_trunk_tokens = 0  -> TEMPORAL reuse only (turns of one
          conversation). kappa == 1 by construction. This is the regime of
          `the-think-gap.md` and the one that matches agentic coding.
      root_tokens > 0                      -> adds SPATIAL reuse (a system prompt
          and/or repo context shared across CONCURRENT conversations). kappa > 1.
          This is the regime of `price-of-a-cache-hit.md`.
    """

    def __init__(self, G=16, policy="cbio", theta=1.0,
                 cascade=True, retain="offload",
                 root_tokens=0, n_repos=40, repo_trunk_tokens=0, zipf_a=1.1,
                 n_sess=1200, ctx_median=70_000, ctx_sigma=0.55, ctx_max=260_000,
                 out_median=500, out_sigma=0.8, delta_median=1500,
                 think_median=15.0, think_sigma=0.9,
                 T_end=200.0, T_warm=80.0, seed=0):
        assert policy in POLICIES and retain in RETAIN
        self.__dict__.update(locals()); del self.self
        self.rng = random.Random(seed)

        self.root = root_tokens
        self.repo_w  = _zipf(n_repos, zipf_a)
        self.repo_tr = [repo_trunk_tokens for _ in range(n_repos)]

        self.S        = [[] for _ in range(G)]   # active requests per node
        self.hold_tok = [0] * G                  # pinned-IDLE KV: resident, not read
        self.hold_of  = {}                       # sid -> (node, tokens)
        self.dram     = [0] * G
        self.P        = max(2, G // 2)           # prefill nodes
        self.clock, self.rr_ptr = 0.0, 0
        self.wait, self.pend = [], []
        self.sess = {}
        self._reset()

        for sid in range(n_sess):
            repo = self.rng.choices(range(n_repos), self.repo_w)[0]
            ctx  = min(ctx_max, self.root + self.repo_tr[repo] +
                       int(_ln(self.rng, ctx_median - self.root - self.repo_tr[repo], ctx_sigma)))
            self.sess[sid] = Sess(sid, repo, ctx)
            heapq.heappush(self.pend, (self.rng.uniform(0, 40.0), sid))

    def _reset(self):
        self.t0 = None
        self.tok = self.done = self.ncold = self.blk = 0
        self.busy = self.idle = self.pf = self.recomp = 0.0
        self.pcie_b = self.fab_b = 0.0
        self.ttft, self.kap, self.util, self.conc = [], [], [], []

    # ---------------------------------------------------------------- the two loads
    def _shared(self, g, extra=None):
        """Tokens of KV physically stored for the ACTIVE requests on g: the UNION of
        their root-to-leaf prefix paths. A weighted coverage function - submodular,
        not a sum. Reduces to a plain sum when root == trunk == 0."""
        rs = self.S[g] + ([extra] if extra else [])
        if not rs:
            return 0
        repos = {r.repo for r in rs}
        return (self.root
                + sum(self.repo_tr[r] for r in repos)
                + sum(max(0, r.ctx + r.d - self.root - self.repo_tr[r.repo]) for r in rs))

    def resident(self, g, extra=None):
        """HBM occupancy. Includes pinned-idle KV. Bounded by M_TOK."""
        return self._shared(g, extra) + self.hold_tok[g]

    def read(self, g, extra=None):
        """KV tokens read per decode step. Idle KV is NOT read. With a cascade kernel
        a shared prefix is read once for the whole batch; without one, every request
        reads its own context in full."""
        if self.cascade:
            return self._shared(g, extra)
        rs = self.S[g] + ([extra] if extra else [])
        return sum(r.ctx + r.d for r in rs)

    # ---------------------------------------------------------------- cache cost
    def cache_cost(self, se, g):
        """(node-seconds, pcie bytes, fabric bytes) to materialise se's prefix on g."""
        n = se.ctx
        if se.home == g and se.tier == "hbm":  return 0.0, 0.0, 0.0
        if se.home == g and se.tier == "dram": return fetch_pcie(n), n * C_KV, 0.0
        if se.home == g and se.tier == "ssd":  return fetch_ssd(n),  n * C_KV, 0.0
        if se.home >= 0 and se.tier != "none":
            return fetch_fabric(n), 0.0, n * C_KV          # cross-node migration
        return prefill_s(n, 0), 0.0, 0.0                    # cold recompute

    # ---------------------------------------------------------------- router
    def pick(self, se, req, R, Rmax):
        best, bg = None, -1
        mean = sum(R) / self.G
        for g in range(self.G):
            if self.resident(g, req) > M_TOK:
                continue
            cc, _, _ = self.cache_cost(se, g)
            Rg  = self.read(g, req)
            bar = barrier_cost(self.G, req.out, Rg - Rmax)

            if   self.policy == "rr":   sc = (g - self.rr_ptr) % self.G
            elif self.policy == "jsq":  sc = R[g]
            elif self.policy == "aff":  sc = (0 if se.home == g else 1) * 1e9 + R[g]
            elif self.policy == "cache_lb":                        # production-style
                hit = se.home == g
                sc  = (0 if (hit and R[g] <= 1.5 * (mean + 1)) else 1e9) + R[g]
            elif self.policy == "bfio": sc = bar + R[g] * 1e-12    # decode balance only
            else:                       sc = cc + self.theta * bar # CB-IO
            if best is None or sc < best:
                best, bg = sc, g
        self.rr_ptr += 1
        return bg

    # ---------------------------------------------------------------- main loop
    def run(self):
        warmed = False
        while self.clock < self.T_end:
            if not warmed and self.clock >= self.T_warm:
                self._reset(); self.t0 = self.clock; warmed = True

            R    = [self.read(g) for g in range(self.G)]
            Rmax = max(R) if R else 0
            T    = ALPHA * Rmax + T_SYNC
            self.clock += T
            self.busy += self.G * T
            self.idle += sum(ALPHA * (Rmax - r) for r in R)

            res = [self.resident(g) for g in range(self.G)]
            nom = sum(r.ctx + r.d for g in range(self.G) for r in self.S[g])
            sh  = sum(self._shared(g) for g in range(self.G))
            if sh > 0:
                self.kap.append(nom / sh)
            self.util.append(sum(res) / (self.G * M_TOK))
            self.conc.append(sum(len(x) for x in self.S) / self.G)

            # --- generate one token per active request; retire the finished
            for g in range(self.G):
                keep = []
                for r in self.S[g]:
                    r.d += 1; self.tok += 1
                    if r.d < r.out:
                        keep.append(r); continue
                    self.done += 1
                    se = self.sess[r.sid]
                    se.ctx = min(self.ctx_max,
                                 r.ctx + r.out + int(_ln(self.rng, self.delta_median, 0.7)))
                    se.turns += 1; se.home = g
                    if se.ctx >= self.ctx_max or se.turns > 80:      # recycle: fresh session
                        se.ctx = self.root + self.repo_tr[se.repo] + int(
                            _ln(self.rng, self.ctx_median - self.root - self.repo_tr[se.repo],
                                self.ctx_sigma))
                        se.turns = 0; se.home = -1; se.tier = "none"
                        heapq.heappush(self.pend, (self.clock + _ln(self.rng, self.think_median,
                                                                    self.think_sigma), se.sid))
                        continue
                    think = _ln(self.rng, self.think_median, self.think_sigma)
                    # ---- THE RETENTION DECISION ----
                    if self.retain == "pin":
                        se.tier = "hbm"
                        self.hold_tok[g] += se.ctx
                        self.hold_of[se.sid] = (g, se.ctx)
                    elif self.retain == "offload":
                        if self.dram[g] + se.ctx <= DRAM_TOK and think < 120:
                            se.tier = "dram"; self.dram[g] += se.ctx
                        else:
                            se.tier = "ssd"
                        self.pcie_b += se.ctx * C_KV                 # writeback
                    else:
                        se.tier = "none"                             # discard
                    heapq.heappush(self.pend, (self.clock + think, se.sid))
                self.S[g] = keep

            while self.pend and self.pend[0][0] <= self.clock:
                _, sid = heapq.heappop(self.pend)
                se = self.sess[sid]
                out = max(80, int(_ln(self.rng, self.out_median, self.out_sigma)))
                self.wait.append(Req(sid, se.repo, se.ctx, out, arr=self.clock))

            # --- admission
            R    = [self.read(g) for g in range(self.G)]
            Rmax = max(R) if R else 0
            still = []
            for req in self.wait:
                se = self.sess[req.sid]
                g  = self.pick(se, req, R, Rmax)
                if g < 0:
                    still.append(req); self.blk += 1; continue
                cc, pb, fb = self.cache_cost(se, g)
                cold = (se.home < 0 or se.tier == "none")
                if cold:
                    self.recomp += cc; self.ncold += 1
                    self.pf += cc
                if se.sid in self.hold_of:                            # release pinned copy
                    hg, ht = self.hold_of.pop(se.sid); self.hold_tok[hg] -= ht
                if se.tier == "dram" and se.home == g:
                    self.dram[g] = max(0, self.dram[g] - se.ctx)
                self.pf += prefill_s(self.delta_median, req.ctx)      # the delta always recomputes
                self.pcie_b += pb; self.fab_b += fb
                req.g = g
                self.ttft.append(self.clock - req.arr + cc)
                self.S[g].append(req)
                se.home, se.tier = g, "hbm"
                R[g] = self.read(g); Rmax = max(R)
            self.wait = still

        el = self.clock - (self.t0 or 0)
        q  = lambda a, p: sorted(a)[min(len(a) - 1, int(p * len(a)))] if a else 0.0
        m  = lambda a: sum(a) / len(a) if a else 0.0
        return dict(
            goodput=self.tok / el, turns_s=self.done / el,
            idle=self.idle / self.busy if self.busy else 0.0,
            kappa=m(self.kap), mem=m(self.util), active=m(self.conc),
            ttft50=q(self.ttft, .5), ttft95=q(self.ttft, .95),
            prefill_util=self.pf / (self.P * el),
            recompute=self.recomp / el, cold_s=self.ncold / el, blocked_s=self.blk / el,
            pcie_gbs=self.pcie_b / el / 1e9 / self.G,
            fabric_gbs=self.fab_b / el / 1e9 / self.G,
            elapsed=el, n=self.done,
        )


def fmt(tag, r):
    return (f"{tag:>12} gp={r['goodput']:7.0f} idle={r['idle']:6.1%} kap={r['kappa']:5.3f} "
            f"mem={r['mem']:5.1%} act={r['active']:4.1f} pf={r['prefill_util']:6.2f} "
            f"ttft95={r['ttft95']:6.2f} pcie={r['pcie_gbs']:5.1f} fab={r['fabric_gbs']:5.1f}")


if __name__ == "__main__":
    print("temporal reuse only (kappa == 1), CB-IO:")
    print(fmt("cbio", Sim(policy="cbio", seed=1, T_end=150).run()))

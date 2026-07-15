# The price of a cache hit

Simulator and papers on **routing + KV placement in disaggregated LLM serving**, for
iterated workloads with within-conversation prefix reuse (agentic coding).

Start with **`CLAUDE.md`** — it carries the research context, what's established, what
was refuted, the known bugs, and the ranked open problems.

## Quick start

```bash
python3 hardware.py        # constants + derivations + the cost of a cache hit
python3 analytics.py       # closed forms: exchange rate, retention, cluster economics
python3 sim.py             # one simulator run
python3 experiments.py E1  # one experiment (~60s).  Bare = full suite (~20 min)
```

## Live dashboard

`docs/index.html` (**kvcalc**) is a self-contained dashboard for the cost model — the
closed forms recompute instantly as you drag any of ~21 hardware/workload constants.
No backend, no build step, no dependencies. Open the file directly, or serve `docs/`
as a static site (GitHub Pages: Settings → Pages → `main` branch, `/docs`). Five
panels: the exchange rate, retention, prefill-pool economics, affinity-vs-balance
routing, and — **panel 5** — think-gap pre-staging, whose feasibility is live and whose
throughput/latency win is measured against the simulator (E8; see `figures/saturation.png`).
The one output that is not closed form (a time-resolved discrete-event run) is the
simulator itself.

## Files

| file | what |
|---|---|
| `hardware.py` | every physical constant, with its derivation. **`W_FABRIC` is the load-bearing one.** |
| `sim.py` | discrete-step barrier simulator. Separates `resident()` from `read()` — do not re-merge. |
| `analytics.py` | closed forms. Three of the five headline numbers live here, not in the sim. |
| `experiments.py` | E1–E8, reproducing every table in the papers (E8 = think-gap pre-staging). |
| `papers/the-think-gap.md` | **the current paper** (temporal / within-conversation reuse) |
| `papers/price-of-a-cache-hit.md` | archived (spatial / cross-conversation reuse). Right reasoning, wrong premise. |
| `papers/sources/` | the two source arXiv papers (Nie et al. 2605.04595, Chen et al. 2601.17855) |
| `figures/` | `saturation.png` — the unified TTFT-vs-throughput story (see `the-think-gap.md` Fig. 1); regenerate with `sweep_load.py` then `plot_saturation.py` |
| `docs/index.html` | **kvcalc** — the live, self-contained cost-model dashboard (see above) |
| `results/` | JSON from runs |

## The one-paragraph version

Turns of one conversation are never concurrent, so they never co-reside, so κ ≡ 1 — and
within-conversation prefix reuse is worth **nothing** to the decode pool. A cascade kernel
is a no-op (verified bit-identical); capacity `μ = M/(b̄·E[g])` is untouched; the turn still
pins 107k tokens of KV for 500 steps to emit 500 tokens. The cache's *entire* value is
prefill elimination — 47× — which makes it a **storage** problem: never pin a session's KV
through the think gap (7.7× concurrency collapse), never discard it (81× a reload), offload
it node-locally. And because κ=1 means the memory cap no longer incidentally balances the
pool, the barrier bites: **cache affinity costs 10.9% goodput and quadruples barrier idle.**
So take the miss. Migrating a session's KV (0.44 node-s) is 11× cheaper than the straggler
it prevents (4.98), and above ~16 DP ranks even a full cold recompute (4.76) is cheaper.
Then stop making the decision at admission at all: the gap is 15 seconds and the move is
43 ms, so pre-stage during the gap and get affinity's TTFT *and* balance's throughput.

## Caveats you must read

Nothing here has touched a GPU. `W_FABRIC = 10` is a judgment call and every "migrate rather
than stall" conclusion depends on it — at W=100 the conclusion *inverts*. See `CLAUDE.md`.

Pre-staging's win is contingent on a usable **return-time predictor**: at prediction noise
σ≤0.3 it delivers affinity's TTFT *and* balance's throughput (E8); at σ=1.0 (no predictive
skill) it collapses back to affinity's numbers. Predicting arrival is far easier than
predicting output length, but this is the second number to earn on real traces.

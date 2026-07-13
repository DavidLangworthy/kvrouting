# CLAUDE.md — research context

## What this is

A research program on **routing and KV-cache placement in disaggregated LLM serving**, sitting in the gap between two recent arXiv papers. There is a simulator, two paper drafts, and a set of results — some verified, one refuted, several open.

The workload of record is **agentic coding**: long-lived conversations where turn *t+1*'s prompt contains turn *t*'s entire context as a literal prefix (including the model's own generated tokens), plus a small delta (a tool result, a test log). Contexts 40k–260k, outputs ~500 tokens, think gaps ~15 s.

## The two source papers

- **Nie, Si & Zhou**, *A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints*, arXiv **2605.04595**. Collapses a request's memory trajectory to a scalar area `g(s,o) ≈ o·(s + o/2)`. Gives `μ = M / (b̄ · E[g(s,o)])`, attained by *any* work-conserving policy. Memory as **capacity**. Validated on real A100s within ~10%. Their multi-GPU experiment is 8 *independent replicas* — no barrier.

- **Chen, Bu, Song, Lu, Ye & Zhou**, *A Universal Load Balancing Principle and Its Application to Large Language Model Serving*, arXiv **2601.17855**. Data-parallel decode behind a per-step collective barrier: `T_step = max_g T_local^(g) + T_sync`, `T_local` linear in resident KV, assignment sticky. Production trace: barrier idle >40%. Method BF-IO minimises predicted imbalance over a short lookahead. Memory as **sticky state**. Simulation only.

**They are the same object under two functionals.** Nie's `g` is the time-integral of Chen's per-step profile `W_i = (s, s+1, …, s+o)`. Integrate → capacity ceiling. Take the max across workers → barrier idle. Neither models reuse.

Two things worth checking in Chen et al. if you go back to them:
- Their theorems appear to be proved for the **H=0 specialisation** — meaning the headline Ω(√(B log G)) guarantee may not need the lookahead they bill as the central insight. Verify against the theorem statements.
- The **energy claim moved between versions**: v1 body reports 3.4%; the v2 abstract advertises 28% measured and >52% at fleet scale, two days later. Reconcile before citing.

## The central modelling move

Separate two quantities that both prior papers conflate:

```
resident(g)   tokens of KV occupying node g's HBM.        Bounded by M.
read(g)       tokens of KV node g must READ this step.    Sets the barrier.
```

They diverge in exactly two ways, and each one is a result:
- **Idle (between-turn) KV is resident but not read.** → the retention question (pin/offload/discard).
- **A cascade kernel makes concurrent requests sharing a prefix read it once.** → the cascade question.

Without that split neither result is statable. `sim.py` implements both as separate methods; do not re-merge them.

## Two kinds of reuse, and they behave oppositely

| | **spatial** (concurrent conversations share a system prompt / repo) | **temporal** (turns of ONE conversation) |
|---|---|---|
| sharing factor κ | > 1 (measured 1.23) | **≡ 1 exactly** — turns are never concurrent |
| worker load | submodular coverage function | plain sum |
| capacity μ | routing-dependent | **routing-independent — Nie et al. stands** |
| cascade kernel | precondition, worth +20% | **no-op (verified bit-identical)** |
| memory | binds at ~95%, incidentally pre-balances the pool | 65–69%, loads diverge freely |
| does routing matter? | barely (all policies within 1.4%) | **a lot** (affinity costs 10.9%) |
| the cache's job | raise decode capacity | **feed the prefill pool, nothing else** |
| paper | `papers/price-of-a-cache-hit.md` | `papers/the-think-gap.md` ← **the real one** |

**The user's workload is temporal.** `the-think-gap.md` is the current paper. The spatial draft is archived because its reasoning is correct but its premise is not the workload of interest. Real systems have both (a shared system prompt *and* within-session growth) and the effects compose — but spatial is worth κ≈1.18 while temporal is worth 47× on prefill. Not close.

## Established (simulator + closed form agree)

1. **Temporal reuse gives the decode pool nothing.** κ = 1.000; cascade on/off is bit-identical; `g(s,o)` is ~99.5% *s* and caching does not shrink *s*. The turn still pins 107k tokens for 500 steps to emit 500. **Confidence: high** — this is near-definitional once you see it.
2. **The cache's entire value is prefill: 47×** (4.76 → 0.10 node-s). Two-thirds of even a *warm* prefill is the delta's attention against the cached prefix. Prefix caching makes prefill `O(Δ·s)`, not free. **Confidence: high.**
3. **Never pin, never discard; offload node-locally.** Pinning through a 15 s gap collapses active concurrency 7.7× and costs ~38% goodput; penalty is `(1 + I/T_s)` and reaches 12× at chat-length gaps. Discard is 81× a reload and inflates the prefill pool 26×. Break-even think time is the PCIe round trip, **117 ms**. **Confidence: high** (closed form in `analytics.retention()`; the sim's `retain="pin"` deadlocks, which *is* the finding but makes it slow — use the closed form).
4. **Balance beats affinity by 10.9%** (24,874 vs 22,426 goodput; 4.1% vs 14.8% barrier idle). With κ=1 the memory cap no longer pre-balances the pool, so cache affinity *manufactures* the stragglers. **Confidence: medium-high** — robust across seeds and G, but synthetic workload.
5. **The exchange rate.** Cache cost is one-time and local; barrier cost is `(G−1)·ô·α·ΔL` — recurring and global. Migrate (0.44 node-s) ≪ recompute (4.76) ≈ barrier at G=16 (4.98) < barrier at G=32 (10.29). **Migrate; when the fabric saturates, recompute; never eat the barrier. Confidence: medium — see W_FABRIC below.**

## Refuted / wrong (kept deliberately)

- **Square-root replication of hot prefixes** (`r* = √(ν·f̄/ℓ)`). Derived, built, tested — it *loses*. Pinned replicas spend HBM (the resource that sets μ) to buy balance the memory cap already provides. Under temporal reuse it is also *inapplicable* (fan-out 1). **Do not rebuild this.**
- **"The routing gain grows with G."** It does not — the fractional waste is G-invariant (E6: +5.7% at G=8, +1.3% at G=16, +1.4% at G=32). Only the *exchange rate* scales with G, because cache cost is borne by one node regardless of pool size. The threshold moves; the fraction does not.

## Known issues — read before trusting a number

- **`W_FABRIC = 10` is the load-bearing constant and it is a judgment call, not a measurement.** Cross-node KV traffic shares the fabric with the EP all-to-all, which is on the barrier's critical path every step. At W=1 migration is ~110× cheaper than the barrier (conclusion strengthens); at W=100 it is comparable (**the conclusion inverts and affinity routing returns**). `experiments.E7` sweeps it. **This is the first thing to measure on real hardware.**
- **The archived spatial runs (`price-of-a-cache-hit.md`) have a bug the clean sim fixes**: pinned trunk replicas were charged *read* bandwidth even with no active requests of that repo. This biases E4-replication *against* replication — the direction of the published conclusion. The conclusion may still hold (it also holds on pure memory-cost grounds) but **re-run it before citing.**
- **Paper says "contexts log-normal (median 107k)". The sim seeds at median 70k** and lets contexts grow toward the 260k cap; 107k is the steady-state *mean* footprint used in the closed forms. Fix the prose, not the code.
- **`ALPHA` shifted 6.06 → 6.21 ns** in the clean rewrite (exact `C_KV = 163,840` vs a rounded `160e3`). All published numbers move ~2%. The clean value is correct.
- **`T_local` is modeled as exactly linear in resident KV**, following Chen et al. Real kernels have fixed overheads; the linear model flatters the barrier story.
- Nothing here has touched a GPU. Nie et al. validated on real A100s; Chen et al. simulate; we simulate.

## Open problems, ranked

1. **Measure `W_FABRIC`.** Run an EP all-to-all at production batch size, then run it while streaming a 17 GB KV blob over the same fabric, and measure the inflation of `T_sync`. Everything in §5 of the paper turns on this one number.
2. **Think-gap rebalancing — designed, not tested.** The idea: the gap is 15 s, the migration is 43 ms of raw fabric time, and the KV is idle with nobody waiting on it. **Pre-stage** the session's KV on a lightly-loaded worker *during the gap*. You get affinity's TTFT (0.12 s) *and* balance's throughput. This dissolves the affinity-vs-balance tension the whole cache-aware-routing literature is organised around — it exists only because everyone assumed the decision happens at admission. **Needs: a return-time predictor** (far easier than the output-length prediction Chen et al. rightly avoid — you're predicting *arrival*, not *duration*). Implement as a new policy in `sim.py:pick` plus a background migration queue. **This is the highest-value next step.**
3. **The fabric budget is a real constraint and is not yet modeled as one.** At ~3 turns/s/node and 17 GB/migration, you can afford ~1–2 migrations/s/node. So the router is *spending a scarce fabric budget on the worst stragglers*, and the shadow price on that budget is the true "price of locality". Currently the sim charges fabric but does not *cap* it. Add the cap; the three-tier policy (migrate → recompute → never stall) should fall out.
4. **CB-IO's barrier term is a hinge** — it penalises *creating* a new maximum but does not reward *filling valleys*. That's why E5's θ sweep keeps improving past the physically-derived θ=1. A term that also rewards leveling should close the gap and make θ=1 optimal, which is the claim we want.
5. **The min-max submodular allocation problem is stated, not solved.** Under spatial sharing, BF-IO's linear assignment becomes min-max *submodular* allocation. Greedy marginal assignment is what we implement; no approximation guarantee. That's the obvious theory paper.
6. **KV quantisation is unmodeled.** int4 for the offload tiers changes `C_KV` by 4× and moves every bandwidth number — probably in offload's favour.

## Conventions

- Everything is in **node-seconds** (one node = one TP=8 group = one logical worker). Never mix GPU-seconds and node-seconds.
- All hardware constants live in `hardware.py` with their derivations. Change them there and every result moves coherently.
- A run at `T_end=150` takes ~10 s; the full `experiments.py` suite is ~20 min single-threaded.
- `retain="pin"` deadlocks by design (HBM fills with idle KV, admission blocks). Use `analytics.retention()`.

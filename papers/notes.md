# Notes

Working notes on the two source papers and the systems background behind them.
Companion to `sources/` and to the two drafts. Terse on purpose.

---

## The move worth keeping from Nie et al.

> The genuinely useful move is collapsing a request's whole memory trajectory to a
> single scalar **area**, `g(s,o) ≈ o·(s + o/2)` — the time-integral of its occupancy.

A decode request doesn't hold a fixed-size server; it holds a KV footprint that grows
by one token every step, from `s` (prompt) up to `s+o` (prompt + output). Plot occupancy
against time and it's a trapezoid; its **area** is `g(s,o) ≈ o·(s + o/2)` token-steps.
That single scalar is the request's true load on a memory-budgeted worker. Everything in
Nie et al. follows: because memory is additive and the budget `M` is a scalar, the
capacity ceiling is `μ = M / (b̄ · E[g])`, and no scheduler beats it.

The reason it's the *useful* move: it turns a time-varying, per-step resource trajectory
into one number you can put in a queueing formula. It's also the hinge for our own split —
the **time-integral** of the per-step profile is Nie's capacity; the **per-step max across
workers** of that same profile is Chen's barrier idle. Same trajectory, two functionals.

**Units: byte·seconds — a space-time area, not a rate.** This is the unusual and telling
part. `g` is memory × time: tokens·steps, or physically **byte·seconds** (× bytes/token ×
seconds/step). It is *not* a bandwidth (bytes/s) and *not* a size (bytes) — it's an **area
in the memory–time plane**. That's exactly why it's the right load term: a request's cost
to a memory-budgeted server is neither how big it is nor how fast it runs, but **how much
memory it holds for how long**. And it makes the capacity formula dimensionally clean —
`capacity = budget / (batch-factor · mean area)` is `bytes / (byte·seconds) = 1/seconds`, a
rate. If you ever see the key metric quoted in bytes/second, someone has collapsed the time
axis by mistake.

---

## The framework: one object, two functionals, one missing split

Both papers are two readings of the *same* object. State it once, in plain names (symbols
in brackets so the math still lines up), and the three results become corollaries.

**The object — a request's footprint over time.** A decode request holds a KV **footprint**
that grows one token per step (ignoring spec decode — see the BF-IO note): at step `t` it
occupies

    footprint(t) = prefix-length + t          [ W_i(t) = s + t,  t = 0 … output-length ]

Plot footprint against step: a trapezoid. Everything below is a way of contracting this one
picture.

**Functional 1 — sum over time → capacity (Nie).** Add a request's footprint over its whole
life = the **area** of that trapezoid:

    area = Σ_t footprint(t) ≈ output-length · (prefix-length + output-length/2)   [ g(s,o) ≈ o·(s + o/2) ]

(Units: **byte·seconds**, the space-time area above.) A node with an HBM **budget** sustains
a request **rate** of

    capacity = budget / (batch-factor · mean area)         [ μ = M / (b̄ · E[g]) ]

Any work-conserving policy reaches it; routing cannot beat it. **Memory as capacity.**

**Functional 2 — max over nodes → barrier (Chen).** Each step, a node carries the summed
footprints of the requests resident on it — its **node-load**. Data-parallel ranks
synchronize every step, so step time is set by the **busiest** node:

    step-time = read-cost · max_over_nodes(node-load) + sync    [ α · max_g L_g + T_sync ]

The gap between the busiest node and the average is pure **idle** (>40% in their trace).
**Memory as instantaneous load / the straggler.**

> Same object (footprint over time). **Integrate it over a request's life → capacity.**
> **Take its max across nodes each step → barrier idle.** One is the time-integral, the
> other the worker-max. That is the entire relationship between the two papers.

**The split both conflate.** Each paper uses a single "load = footprint." Separate it into
two quantities that the base model happens to make equal:

    resident = tokens occupying a node's HBM        (capped by the budget → Nie's constraint)
    read     = tokens a node must READ this step     (sets the barrier   → Chen's cost)

They come apart on the one axis that matters for this workload:
- **Idle KV** — between turns, a session's KV is **resident but not read**. That single gap
  is where retention, the exchange rate, and pre-staging all live.

(They *could* also diverge if concurrent requests shared a prefix — read once though resident
once — but turns of one conversation are never concurrent, so there is nothing to share. That
regime is a separate, archived line of work, `price-of-a-cache-hit.md`, and touches nothing
below.)

**The three results, as corollaries** (full closed forms in `analytics.py`):

| result | drops out of | where |
|---|---|---|
| **never pin; offload node-local** | the idle-KV gap — pinning inflates `resident` without touching `read`; a Little's-law fixed point. Break-even = one PCIe round trip | `analytics.retention()` |
| **the exchange rate** | `resident`-cost (one-time, local) vs `read`-cost through the barrier (recurring, global, ×(nodes−1)). Migrate < recompute ≈ barrier | `analytics.exchange_rate()` |
| **think-gap pre-staging** | move `resident` to a balanced node *while `read = 0`* for that session (the gap), so at admission `read` lands balanced for free | E8 / `analytics.prestage_economics()` |

That's the framework: one trapezoid, integrated for capacity and maxed for the barrier,
with the `resident`/`read` split so idleness has somewhere to live.

---

## Sharding dimensions (parallelism axes for LLM inference)

The full menu — recorded for completeness even if the blog only needs DP vs the rest.
"Per-step collective" = what fires on **every generated token** during decode/sampling
(every token is one forward pass through all layers, so any per-layer collective is
per-token).

| axis | what it splits | per-step collective during decode | where it lives |
|---|---|---|---|
| **Data parallel (DP)** | nothing — replicate the model, split *requests* across replicas | **none** — replicas are independent … *unless* fused with EP (below) | across instances |
| **Tensor parallel (TP)** | weight matrices *within* each layer | **all-reduce every layer** (after attn, after MLP) | inside one NVLink node (tight) |
| **Pipeline parallel (PP)** | layers into sequential stages | point-to-point send between stages + pipeline bubbles (not a global barrier) | across nodes |
| **Expert parallel (EP)** | MoE experts across devices | **all-to-all every MoE layer** (dispatch tokens to their experts, combine back) | across the EP group (cross-node) |
| **Context / sequence parallel (CP/SP)** | the sequence/context dimension | attention-time comms (ring attention / K,V all-gather) | long-context regime |

Which of these is the "barrier" that sets `T_step = max_g T_local + T_sync`:
- **TP** all-reduce is a real per-token barrier, but intra-node — it lives *inside* what we
  call one worker (one TP=8 node = one logical worker), so it's not the cross-worker coupling.
- **EP** all-to-all is the cross-node, per-token global collective. **This is the barrier
  the load-balancing story is about.**
- Plain **DP** replicas have no per-step sync — that's Nie's "8 independent replicas."

---

## The coupled-DP architecture: **DP Attention + EP**

Chen et al.'s premise — "data-parallel ranks coupled by a per-step collective barrier" —
sounds like a contradiction (DP replicas are supposed to be independent). It isn't; it
describes a specific, now-dominant MoE serving layout:

- Run the **attention layers data-parallel** — each rank keeps its own batch and its own
  KV cache. You do this because attention is memory-bound and KV-heavy; TP-sharding it
  either splits or replicates the KV wastefully. (Especially true with DeepSeek's **MLA**,
  which already shrinks the KV cache — DP-ing attention avoids re-duplicating it.)
- Run the **expert/FFN layers expert-parallel (EP)** across those *same* GPUs.
- Every decode step, the MoE **all-to-all** shuffles tokens from every attention-DP rank
  out to the experts and back. So the DP ranks **synchronize around the MoE layer every
  step.** A rank with heavier resident KV finishes attention late, reaches the all-to-all
  late, and stalls everyone. → `max_g T_local + T_sync`.

**Name:** *DP Attention* (Data-Parallel Attention) + Expert Parallelism (EP). The experts
are **EP, not TP** — "TP MoE" is a misnomer.

**Introduced:** **SGLang v0.4**, Dec 2024 (LMSYS / SGL team), for DeepSeek-series MLA
models — "each DP worker independently handles prefill/decode/idle batches, synchronized
before and after the MoE layer." Reported ~1.9× decode throughput. Scaled to a 96×H100,
PD-disaggregated, large-scale-EP deployment in the May 2025 LMSYS writeup.
- SGLang v0.4 / DP attention: https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/
- Large-scale EP (96×H100, PD disaggregation): https://www.lmsys.org/blog/2025-05-05-large-scale-ep/

**Why this matters for the two papers:**
- Dense model, independent DP replicas → **no barrier → Nie's regime** (routing can't help).
- Large MoE, DP-attention + EP → the EP all-to-all **is** the barrier → **Chen's regime**
  (imbalance is the dominant cost, >40% idle in their trace).

---

## BF-IO's lookahead: billed as central, proved irrelevant, and it assumes away spec decode

**What it is.** BF-IO picks, each step `k`, the assignment minimizing predicted imbalance
summed over a short window: `J(S) = Σ_{h=0}^{H} Imbalance(k+h)`. `H` = lookahead horizon.
It explicitly does **not** predict output lengths ("does not require accurate prediction
of the total remaining workload of newly arriving jobs"); it uses "the near-future
evolution of the **currently active** jobs."

**The catch (verified against `sources/2601.17855`).** The headline bound — Theorem 2,
`IIR = Ω(√(B log G))` improvement over FCFS — is proved for **`H=0`**. Their words:
"we therefore focus on the case `H=0` … minimizes the **current-step** imbalance"; "even
this purely myopic variant achieves substantial improvement"; the guarantees "do not
depend on `H>0`." So:
- **Proven** = myopic, forecast-free, current-load balancing (≈ best-fit / JSQ in the
  barrier metric). Robust, boring, and basically what production least-loaded routers
  already do.
- **Lookahead (`H>0`)** = the "central idea," empirically-only, theoretically inert, and
  **simulation-only — not something that exists in production today.**

**Why the "it's not really forecasting" defense is weak — speculative decoding.** The
lookahead looks robust *only* if an active job's KV grows by a known **+1 token/step**;
then the near-future is deterministic and you only sweat completions. **Spec decode breaks
this.** A draft model (EAGLE / Medusa / n-gram) proposes `k` tokens, the target verifies
in one pass and appends a **variable, content-correlated chunk** (0…k+1 accepted) per
step — big chunks on predictable text (code boilerplate), small on novel text. So under
spec decode the near-future growth of active jobs is **itself stochastic and
workload-dependent**; predicting it is a real forecasting problem, not a free deterministic
read. The lookahead's "weaker, more robust signal" quietly re-becomes the hard thing it
claimed to avoid. (It does *not* kill the barrier premise — per step you still read
~resident KV, so imbalance still matters; spec decode breaks growth-*determinism*, not the
barrier. It does mildly push decode toward compute-bound by amortizing weight loads over
`k` tokens, nudging against "`T_local` linear in resident KV.")

**Net.** The lookahead is (1) not what earns the theorem, (2) not deployed anywhere, and
(3) built on a +1/step idealization that modern spec-decode serving already violates. Cite
the *myopic `H=0`* result; discount the lookahead.

**Where we stand.**
- Our `bfio`/`cbio` are the **`H=0`** policy — current-load balancing, no lookahead. Every
  result we lean on sits on the proven, forecast-free part.
- Our sim *also* assumes +1/step (`sim.py: r.d += 1`), **no spec decode** — already flagged
  in `the-think-gap.md` §9. The +1/step idealization is a **shared** limitation to name,
  not a stick just for Chen.
- Pre-staging inverts their move honestly: it forecasts **arrival** (return-time after a
  think gap), never **duration / acceptance-length**, and we *show* the win's dependence on
  predictor quality (E8 σ-sensitivity) instead of hiding behind an `H=0` theorem.

---

## Open wrinkle in *our* model (flag before publishing)

Our model card says **70B dense** (`hardware.py: N_PARAMS = 70e9`, no expert count), yet
`hardware.py` hard-codes `T_SYNC = 2e-3  # EP all-to-all per decode step` and the whole
simulator assumes the nodes are barrier-coupled. A dense 70B on TP=8 has **no EP
all-to-all** — separate nodes would be independent DP replicas (Nie's world), no barrier.

So we're implicitly modeling a **DP-attention + EP (MoE)** deployment while labeling the
model dense. Two honest fixes:
1. **Make the model an MoE** (add an expert count; route the FFN through EP). Then the
   per-step all-to-all is physically justified and the barrier is real. Cheap, and it
   *strengthens* the pre-staging result — MoE decode is even more KV-bound per FLOP, so
   resident-KV imbalance bites harder. **Preferred.**
2. Keep it dense but reframe "barrier" as a stand-in for whatever couples the pool, with
   the caveat stated. Weaker.

Ties directly to the `W_FABRIC` known-issue: cross-node KV migration rides the *same*
fabric as the EP all-to-all, which is why moving KV can delay the collective every rank
is waiting on.

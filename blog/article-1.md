# A KV cache is an area, not a size

*Two recent papers put the economics of LLM serving on a firm footing — one from queueing
theory, one from load balancing. They look unrelated. They are the same picture read two ways,
and once you see the picture, what to do about an agentic-coding workload falls out of it.*

*This is part one of two. Everything here follows from the two papers themselves, honestly
generalized — no new assumptions, no numbers you have to take on faith. Part two breaks one of
their assumptions on purpose.*

---

## The workload

Agentic coding is a specific and punishing way to use a model. A session is a long
conversation, and each turn's prompt is the **entire previous turn** — the context plus the
model's own output — carried forward as a literal prefix, with a small delta bolted on (a tool
result, a test log). Contexts run 40k–260k tokens; the model emits maybe 500; then comes a
**think gap** of seconds to minutes while a test runs or a human reads.

Two features of that shape lure you into mistakes. The reuse is enormous — turn *t+1* shares
almost all of turn *t*'s KV cache — so it is natural to reach for prefix caching and treat the
cache as a throughput lever. And the KV cache is huge and lives on particular GPUs, so it is
natural to treat placement as a routing problem. Both instincts are mostly wrong, and the two
papers tell you exactly why — but only after you see the object they share.

## The object: a request's footprint over time

Take one decode request. At each step it appends one token to its KV cache.[^spec] So its
**footprint** — the bytes of KV it occupies — grows by one token per step, from the prompt
length up to prompt-plus-output:

```
footprint at step t  =  prefix length  +  t
      W_i(t)         =       s          +  t            for t = 0 … output length o
```

Plot footprint against time and you get a trapezoid: it starts at the prompt size, ramps up as
the model generates, and vanishes when the request finishes. **That trapezoid is the whole
story.** The two papers are two different ways of collapsing it to a single number.

## Reading one: integrate over time → a capacity ceiling

First question: how many requests can one GPU worker sustain? A worker has a fixed KV budget —
the HBM left over after weights and workspace, call it the **budget**, perhaps 500 GB.

The mistake is to think a request's cost is its *size* (peak footprint) or its *speed*
(tokens/sec). It is neither. The cost is the **area under the trapezoid** — how much memory it
ties up, integrated over how long it ties it up:

```
footprint-area  ≈  output length  ×  ( prefix length  +  output length / 2 )
    g(s, o)     ≈        o         ×  (      s         +        o / 2       )
```

The units give it away: this is **byte·seconds** — a memory×time area, not bytes and not
bytes-per-second. A request's claim on a memory-bound server is *how much it holds for how
long*. That is the one idea the whole field is missing when it sizes a cache by its peak.

Sum that area over the requests sharing a worker, cap the sum at the budget, and you get a
ceiling on the sustainable request rate:

```
sustainable rate  =  budget  ÷  ( batch factor  ×  average footprint-area )
      μ           =    M      ÷  (     b̄        ×          E[g]          )
```

The first paper (Nie, Si & Zhou) makes this rigorous — for a **single worker** — and proves
something sharp: no scheduling policy beats this ceiling. Being *work-conserving* (never idle
while requests wait) is all it takes; the order and batching you pick cannot move the wall. That
is the whole theorem: one GPU, scheduling only. It says nothing about routing across GPUs — and
it does not need to, because memory is additive, so spreading requests over more workers just
sums the same ceiling. Routing moves byte·seconds around; it never reduces them. They confirmed
the single-worker result on real A100s to within ~10%. Call this reading **memory-as-capacity**.

Notice what it does *not* involve: coordination. Their eight-GPU test is independent replicas —
requests round-robined across eight separate servers, no cross-talk, capacity just multiplied by
eight. Hold that thought.

## Reading two: max across workers → a barrier

Now put the same trapezoids on a real multi-GPU deployment, and a second cost appears that the
first reading cannot see.

Large models are served with their layers split across GPUs. In the layout that is now standard
for big mixture-of-experts models — attention run **data-parallel**, experts run
**expert-parallel** across the same GPUs (the SGLang / DeepSeek arrangement) — the GPUs
**synchronize on every single decode step**: each step, an all-to-all collective shuffles tokens
out to their experts and back. That collective is a barrier. Every rank waits for the slowest.

And the slowest rank is the one with the most resident KV to read: attention each step reads the
whole resident cache, so more resident KV means a slower step. Hence:

```
step time  =  per-token read cost  ×  busiest rank's resident KV  +  collective
    —       =           α           ×          max_g L_g           +   T_sync
```

The busiest rank sets the pace; every other rank idles for the difference. The second paper
(Chen et al.) measures this in production and finds the idle exceeds **40% per step**. At that
level, imbalance is not a tax on throughput — it *is* the throughput problem. Call this reading
**memory-as-barrier**.

## The same trapezoid, twice

Here is the unification, and the reason to hold both papers in one hand:

> **Integrate a request's footprint over its lifetime → the area → the capacity ceiling.** *(paper one)*
> **Take the max of footprints across workers each step → the barrier idle.** *(paper two)*

One trapezoid. Collapse the **time** axis and you get capacity; collapse the **worker** axis and
you get the barrier. Paper one is memory summed over time on one worker; paper two is memory
maxed across workers at one instant. They are not competing models — they are two projections of
the same object. And neither one models reuse, which is exactly the door the workload walks
through.

## The one refinement that unlocks the workload: resident vs. read

Both papers use a single quantity for "load" — the footprint. But two footprints hide inside it,
and they are equal only by accident:

```
resident  =  KV sitting in a worker's HBM      →  this is what the budget caps  (capacity)
read      =  KV a worker must READ this step   →  this is what sets the barrier (idle)
```

In the base model they are the same number. In an iterated workload they come apart — because
between turns, a session's KV is **resident but not read**. The human is thinking; the test is
running; the KV is sitting in HBM doing nothing. That single gap between `resident` and `read`
is where the entire agentic-coding story lives. Keep the two words apart and everything below
follows; merge them and none of it is even statable.

## Agentic coding: the cache gives the decode pool nothing

Now the payoff, and the first counter-intuitive result. Turn *t+1* reuses almost all of turn
*t*'s KV. Surely that helps throughput?

It does not — because **turns of one conversation are never concurrent.** Turn *t+1* begins only
after turn *t* finishes, so two turns of one session never sit on a worker at the same instant,
so there is nothing to deduplicate, so the cache buys the *decode* pool exactly nothing. Look at
the area: `g ≈ o·(s + o/2)` is about 99.5% the prefix `s`. Caching does not shrink `s` — the
active turn still pins ~107k tokens of KV for 500 steps just to emit 500 tokens. The capacity
ceiling from paper one is untouched. A prefix-sharing attention kernel finds nothing to share:
turn it on, and on this workload the output does not change by a single token.

So where does all that reuse go? Into **prefill**. A warm turn — 1.5k new tokens attending to a
107k cached prefix — is about **47× cheaper** than recomputing the context cold. That part is
real and large. But it makes the cache a **storage** question, not a throughput or routing one:
the only thing that matters is *where a session's 107k tokens of KV live during the think gap.*

## Result one, from paper one: never pin idle KV

A turn finishes; the session's KV is now resident but idle for the next 15 seconds. Three
options.

**Pin it in HBM** so it stays instantly warm. This looks conservative and is a quiet disaster.
Idle KV occupies the budget while doing no work — it is, in paper one's own terms,
*non-work-conserving* — and paper one's arithmetic tells you the price. If a session is idle for
`I` seconds between turns and active for `T_s` per turn, pinning shrinks your concurrency by a
**duty-cycle factor**:

```
concurrency you keep  ∝  1  ÷  ( 1  +  idle time ÷ active time )
        —             ∝  1  ÷  ( 1  +      I     ÷     T_s      )
```

At coding gaps (≈15 s idle over ≈10 s of service) that is already ~2.5×; pinning through the gap
collapses active concurrency about **7.7×** and costs ~38% of goodput. Pinning is not caution —
it is a capacity leak with a friendly face.

**Discard it** and recompute next turn. Now every turn is a cold prefill — the O(s²) attention
you were trying to avoid — about **81×** the cost of simply reloading the KV, and it inflates the
prefill pool ~26×.

**Offload it node-local** — DMA it down to host DRAM and back when the turn returns. It costs
essentially no GPU time. The break-even think-gap is a single PCIe round trip, **~117 ms**; real
gaps are 1–300 seconds.

So: **offload, always. Never pin, never discard.** This is nothing but paper one, read honestly
and applied to a workload it never considered — and it *vindicates* paper one. Memory-as-
capacity, taken seriously, says do not squat on HBM with idle state.

## Result two, from paper two: cache affinity manufactures the barrier

Now the routing question. A turn arrives; its KV is offloaded at "home," the worker that served
the last turn. Where do you run it?

**Cache affinity** says: send it home — the KV is right there, first-token latency is low.
**Balance** says: send it to the least-loaded worker. Under agentic coding, affinity is a trap.
With no sharing across sessions, nothing forces workers to fill evenly, so routing every turn
back to its home piles the long-context sessions wherever they happen to have landed. The
busiest worker's resident KV runs far above the average — and paper two's barrier bites: the
whole pool idles waiting on the straggler. Measured, cache affinity runs **~15% barrier idle
against ~4% for balance** — about 11% of goodput, gone.

That is the sharp part: **paper two's 40%-idle pathology, in this workload, is caused by cache
affinity.** The router pulls the trigger. And this too is just paper two, applied honestly — it
*vindicates* Chen: the barrier dominates, and here it is cache-aware routing that feeds it.

## The standoff

Put the two results together and you are in a vise:

- **Affinity** gives you a warm local cache and cheap first-token latency — and concentrates
  load, so the barrier wrecks throughput.
- **Balance** gives you throughput — and now the session's KV is on the *wrong* worker, so you
  have paid a first-token penalty to get it there.

The obvious fix is to have both: run the session on a lightly-loaded worker *and* have its KV
already sitting there. But that means **moving the KV** from home to the balanced worker — and
moving KV is exactly the thing paper two assumes you cannot do. Its entire model rests on
*sticky* assignment: migrating a cache is impractical, so you place new work and live with it.

So, staying strictly inside the two papers, you are stuck. And notice what everything to this
point has cost: nothing but a careful re-reading of two existing models. No new assumptions, no
unmeasured constants — and both source papers come out looking *more* right, not less.

The way out requires breaking their shared assumption that a KV cache cannot move. What that
costs, whether it is worth it, and how far you can push it — reactively when a turn lands, or
proactively during the think gap — is the next article.

[^spec]: Speculative decoding appends a variable-length chunk per step rather than one token; it
    dents the "+1 per step" picture but not the shape of the argument. Set it aside for now.

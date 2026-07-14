# The Think Gap

### Within-Conversation KV Reuse, Barrier Synchronization, and Where a Session's Cache Should Live

*(Revision. An earlier draft assumed prefix similarity meant sharing **across** concurrent conversations. It does not: it means repetition **between turns of one** conversation. Two of that draft's five results die outright as a result, and are recorded as such in §9. The remaining paper is shorter, sharper, and says something more surprising.)*

---

## Abstract

In an iterated conversation — an agentic coding loop, say — turn *t+1*'s prompt contains turn *t*'s entire context as a literal prefix, including the model's own generated tokens. The reuse is enormous: with a 107k context and a 1.5k delta, a warm turn's prefill is **47× cheaper** than a cold one. It is natural to expect this to be a routing problem, and to expect the KV cache to be a throughput lever.

**It is neither.** We show:

1. **Within-conversation reuse is worth exactly nothing to the decode pool.** Successive turns of one conversation are never concurrent, so they never co-reside, so there is no deduplication: κ ≡ 1. The memory-time area g(s,o) ≈ o·(s + o/2) is dominated by *s*, and caching does not shrink *s* — the active turn still pins 107k tokens of KV for 500 steps to emit 500 tokens. The capacity ceiling μ = M/(b̄·E[g]) is untouched, Nie et al.'s policy-independence survives intact, and a cascade / shared-prefix attention kernel has literally no work to do. *Verified: cascade on and cascade off produce bit-identical simulator output.*

2. **The entire value of the cache is prefill elimination — and that is a storage problem, not a routing problem.** The question is only: where does a session's 17 GB of KV live during the think gap between turns?

3. **The answer is: node-local host memory. Never pin, never discard.** Pinning through a 15 s gap collapses active concurrency 7.7× and costs 38% of goodput; the penalty is (1 + I/T_s) in sessions-per-node and grows to 12× for chat-length gaps. Discarding costs 83× a reload, and inflates the prefill pool 26×. The break-even think time for offload is **114 ms** — the PCIe round-trip. Real gaps are 1–300 s.

4. **And now — with κ = 1, memory stops binding, and the barrier bites hard.** In the shared-prefix regime the memory cap incidentally pre-balances the pool. Remove the sharing and it doesn't: utilization falls to 67%, worker loads diverge freely, and **pure cache affinity costs 15.8% barrier idle against 4.1% for pure balance — 10.9% of goodput.** The 40%-idle pathology that motivates Chen et al. is, in this workload, *caused by cache affinity.*

5. **So: yes, take the miss.** The exchange rate is not close. Migrating a session's KV costs 0.43 node-s; a full cold recompute costs 4.76; the barrier cost of holding it on a straggler is 4.86 node-s at 16 DP ranks and 10.05 at 32. **Migrate first; when the fabric is saturated, recompute; never eat the barrier.** The prefill pool — which caching was supposed to shrink — becomes the pressure-relief valve for the fabric.

6. **The tension dissolves if you rebalance during the think gap — and it does.** The gap is 15 s; the migration is 43 ms; the KV is idle and nobody is waiting on it. Pre-stage a session's KV onto a well-balanced worker *during the gap* and the turn arrives to a warm cache on a balanced node: **balance's throughput (24,713 — within 0.02% of pure balance, +10.6% over affinity) and affinity's median TTFT (0.04 s — a 7× cut) at once,** verified in simulation (§5.3). The tension the whole cache-aware-routing literature is organized around exists only because the decision was assumed to happen at admission; in an iterated workload it need not. The one load-bearing dependency is a return-time predictor — and predicting *arrival* is far easier than predicting *duration*.

Numbers come from a discrete-step simulator with first-principles hardware constants (§6) and from closed-form fixed points where the simulator would only add noise. §9 states plainly what this does and does not license, including the results the clarification killed.

---

## 1. The two prior papers, and the shape of the gap

**Nie, Si & Zhou** collapse a request's memory trajectory to a scalar area g(s,o) ≈ o·(s + o/2), give the worker a budget M, and derive μ = M/(b̄·E[g]) — a ceiling *any* work-conserving policy attains. Routing cannot change μ, because memory is additive.

**Chen et al.** observe that with EP/TP inside the decode step, the DP ranks are coupled by a collective: T_step = max_g T_local^(g) + T_sync, with T_local linear in resident KV, and assignments sticky because migrating a KV cache is impractical. Their production trace shows barrier idle above 40% per step.

They are the same object under two functionals: Nie's g is the time-integral of Chen's per-step profile W_i = (s, s+1, …, s+o). Integrate → capacity. Take the max across workers → idle.

Neither models reuse. The question this paper started from was whether reuse creates a routing tension — *should you take a cache miss to route to a less-loaded worker?* The answer turns out to depend entirely on **what kind of reuse you mean**, and the two kinds behave completely differently. This paper is about the one that matters for agentic coding.

---

## 2. Temporal reuse is not spatial reuse

**Spatial reuse:** many *concurrent* requests share a prefix (a system prompt, a repo trunk). They co-reside on a worker, the shared blocks are stored once, and a worker's load becomes a submodular coverage function rather than a sum. This raises capacity, makes μ routing-dependent, and makes a cascade attention kernel a precondition for any of it to convert into throughput.

**Temporal reuse:** *successive turns of one conversation* share a prefix. Turn *t+1*'s prompt is turn *t*'s context plus a delta:

$$s_{z,t+1} = s_{z,t} + o_{z,t} + \Delta_{z,t+1}, \qquad \Delta \ll s$$

**These never co-reside.** Turn *t+1* begins after turn *t* completes. There is no concurrent sharing, so nothing is deduplicated, so:

$$\kappa \;=\; \frac{\sum_{i \in S} f_i}{\big|\bigcup_{i\in S}\text{blocks}(i)\big|} \;=\; 1 \quad\text{exactly.}$$

Every downstream consequence follows. Load is a plain sum again. Memory is additive again. **μ = M/(b̄·E[g]) is exactly Nie et al.'s, and routing cannot change it.** A cascade kernel, which batches attention over a prefix shared by concurrent queries, finds no shared prefix in the batch and does nothing.

*Verified.* Setting all cross-session prefix to zero in the simulator: κ = 1.000, and cascade-on vs cascade-off produce **identical** goodput (24,770), idle (4.3%), and memory (67.4%) — bit-for-bit. The kernel is a no-op.

**And caching does not reduce decode cost either.** g(s,o) = 500 × (107k + 250) ≈ 5.4 × 10⁷ token-steps. It is ~99.5% *s*. The turn holds 107k tokens hostage for 500 steps in order to emit 500 tokens. Prefix caching does not shrink *s*; it only avoids *recomputing* it. **Throughput per byte of HBM is ∝ 1/s, and the cache does not help.**

So the decode pool gains nothing. What gains is prefill:

| | cold | warm | ratio |
|---|---:|---:|---:|
| prefill a 107k-token turn | 4.76 node-s | 0.10 node-s | **47×** |

Note what the warm number is *made of*: θ_lin·Δ = 0.033 node-s of linear work, and θ_att·Δ·s = 0.067 node-s of **attention by the 1.5k new tokens against the 107k cached prefix**. Two-thirds of a warm prefill is attention against the cache. **Prefix caching makes prefill O(Δ·s), not free.**

---

## 3. Therefore: the only question is where the KV lives during the gap

A session's life is a duty cycle: **T_s seconds of service** (107k tokens of KV resident and read every step, ~500 steps, ~10.7 s) followed by **I seconds of think time** (the same 107k tokens idle — the agent is running a test, the human is reading). Three fates for that KV:

### Pin it in HBM

Idle KV is *stored* but not *read*, so it does not slow the step. It displaces active requests. With HBM holding both active and idle, Little's law gives Z/A = I/T_s, so

$$A \;=\; \frac{M/\bar f}{1 + I/T_s}, \qquad T_s = o\left(\alpha A \bar f + T_{\text{sync}}\right)$$

a fixed point (fewer active → faster steps → shorter T_s → *worse* ratio → fewer active still). Solving:

| think gap *I* | | active/node | step | tok/s/node | sessions/node |
|---:|---|---:|---:|---:|---:|
| 1 s | pin | 27.1 | 19.6 ms | 1,385 | 29.9 |
| | offload | 29.9 | 21.4 ms | 1,398 | 32.7 |
| 5 s | pin | 16.9 | 12.9 ms | 1,304 | 29.9 |
| | offload | 29.9 | 21.4 ms | 1,398 | 43.9 |
| **15 s** *(coding)* | **pin** | **3.9** | **4.6 ms** | **865** | **29.9** |
| | **offload** | **29.9** | **21.4 ms** | **1,398** | **71.8** |
| 60 s *(chat)* | pin | 0.6 | 2.4 ms | 244 | 29.9 |
| | offload | 29.9 | 21.4 ms | 1,398 | 197.7 |
| 120 s | pin | 0.3 | 2.2 ms | 123 | 29.9 |
| | offload | 29.9 | 21.4 ms | 1,398 | 365.4 |

At the agentic-coding gap of 15 s, **pinning collapses active concurrency 7.7× (29.9 → 3.9), costs 38% of goodput, and hosts 2.4× fewer sessions per node.** The penalty is the duty-cycle factor **(1 + I/T_s)** and it is vicious in exactly the regime that matters — long gaps relative to short turns. For a chat assistant it is 6.6×. **Pinning is not a conservative default; it is a catastrophe with a plausible face.**

### Discard it and recompute

4.76 node-seconds of prefill, every turn. Measured consequence: the prefill pool must run at **26.5× the utilization** of a properly-fed one — 212 prefill nodes to sustain 16 decode nodes. Cluster goodput collapses from **1,179 to 106 tokens/s per node**, an 11.2× loss, and TTFT p95 goes from 0.72 s to 12.29 s.

### Offload to node-local host memory

17.1 GB over ~300 GB/s of aggregate PCIe Gen5: **57 ms**, essentially zero GPU-seconds (it is DMA; the write into HBM consumes 1.5% of HBM bandwidth). Round-trip 114 ms.

$$\boxed{\;\text{Moving KV is } 83\times\text{ cheaper than making it, and the ratio grows with model size.}\;}$$

(The ratio is (B_link · 2P)/(c · FLOPS). Since c ∝ L·n_kv·d_head while P ∝ L·d_model², it scales as d_model²/(n_kv·d_head) — 83× here, ~12× for an 8B model. **Offload gets better as models get bigger.** MLA-style KV compression improves it further.)

**Break-even think time is I\* = 114 ms** — you cannot complete the round trip faster than that. Real gaps are 1–300 s. **Offload always. Never pin. Never discard.**

**The verdict on the CPU/SSD question, stated plainly:** it is not an optimization to be added if it helps. Under within-conversation reuse it is *the entire mechanism by which the reuse is realized at all*. Its cost — 63 GB/s per node, 21% of an otherwise-idle PCIe budget, ~0 GPU-seconds — is the cheapest thing in the paper.

---

## 4. Now the barrier bites — and cache affinity is what causes it

Here is the consequence I did not anticipate, and the one that most directly answers the original question.

With cross-session sharing, memory binds at ~95% and — because L_g = F(S_g) ≤ M — **the memory cap incidentally balances the barrier for free.** Every worker fills to M, loads equalize, and every routing policy performs within 1.4% of every other. The router barely matters.

**Remove the sharing and that safety net vanishes.** κ = 1, memory runs at 67%, nothing forces the loads to level, and worker loads diverge freely. Measured (G = 16, pure temporal reuse):

| policy | goodput | **barrier idle** | TTFT p95 | fabric GB/s/node | PCIe GB/s/node |
|---|---:|---:|---:|---:|---:|
| **pure affinity** (always go home) | 22,683 | **15.8%** | **0.10 s** | 0.5 | 57.2 |
| round-robin | 22,818 | 14.4% | 0.66 s | 27.4 | 30.8 |
| **pure balance** (JSQ / BF-IO) | **25,156** | **4.1%** | 0.68 s | 29.9 | 34.9 |
| **CB-IO** (priced) | 24,770 | 4.3% | 0.63 s | **20.5** | 44.5 |

**Balance beats affinity by 10.9% goodput and cuts barrier idle by a factor of four.** Chen et al.'s 40%-idle pathology, in this workload, is *manufactured by cache affinity*: sending every turn back to its home worker concentrates load wherever the long-lived, large-context sessions happen to live, and the barrier makes everyone else pay for it.

Note the TTFT column, though: affinity gives **0.10 s** p95 against 0.63–0.68 s for the balancers. The 0.5 s is the migration. Over a 50-turn agentic session that is 25 extra seconds of wall-clock. **There is a real product tension here, and §5.3 dissolves it.**

CB-IO's contribution is visible in the fabric column: 98.5% of pure-balance goodput at **31% less barrier-critical fabric traffic** (20.5 vs 29.9 GB/s/node — 1.2 migrations/s/node instead of 1.75, i.e. 39% of turns migrated instead of 56%).

---

## 5. The exchange rate

### 5.1 The two costs are different kinds of object

| | incidence | duration | who pays |
|---|---|---|---|
| **Cache cost** | one-time | at admission | one prefill node, or one link |
| **Barrier cost** | **recurring** | **every step, for ô steps** | **all G decode nodes** |

$$C_{\text{barrier}}(i,g) \;=\; (G-1)\cdot\hat o_i\cdot\alpha\cdot\big(L_g + f_i - L_{\max}\big)_+ \quad\text{[node-seconds]}$$

The hinge matters: the cost is zero unless the placement creates a *new* maximum. Fill valleys freely. The tension only bites when the session's home is *already* the straggler — which, under affinity, is exactly what happens to the workers holding the biggest sessions.

### 5.2 The numbers, for a 107k-token session with 500 steps remaining

| action | cost (node-seconds) |
|---|---:|
| stay home, KV in HBM | **0** |
| stay home, reload from host DRAM | **0.057** |
| **migrate to another node** (fabric, charged W=10 for collective contention) | **0.43** |
| **cold recompute on a prefill node** | **4.76** |
| **barrier cost of holding it on the straggler, G = 16** | **4.86** |
| **barrier cost of holding it on the straggler, G = 32** | **10.05** |
| barrier cost, G = 64 | 20.42 |

$$\boxed{\;\text{migrate} \;\ll\; \text{recompute} \;\approx\; \text{barrier}\,(G{=}16) \;<\; \text{barrier}\,(G{=}32)\;}$$

**A three-tier policy falls straight out:**

1. **Migrate the KV** — 11× cheaper than the barrier it avoids. Do this by default.
2. **When the fabric budget is exhausted, recompute** — still cheaper than the barrier above G ≈ 16, and by 2× at G = 32.
3. **Never eat the barrier.**

The fabric budget is real and binding: at 400 GB/s per node, of which the EP all-to-all consumes a large share and any KV traffic delays it, only ~1–2 migrations/s/node are affordable against a turn rate of ~3/s. **So the prefill pool — the thing prefix caching was supposed to shrink to 1:3 — reappears as the pressure-relief valve for the fabric.** Size it for migration overflow, not for steady-state prefill. That is a provisioning result we have not seen stated.

**And it inverts the design philosophy of production cache-aware routers.** They treat prefix hits as close to sacred and load as a guardrail. In a barrier-synchronized decode pool with within-conversation reuse, the ordering is reversed: **load is the objective; the cache is the tiebreak.**

### 5.3 The think gap dissolves the tension — measured

Everything above treats routing as a decision made *when the turn arrives*, under time pressure, trading TTFT against balance. **That framing is a mistake, and the workload hands us the way out.**

The think gap is **15 seconds**. The migration is **43 ms** of raw fabric time (0.44 node-s after the contention charge). The KV is idle. **Nobody is waiting on it.** The feasibility is not close: a 107k-token session is 17.5 GB, crosses a 400 GB/s fabric in 44 ms, and fits inside a 15 s gap **342× over** (closed form in `analytics.prestage_economics`).

So: **pre-stage it.** During the gap, move the session's KV to whichever worker will be lightly loaded when it returns, so the turn arrives to a warm cache on a well-balanced worker. Crucially, **the router does not change.** A pre-staged session is simply sticky to its (relocated) home; the balancing has already happened, in the gap. We implement four pieces:

- a **background migration queue** (`Sim._prestage`) that scans idle sessions each step, soonest-return-first, and moves the most-imbalancing ones;
- a **fabric-budget token-bucket** that meters migrations to a `rate` per node per second — the §5.2 constraint made explicit rather than merely charged;
- a **return-time predictor** (`Sim._predict_return`) — log-normal noise on the true return, σ a free parameter, σ = 0 the oracle;
- an **inbound load model** — each node tracks the KV already committed to arrive, so a burst of concurrent stages spreads across nodes instead of piling onto whichever node looks lightest at the instant of the decision.

**The result** (E8, G = 16, 5-seed means, pure temporal reuse):

| policy | goodput | barrier idle | TTFT p50 | TTFT p95 | staged/s | gap fabric |
|---|---:|---:|---:|---:|---:|---:|
| pure affinity | 22,340 | 14.8% | **0.04 s** | **0.12 s** | — | — |
| pure balance (BF-IO) | 24,717 | **4.1%** | 0.31 s | 0.71 s | — | — |
| CB-IO (priced) | 24,583 | 5.0% | 0.21 s | 0.62 s | — | — |
| **think-gap pre-stage** | **24,713** | **4.6%** | **0.04 s** | **0.29 s** | 32 | 28.5 GB/s |

*(All four rows are 5-seed means from one experiment, E8, on the current constants; the α refresh noted in §6 puts them ~2% below §4's earlier figures, but they are internally comparable.)*

**Pre-staging takes both.** Goodput matches pure balance to within 0.02% (24,713 vs 24,717 — +10.6% over affinity); barrier idle matches balance (4.6% vs 4.1%); and **median TTFT drops to affinity's floor, 0.04 s** — a 7× improvement over balance's 0.31 s — with p95 at 0.29 s against balance's 0.71 s. The fabric it spends (28.5 GB/s/node) is spent *in the gap, off the barrier's critical path*, not at admission; every staged session lands on its intended target (hit rate 1.00) and reloads locally over PCIe rather than migrating on arrival.

**A modest budget is a feature, not a limitation.** Sweeping the token-bucket, the optimum is `rate = 2/s/node, lookahead = 1 s` — precisely the ~1–2 migrations/s/node §5.2 said the fabric could afford. Below it (rate = 1) only half the returns are staged and the p95 tail stays at balance's 0.60 s; above it (rate = 3+) the scheme stages marginal, non-imbalancing sessions, and stages them *earlier*, against a load prediction that has gone stale by return time — throughput falls back toward affinity (23,786, idle 9.3%). The budget is not a cost to be minimized; it is the filter that keeps pre-staging trained on the most-imminent, most-imbalancing migrations. The lookahead cuts the same way: stage *close* to the return, or the node you picked is the wrong one by the time the turn lands.

**The load-bearing dependency is the predictor**, exactly as §5's verdict is load-bearing on W. E8's noise sweep:

| predictor noise σ | goodput | barrier idle |
|---|---:|---:|
| 0 (oracle) | 24,800 | 4.6% |
| 0.3 | 24,586 | 4.5% |
| 0.6 | 24,154 | 8.1% |
| 1.0 (no skill) | 22,866 | 14.4% |

At σ ≤ 0.3 the full win survives; by σ = 1.0 it has collapsed back to affinity's numbers. The failure mode is benign — a mispredicted session is staged to a *wrong-but-still-local* node and lands there anyway, so TTFT stays low (p95 actually *improves*) while balance is what slips. We predict *arrival*, not *duration* — far easier than the output-length prediction Chen et al. correctly avoid — so σ ≤ 0.3 is a plausible target. But it is a target, not a measurement: **σ is the second number to earn on a real trace, after W.**

The affinity-versus-balance tension that the entire cache-aware-routing literature is organized around **exists only because everyone assumed the decision must be made at admission.** In an iterated workload it does not. The gap between turns is the scheduling resource, and it was sitting there unused.

---

## 6. Simulation: methodology and constants

Discrete-step simulator; 16 decode nodes with a global per-step barrier; sticky assignment; PD-disaggregated prefill pool at G/2; time-based warm-up (80 s discarded) and 70–180 s of steady-state measurement. Hardware constants derived from first principles for a **70B-class model, TP=8 H100 node, fp8 weights and KV**:

| quantity | value | derivation |
|---|---|---|
| c (KV bytes/token) | 160 KB | 2 × 80 layers × 8 KV heads × 128 dim × 1 B |
| B_HBM (per node) | 26.4 TB/s | 8 × 3.3 TB/s |
| **α** | **6.06 ns** | c / B_HBM — sec per resident token per decode step |
| M (KV budget/node) | 3.2 M tokens | ~520 GB after weights and workspace |
| step time at full M | **21.4 ms** | ≈ 47 tok/s/stream — correct for long-context decode |
| T_sync (EP all-to-all) | 2 ms | |
| θ_lin | 2.22e-5 s/tok | 2P / FLOPS (P = 70e9, FLOPS = 6.3e15 at ~40% MFU) |
| θ_att | 4.16e-10 s/tok/tok-ctx | 4·d_model·L / FLOPS |
| B_PCIe (per node) | 300 GB/s | 8 × PCIe5 x16, derated for host memory bandwidth |
| B_fabric (per node) | 400 GB/s | 8 × CX-7 |
| **W (fabric contention weight)** | **10** | **a judgment call — see §9** |

**Workload:** 1,200 conversations; **no cross-session prefix of any kind**; contexts log-normal (median 107k) growing by o + Δ each turn to a 260k cap; outputs log-normal (median 500 tokens); deltas log-normal (median 1.5k); think gaps log-normal (median 15 s, heavy tail). The system runs at **67% of HBM** — *not* memory-bound, which is the regime that makes the barrier matter.

---

## 7. What this says to a system designer

Ranked by measured effect:

1. **Offload session KV to node-local host DRAM (then NVMe). Never pin it in HBM; never discard it.** Pinning costs 38% of goodput and 2.4× the fleet at coding think times; discarding costs 83× a reload and inflates the prefill pool 26×. This is the whole ballgame and it is a storage decision, not a routing one.
2. **Do not expect the cache to help decode. It cannot.** κ = 1, μ is unchanged, and a cascade kernel is a no-op. If someone shows you a throughput win from within-conversation prefix caching, it came from the prefill pool.
3. **Balance the decode pool; do not chase hit rate.** Affinity costs 10.9% goodput and quadruples barrier idle. Migrating a session's KV is 11× cheaper than the straggler it prevents; above ~16 DP ranks even a full cold recompute is cheaper.
4. **Rebalance during the think gap.** 15 s of slack against a 43 ms move buys balance's throughput at affinity's TTFT (§5.3, verified: +10.6% goodput over affinity, median TTFT held at 0.04 s). It costs a return-time estimate — and the win degrades gracefully as that estimate does.
5. **Keep KV off the fabric except deliberately.** Node-local offload rides PCIe (idle); migration rides the fabric (barrier-critical). Budget the fabric explicitly and let the prefill pool absorb the overflow.
6. **Size the prefill pool at ~1:3, plus migration overflow.** Even with a perfect cache, the delta's attention against the cached prefix is irreducible — two-thirds of a warm prefill.

---

## 8. Relation to prior work

**Nie, Si & Zhou** are *right*, and more right than their own framing claims. Under within-conversation reuse κ = 1, capacity is genuinely routing-independent, and their μ = M/(b̄·E[g]) is exact. Our earlier draft claimed to generalize them with a routing-dependent κ; under the correct workload that generalization is vacuous. Their result stands.

**Chen et al.** are right that the barrier dominates, and our numbers support them: 15.8% idle under affinity, and this is with a *balanced-by-construction* synthetic workload. But their model has no notion of where a request's KV already is, and the decode-side balance they optimize is in direct tension with the prefill-side locality they never model. **The joint (prefill node, decode node) decision, with the transfer between them priced against the collective it contends with, is the problem their paper implies and does not pose.**

The most useful thing we can say back to them: **their lookahead may be solving the wrong problem.** They predict whether *active* requests will finish in the next H steps, in order to place *new* arrivals well. In an iterated workload there is a far stronger signal available for free — you know a session will return, roughly when, and exactly how much KV it will bring — and you have 15 seconds of idle time in which to act on it. **Predicting returns beats predicting completions.**

**Cache-aware routers** (SGLang's router, Preble, Mooncake) do prefix affinity with a load guardrail. Under within-conversation reuse the guardrail should be the objective. We have not found prior work that prices barrier cost against cache cost in a common currency, nor any that exploits the inter-turn gap as a rebalancing window.

---

## 9. Limitations, and the results this revision killed

**Killed by the clarification** (recorded because the reasoning was wrong, not merely the conclusion):

- **"Capacity is routing-dependent," μ(π) = κ(π)·M/(b̄·E[g]).** True under *spatial* sharing; vacuous under temporal reuse, where κ ≡ 1. Nie et al.'s policy-independence stands.
- **"A cascade kernel is a precondition."** True under spatial sharing (it converts κ into throughput, worth +20%); a *no-op* under temporal reuse — verified bit-identical.
- **"Square-root replication of hot prefixes."** Already refuted empirically in the earlier draft; now also inapplicable, since a conversation's history has fan-out 1.

If your workload has *both* kinds of reuse — a shared system prompt across sessions *and* within-session growth — both papers apply and the effects roughly compose. Our simulated system prompt of 15k against 107k contexts gives κ ≈ 1.18, worth ~18% via cascade. Worth having; not the main event.

**Still standing, but caveated:**

- **These are simulation results from a simulator we wrote.** The hardware constants are first-principles and the sanity anchors (21.4 ms step, ~47 tok/s/stream, 67% memory, 1:3 prefill:decode) are plausible. Nothing here has touched a GPU. Nie et al. validated on real A100s within ~10%; Chen et al. simulate. We simulate.
- **W = 10 is load-bearing and unmeasured.** The whole "migrate rather than stall" conclusion depends on migration being ~11× cheaper than the barrier; at W = 1 it is 110× cheaper (the conclusion strengthens), at W = 100 it is comparable (the conclusion inverts and affinity returns). **This is the first number to measure.**
- **T_local is modeled as exactly linear in resident KV**, following Chen et al. Real kernels have fixed overheads; the linear model flatters the barrier story.
- **Think-gap pre-staging is now tested in simulation (§5.3, E8), but its win is contingent on the return-time predictor.** At prediction noise σ ≤ 0.3 it delivers balance's throughput at affinity's TTFT; at σ = 1.0 (no predictive skill) it collapses to affinity's numbers. The failure mode is benign — a misprediction stages to a wrong-but-local node, so TTFT holds while balance slips — and we predict arrival rather than duration, which is the easy direction. But σ is unmeasured on a real trace, and like W = 10 it is load-bearing. The scheme also uses a flat fabric-budget token-bucket, not the three-tier migrate → recompute → never-stall policy §5.2 implies; and it selects targets from instantaneous-plus-inbound load, a proxy for a true per-node forecast at the predicted return instant.
- **No KV quantization, no speculative decoding, no chunked-prefill overlap.** Int4 KV for the offload tiers would change *c* by 4× and shift every bandwidth number — probably in offload's favour.

---

## Appendix: the one-line version

- **Between-turn reuse gives the decode pool nothing.** κ = 1. The cascade kernel is a no-op. μ is unchanged.
- **Its entire value is prefill: 47×. That makes it a storage problem.**
- **Never pin (−38%, 2.4× fleet), never discard (83×). Offload node-locally.** Break-even gap: 114 ms.
- **Without sharing, memory stops pre-balancing the pool — so the barrier bites, and cache affinity is what causes it.** 15.8% idle vs 4.1%.
- **Take the miss.** Migrate (0.43 node-s) ≪ recompute (4.76) ≈ barrier (4.86 at G=16, 10.05 at G=32).
- **And then stop making the decision at admission.** You have a 15-second gap and a 43-millisecond move — pre-stage into it and you get balance's throughput at affinity's TTFT (verified in simulation; contingent on a return-time predictor).

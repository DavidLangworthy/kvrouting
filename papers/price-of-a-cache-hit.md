# The Price of a Cache Hit

### Prefix Locality, Barrier Synchronization, and KV Offload in Disaggregated LLM Serving

---

## Abstract

Two recent lines of queueing-theoretic work on LLM inference treat the KV cache as two different objects. Nie, Si & Zhou treat it as a **capacity**: each request consumes a lifetime memory-time area, the GPU supplies a fixed budget of it, and the ratio gives a throughput ceiling that any work-conserving policy attains. Chen et al. treat it as **sticky, non-migratable state**: in data-parallel decoding behind a per-step collective barrier, the step is gated by the worker holding the most resident KV, and production traces show over 40% of compute lost to barrier idle. Neither models the one thing that dominates agentic workloads: **prefix reuse**.

We show that under prefix reuse the two objects become one, and that routing acquires powers it does not have in either prior model. A worker's load is no longer a sum over its requests but the **union of their KV blocks** — a monotone submodular coverage function. This single change has four consequences:

1. **Capacity becomes routing-dependent.** The stability threshold picks up a multiplicative deduplication factor κ set by the routing policy: μ(π) = κ(π)·M / (b̄·E[g(s,o)]). The policy-independence of the prior stability result is exactly the κ ≡ 1 special case.
2. **Prefix sharing buys nothing without a cascade kernel.** In the KV-bandwidth-bound regime, sharing reduces *stored* KV but not *read* KV. Per-worker goodput is 1/(α·f̄) regardless of how much the requests share, unless the attention kernel deduplicates shared-prefix reads. Measured: cascade converts the entire sharing factor into throughput (+20%, matching κ = 1.23 to within 2%); without it, sharing saturates memory and collapses admission. **A shared-prefix attention kernel is a precondition for cache-aware routing, not an optimization.**
3. **A cache hit is worth far less than the literature assumes.** The cache cost of a placement is *one-time and local*; the barrier cost is *recurring and global*, scaling with the pool size G and the remaining output length ô. The resulting **miss-tolerance ratio** crosses 1 against a *full cold recompute* at G ≈ 16 for a 70B-class model at 100k context. Above that, it is cheaper to recompute a 100k-token prefix from scratch than to place a request on a worker that would become the straggler. Against a warm node-local fetch, the barrier is 85–722× more expensive. Cache-aware routers that maximize hit rate are optimizing the wrong term.
4. **CPU/SSD KV offload is not an optimization; it is the difference between a 20-node cluster and a 228-node cluster.** Moving a token of KV over a node-local link is 81× cheaper than recomputing it, and the ratio *grows* with model size. Without offload, the prefill pool must be 26× larger to sustain the same decode pool. With it, PCIe runs at 21% of budget. **But it must be node-local**: cross-node KV migration contends with the EP all-to-all on the barrier's critical path and is charged roughly an order of magnitude above its raw bandwidth cost.

We also report a **refuted hypothesis**. We derived a square-root replication rule for hot prefix nodes, implemented it, and found it *reduces* throughput: pinned trunk replicas consume HBM — the binding resource — to buy balance that the memory cap already provides for free. Replication should be emergent from the coverage function, not an explicit mechanism.

All numbers below come from a discrete-step simulator with first-principles hardware constants, described in §6. Like Chen et al., we simulate; unlike Nie et al., we do not have GPU measurements. §9 is explicit about what this does and does not license.

---

## 1. What the two prior papers each leave out

**Nie, Si & Zhou** collapse a request's memory trajectory into a scalar: g(s,o) ≈ o·(s + o/2), the memory-time area of a request with prompt s and output o. A worker with KV budget M and batch step b̄ supplies M/b̄ token-steps per second, giving

$$\mu \;=\; \frac{M}{\bar b \cdot \mathbb{E}[g(s,o)]}$$

with instability above μ under *any* policy and stability below μ(1−δ). The result is clean precisely because memory is **additive**: whichever requests you pack, the GPU retires M token-steps per step. Routing cannot change μ. Their 8-GPU validation is eight *independent replicas* under round-robin, and the cluster rate is 8 × μ_single.

**Chen et al.** point out that eight independent replicas is not what production looks like. With expert- or tensor-parallelism inside the step, the DP ranks are coupled by a collective, so

$$T_{\text{step}}(t) \;=\; \max_g T^{(g)}_{\text{local}}(t) \;+\; T_{\text{sync}}(t)$$

and T_local is linear in the worker's resident KV. Assignments are sticky because migrating a KV cache is impractical. Their trace shows mean and median barrier idle above 40% per decode step. This quietly guts the prior paper's cluster result: 8 × μ_single is clean only because independent replicas have no barrier.

The two papers are the **same object under two functionals**. Nie's g(s,o) is the time-integral of Chen's per-step workload profile W_i = (s, s+1, …, s+o); their own worked example, W_i = (3,4,5,6) for prefill 3 and four decode steps, is literally the memory trajectory whose sum is g. Integrate over time → capacity ceiling. Take the max across workers → barrier idle.

Neither models reuse. In both, request i's footprint is its own, and the cost of putting two requests on one worker is the sum of their costs. That assumption is false for the workload that matters most.

---

## 2. The workload: iterated coding agents

An agentic coding session is not a sequence of independent requests. Turn *t+1*'s prompt **contains turn *t*'s entire context as a literal prefix** — including the model's own generated tokens, whose KV was already computed during turn *t*'s decode. The only genuinely new tokens are the delta: a tool result, a test log, a user message.

$$s_{z,t+1} \;=\; s_{z,t} \;+\; o_{z,t} \;+\; \Delta_{z,t+1}, \qquad \Delta \ll s$$

With s ≈ 100k and Δ ≈ 1.5k, a warm session's prefill is O(Δ·s) instead of O(s²) — 45× less work in our parameterization. And the reuse is not only temporal. The prefix tree has three levels, each with a different fan-out and a different economics:

| Level | Length | Concurrent fan-out | Reuse type |
|---|---|---|---|
| **Root** — system prompt + tool schemas | ~15k | *every session* | spatial |
| **Trunk** — repo context, CLAUDE.md, key files | ~20k | sessions on that repo (Zipf) | spatial |
| **Branch** — session history | 40k–250k, growing | 1 (but every turn) | temporal |
| **Leaf** — the delta | ~1.5k | 0 | none |

This yields a clean decomposition that organizes everything below:

- **Spatial sharing** (concurrent requests sharing root/trunk) → raises κ → raises capacity. Requires **co-location** *and* **a cascade kernel**.
- **Temporal sharing** (successive turns of one session) → eliminates prefill. Requires **retention** (HBM / DRAM / SSD) *and* **routing back to where the KV lives**.

These are different mechanisms, with different costs, and different failure modes. Conflating them is what makes "cache-aware routing" sound like one problem when it is two.

---

## 3. Model

**Topology.** A pool of *P* prefill nodes and *G* decode (sampling) nodes. Each node is a TP group treated as one logical worker. The decode pool is data-parallel with a per-step collective barrier (EP all-to-all), following Chen et al. Assignment is sticky.

**Prefix tree.** Each request *i* is a leaf; its KV footprint is the root-to-leaf path. Co-locating requests on a worker stores their shared path **once** (radix/paged KV with copy-on-write, as in RadixAttention and vLLM prefix caching).

**The coverage function.** For a set *S* of requests on worker *g*:

$$F(S) \;=\; \Big|\bigcup_{i \in S} \text{path}(i)\Big| \quad \text{(tokens)}$$

This is a **weighted coverage function**: monotone, submodular, and *not* a sum. Define the sharing factor

$$\kappa(S) \;=\; \frac{\sum_{i \in S} f_i}{F(S)} \;\geq\; 1$$

where f_i = s_i + d_i is request *i*'s nominal footprint. This single object replaces both prior papers' load terms:

- **Memory constraint:** F(S_g) ≤ M.
- **Barrier load:** $L_g = (1-\rho)\sum_{i \in S_g} f_i + \rho\, F(S_g)$, where ρ ∈ [0,1] is the fraction of shared-prefix KV reads the attention kernel deduplicates. **ρ = 0** for a standard paged-attention kernel (each request reads its full context every step, regardless of sharing). **ρ = 1** for a cascade / shared-prefix kernel (Hydragen, FlashInfer's multi-level cascade), which computes attention over the shared prefix once as a batched GEMM and merges via log-sum-exp.

$$T_{\text{step}} = \alpha \max_g L_g + T_{\text{sync}}, \qquad \alpha = \frac{c}{B_{\text{HBM}}}$$

with *c* bytes of KV per token. **Everything in this paper follows from the fact that F is submodular and that ρ decides whether the barrier sees F or the sum.**

---

## 4. Theory

### T1 — Capacity is routing-dependent

Repeat the Foster–Lyapunov argument with F in place of Σ. The outstanding memory-time area V(t) is retired at a rate equal to the **nominal** area advanced per step, Σ_{i∈S} f_i = κ·F(S) ≤ κ·M, not M. Hence the drift condition becomes λ·b̄·E[g] < κ·M(1−δ), and

$$\boxed{\;\mu(\pi) \;=\; \kappa(\pi)\cdot\frac{M}{\bar b \cdot \mathbb{E}[g(s,o)]}\;}$$

where κ(π) is the time-averaged deduplication factor achieved by policy π. **Nie et al. is the κ ≡ 1 corollary**, and their policy-independence result holds *because* additive memory makes κ ≡ 1 for every policy. Under sharing, routing sets κ, and therefore sets the stability region. A cache-oblivious router that scatters same-trunk requests across G workers drives κ → 1 and forfeits the entire margin.

*Measured:* κ = 1.22–1.31 across policies (§7). The capacity gain is real (+22–31%), but the *policy* lever on κ is weaker than the theorem permits — see T4.

### T2 — Prefix sharing is throughput-neutral without a cascade kernel

Consider a worker in the KV-bandwidth-bound regime (the relevant one: at c = 160 KB/token and f̄ = 100k, one request's KV is 16 GB, dwarfing the per-node weight read; batching amortizes weights, not KV). Let the memory constraint bind, F(S) = M.

**With cascade (ρ=1):** reads = F(S) = M. Requests resident: |S| = κM/f̄. Goodput = |S|/(αM) = **κ/(α f̄)**.

**Without cascade (ρ=0):** reads = Σ f_i = κ·F(S) = κM. Requests resident: still |S| = κM/f̄. Goodput = (κM/f̄)/(ακM) = **1/(α f̄)**.

$$\boxed{\;\text{cascade multiplies decode goodput by exactly } \kappa;\ \text{without it, } \kappa \text{ buys nothing.}\;}$$

Sharing without cascade converts a memory constraint into a bandwidth constraint at par. You fit more requests and each step takes proportionally longer. The memory you save buys *queue* capacity, not *service rate* — it raises the number-in-system, not μ.

**Corollary (the trap).** Under a barrier, it is worse than neutral. Affinity routing makes κ_g *unequal* across workers. Without cascade, L_g = κ_g·M, so the high-affinity worker becomes the straggler *because* it is sharing well. A cache-aware router deployed on a non-cascade engine actively manufactures the stragglers that the barrier punishes.

*Measured (§7.2):* cascade-off costs 17% goodput, drives memory to 99.7%, and explodes admission blocking from ~0 to 2,630/s. The goodput ratio 22,098/18,381 = **1.20** against κ = **1.23** — the theorem, to within 2%.

### T3 — The exchange rate, and the miss-tolerance ratio

This is the answer to the question that motivated the paper: *should you take a cache miss to route to a less-loaded worker?*

The two costs are **not the same kind of object**:

| | incidence | duration | who pays |
|---|---|---|---|
| **Cache cost** C_cache(i,g) | one-time | at admission | one prefill node, or one link |
| **Barrier cost** C_barrier(i,g) | **recurring** | **every step for ô_i steps** | **all G decode nodes** |

$$C_{\text{barrier}}(i,g) \;=\; (G-1)\cdot \hat o_i \cdot \alpha \cdot \big(L_g + \Delta f_i - L_{\max}\big)_+ \quad\text{[node-seconds]}$$

Note the hinge: the cost is zero unless the placement pushes *g* above the current maximum. The router should fill valleys freely; the tension only bites when the cache-hot worker is *already* the straggler — which is exactly what hot prefixes cause.

Define the **miss-tolerance ratio** Θ = C_barrier / C_cache. Take the miss iff Θ > 1. With the constants of §6 (α = 6.06 ns/token/step, f = 100k, ô = 500):

| G | C_barrier (node-s) | vs. local DRAM fetch | vs. cross-node migration | vs. **cold recompute** | verdict |
|---:|---:|---:|---:|---:|:---|
| 4 | 0.91 | 17× | 2× | 0.21× | keep the hit |
| 8 | 2.12 | 40× | 5× | 0.49× | keep the hit |
| **16** | **4.55** | **85×** | **11×** | **1.06×** | **take the cold miss** |
| 32 | 9.39 | 176× | 23× | 2.18× | take the cold miss |
| 64 | 19.09 | 358× | 48× | 4.44× | take the cold miss |
| 128 | 38.48 | 722× | 96× | 8.94× | take the cold miss |

Read the last column carefully. **Above ~16 DP ranks, it is cheaper to recompute a 100k-token prefix entirely from scratch — the single most expensive cache action available — than to place the request on a worker that would become the straggler.** At 32 ranks it is twice as cheap; at 128, nine times.

And the warm cases are not close: a node-local DRAM fetch is 85–722× cheaper than the barrier it avoids. **In a barrier-synchronized decode pool, you should essentially always take the fetch.** Hit rate is not the objective; it is a term, and a small one.

This inverts the design philosophy of production cache-aware routers, which treat prefix hits as close to sacred and treat load as a tiebreak or a guardrail. In this regime the ordering is reversed: **load is the objective and the cache is the tiebreak.**

### T4 — Replication should be emergent (a refuted hypothesis)

We hypothesized that hot prefix nodes should be explicitly replicated. Minimizing (r−1)·ℓ + ν·f̄/r over the replication factor gives r\* = √(ν·f̄/ℓ) — a pleasing square-root law with good pedigree.

**We implemented it and it made things worse.** Explicit pinning of trunk replicas consumes HBM, which is *the binding resource* (μ ∝ M), in order to buy load balance that the memory cap **already provides for free** — when cascade is on, L_g = F(S_g) ≤ M, so the capacity constraint itself bounds the barrier load and workers auto-level as they fill.

The deeper error: **replication is not a mechanism, it is an emergent property of the coverage function.** A trunk is materialized on exactly the nodes the router sends its traffic to; the footprint term F already charges r·τ tokens for spreading a trunk across r nodes. The root trunk ends up on all G nodes automatically, at 0.47% of M each, with no mechanism at all.

*Measured (§7.4):* no pinning (cap=1) gives the best goodput of any setting tested. We report this because the discipline of the exercise demands it: the mechanism was designed, built, tested, and does not earn its keep.

### T5 — KV offload dominates, but only node-locally

An idle session between turns has three fates. Price them.

**Pin in HBM.** Occupying f/M of a decode node for the think time I. At f = 100k, M = 3.2M, that is 3.1% of a node per idle session. With think-time/service ≈ 1.5 and 30 active sessions per node, you would need to reserve roughly half of HBM for sessions that are doing nothing. HBM is the resource that sets μ. **Pinning loses.**

**Evict and recompute.** A full 100k-token prefill costs **4.30 node-seconds**.

**Offload to node-local host DRAM.** 100k tokens × 160 KB = 16 GB over ~300 GB/s of aggregate PCIe Gen5: **0.053 node-seconds** of link time, and essentially zero GPU-seconds (it is DMA; the ~50 GB/s write into HBM is 1.5% of HBM bandwidth).

$$\boxed{\;\text{Moving KV is } 81\times \text{ cheaper than making it.}\;}$$

And the ratio *improves* with scale: it is (B_link · 2P)/(c · FLOPS), and since c ∝ L·n_kv·d_head while P ∝ L·d_model², the ratio grows as d_model²/(n_kv·d_head). **Offload gets better as models get bigger.** (For an 8B model it is ~12×; still decisive, but less so.)

**The constraint that matters: node-local only.** A cross-node KV migration rides the same fabric as the EP all-to-all, which sits on the barrier's critical path *every decode step*. A 16 GB transfer at 400 GB/s is 40 ms — twenty decode steps' worth of the entire fabric. It must be rate-limited, and its true cost is far above its raw bandwidth cost. We charge it a contention weight W = 10 throughout. **This, not prefill compute, is the real reason to keep a session near its node.** Session affinity is a bandwidth argument, not a FLOPs argument.

*Measured (§7.5):* without offload, the prefill pool must be **26× larger** — 212 prefill nodes to sustain 16 decode nodes. Cluster goodput falls from **1,179 to 106 tokens/s/node**, an 11.2× collapse. TTFT p95 goes from 0.72 s to 12.29 s. The cost of offload is 63 GB/s per node of PCIe — **21% of budget**, on a link that was otherwise idle.

**Verdict on the user's gate: CPU/SSD KV caching is not a nice-to-have. It is the single largest lever in the paper, and it earns its cost by an order of magnitude.**

---

## 5. Algorithm: CB-IO (Cache–Barrier Integer Optimization)

BF-IO minimizes accumulated predicted imbalance over a short horizon. It has no notion of where a request's KV already is. CB-IO extends it by pricing every resource in one currency — **node-seconds** — so there is no free tuning parameter:

$$\min_{g}\;\; \underbrace{C_{\text{cache}}(i,g)}_{\text{one-time, local}} \;+\; \underbrace{(G-1)\,\hat o_i\, \alpha \,\big(L_g + \Delta F(i,g) - L_{\max}\big)_+}_{\text{recurring, global}} \;+\; \underbrace{\sigma \cdot \Delta F(i,g)}_{\text{capacity shadow price}}$$

$$\text{s.t.}\quad F(S_g \cup \{i\}) \le M(1-\delta)$$

with

$$C_{\text{cache}}(i,g) = \begin{cases}
0 & \text{prefix resident in } g\text{'s HBM}\\
c\,s_i / B_{\text{PCIe}} & \text{in } g\text{'s host DRAM}\\
c\,s_i / B_{\text{SSD}} & \text{on } g\text{'s NVMe}\\
W \cdot c\,s_i / B_{\text{fabric}} & \text{on another node (charged for collective contention)}\\
\theta_{\text{lin}} s_i + \tfrac12\theta_{\text{att}} s_i^2 & \text{cold recompute}
\end{cases}$$

Two structural notes.

**ΔF(i,g) = F(S_g ∪ {i}) − F(S_g)** is the marginal of a submodular coverage function — computable in O(tree depth) from the radix tree, and *this is where the prefix structure enters*. It is small exactly when *g* already holds *i*'s prefix. So the same term that prices memory also prices locality; they are not separate concerns.

**The problem class changed.** BF-IO's per-step decision is a linear assignment (min-max of *sums*). CB-IO's is min-max **submodular** allocation — the objective is not linear in the assignment variables, because the marginal cost of adding request *i* to worker *g* depends on what else is on *g*. Greedy marginal assignment is what one implements; the approximation guarantees for min-max submodular load balancing are the natural place for the theory of this paper to go next, and we do not claim them here.

**And the whole thing is dominated by (i) turning cascade on, and (ii) offloading node-locally.** The router is the third-order term. We say this plainly because our own numbers say it.

---

## 6. Simulation: methodology and constants

Discrete-step simulator, 16 decode nodes (swept to 32), a global barrier, sticky assignment, time-based warm-up (80 s discarded), 180 s of steady-state measurement. All hardware constants derived from first principles for a **70B-class model, TP=8 H100 node, fp8 weights and KV**:

| Quantity | Value | Derivation |
|---|---|---|
| c (KV bytes/token) | 160 KB | 2 × 80 layers × 8 KV heads × 128 dim × 1 B |
| B_HBM (per node) | 26.4 TB/s | 8 × 3.3 TB/s |
| **α** | **6.06 ns** | c / B_HBM — sec per resident token per step |
| M (KV budget/node) | 3.2 M tokens | ~520 GB after weights & workspace |
| step time at full M | **19.4 ms** | α·M + T_sync — a plausible long-context TPOT |
| T_sync (EP all-to-all) | 2 ms | |
| θ_lin | 2.22e-5 s/tok | 2P / FLOPS, P = 70e9, FLOPS = 6.3e15 (8×H100, fp8, 40% MFU) |
| θ_att | 4.16e-10 s/tok/tok-ctx | 4·d_model·L / FLOPS |
| B_PCIe (per node) | 300 GB/s | 8 × PCIe5 x16, derated for host memory bandwidth |
| B_fabric (per node) | 400 GB/s | 8 × CX-7 |
| **W (fabric contention weight)** | **10** | **judgment call — see §9** |

**Workload:** 1,200 coding sessions; 40 repos with Zipf(1.1) popularity; root trunk 15k tokens shared by all; repo trunks ~20k; session contexts log-normal (median 105k) growing by o + Δ each turn to a 260k cap; outputs log-normal (median 500); think times log-normal (median 15 s, heavy tail). Prefill pool sized at G/2.

The system runs **memory-bound at 78–96% of M**, which is the regime that matters and the regime in which prior work's assumptions bite hardest.

Sanity anchors: a 19.4 ms step at full memory (≈51 tok/s/stream, correct for long-context decode); an incremental prefill of a 1.5k delta against a 100k context costs 0.096 node-s, of which **65% is attention against the cached prefix** — prefix caching makes prefill O(Δ·s), not free.

---

## 7. Results

### 7.1 Baseline: six policies, cascade on, memory-bound (G=16)

| policy | goodput (tok/s) | barrier idle | κ | mem | TTFT p95 | PCIe GB/s/node | fabric GB/s/node |
|---|---:|---:|---:|---:|---:|---:|---:|
| round-robin | 22,098 | 4.3% | 1.23 | 94.8% | 1.16 s | 42.8 | 37.6 |
| JSQ (least KV) | 22,386 | 2.6% | 1.22 | 94.2% | 0.92 s | 43.8 | 38.3 |
| pure affinity | 22,351 | 3.8% | 1.23 | 95.4% | 0.90 s | 63.4 | **18.1** |
| cache-aware + load guardrail | 22,351 | 3.8% | 1.23 | 95.4% | 0.90 s | 63.4 | 18.1 |
| BF-IO (balance only) | 22,413 | **2.2%** | 1.22 | 93.3% | 1.15 s | 44.4 | 39.0 |
| **CB-IO** | 22,400 | 2.8% | 1.23 | 94.6% | **0.90 s** | 55.6 | 26.7 |

**All six policies land within 1.4% on goodput.** This is a finding, not a failure. With cascade on and memory binding, L_g = F(S_g) ≤ M — **the capacity constraint balances the barrier for free.** The throughput fight is over before the router speaks.

Where the policies *do* differ is the interconnect: fabric traffic ranges 18.1 → 39.0 GB/s/node, a 2.2× spread. Affinity halves it; balance-only doubles it. Since fabric collides with the barrier-critical all-to-all, this is the term that survives. **The router's job in this regime is not throughput. It is keeping KV off the fabric while not manufacturing stragglers.** CB-IO sits where it should: p95 TTFT of the affinity policies, fabric traffic 32% below the balance-only policies.

### 7.2 T2: the cascade kernel is a precondition

| | goodput | idle | κ | mem | blocked/s | TTFT p95 |
|---|---:|---:|---:|---:|---:|---:|
| RR, cascade **on** | 22,098 | 4.3% | 1.23 | 94.8% | ~0 | 1.16 s |
| RR, cascade **off** | 18,381 | 7.0% | 1.27 | **99.7%** | **2,630** | 8.79 s |
| JSQ, cascade **on** | 22,386 | 2.6% | 1.22 | 94.2% | ~0 | 0.92 s |
| JSQ, cascade **off** | 18,795 | 4.4% | 1.27 | **99.7%** | **2,257** | 6.87 s |

−17% goodput; memory pinned at 99.7%; admission blocking from nothing to thousands per second; TTFT p95 up 7×. The goodput ratio **22,098/18,381 = 1.20** against a measured **κ = 1.23**. T2 holds to 2%.

The mechanism is exactly as predicted: without cascade, sharing lets you *pack* more requests but each one still *reads* its full context, so step time inflates by κ and cancels the packing gain — and the surplus requests pile up against a memory wall.

### 7.3 T3: the price of locality

Sweeping the price θ on the barrier term (θ = 0 is pure cache affinity; θ = 1 is the physically-derived exchange rate):

| θ | goodput | barrier idle | fabric | mem |
|---:|---:|---:|---:|---:|
| 0 (pure affinity) | 23,129 | **10.2%** | 13.8 | 88.3% |
| 0.3 | 22,809 | 11.2% | 12.9 | 86.1% |
| **1.0 (physical)** | 23,666 | 6.4% | 21.8 | 77.8% |
| 3.0 | 23,712 | 4.7% | 24.2 | 80.9% |
| 10.0 | 23,901 | **3.7%** | 28.1 | 78.5% |

Pricing the barrier cuts idle from 10.2% to 3.7% and adds 3.3% goodput. The physically-derived price captures most of it. Goodput keeps creeping up past θ = 1 because our hinge form only penalizes *creating* a new maximum and does not reward *filling* valleys; a term that also rewards leveling would close this.

**The honest reading:** in the memory-bound regime the goodput gains from routing are modest (1–6%) *because the memory cap does most of the balancing*. The barrier-vs-cache exchange rate is nonetheless enormous — 85× to 722× against a warm fetch — and it says something the goodput number does not: **you should never hold a request on a straggler to preserve a cache hit.** The idle column is where you see it.

We also tested whether the routing gain grows with G, as the (G−1) factor might suggest. **It does not** (G=8: +5.7%; G=16: +1.3%; G=32: +1.4%). The absolute barrier cost scales with G, but so does total capacity; the *fractional* waste is G-invariant. What *does* scale with G is the exchange rate itself, because the cache cost is borne by one node regardless of pool size. The threshold moves; the fraction does not. We were wrong about this and the simulator caught it.

### 7.4 T4: explicit replication, refuted

CB-IO, cascade on, varying the cap on hot-trunk replication:

| replication cap | goodput | idle | mem | κ |
|---:|---:|---:|---:|---:|
| **1 (none — emergent only)** | **23,759** | 4.3% | 83.2% | 1.29 |
| 2 | 22,767 | 3.6% | 91.6% | 1.26 |
| 4 | 23,272 | 2.2% | 96.9% | 1.26 |
| 8 | 22,460 | 4.2% | 92.8% | 1.23 |
| 16 (all nodes) | 22,500 | 3.5% | 92.7% | 1.23 |

No pinning wins. Replication does buy balance (idle falls to 2.2% at cap=4) but pays for it in memory (83% → 97%), and memory is the resource that sets μ. **Net: negative.** The mechanism is cut.

### 7.5 T5: the offload gate

| | goodput | **prefill pool utilization** | recompute | cold prefills/s | TTFT p95 | PCIe |
|---|---:|---:|---:|---:|---:|---:|
| offload **on** | 23,666 | **0.51** | 0.34 node-s/s | 0.06 | **0.72 s** | 63 GB/s/node |
| offload **off** | 24,156 | **26.53** | 208.5 node-s/s | 34.3 | 12.29 s | 0 |

The decode pool barely notices — goodput is even marginally *higher* without offload, because sessions evicted from HBM free memory. **The prefill pool is annihilated.** A utilization of 26.5 on a pool sized at G/2 = 8 means you would need **212 prefill nodes to feed 16 decode nodes.**

Cluster economics, counting every node:

| | decode | prefill | total | **tokens/s per node** |
|---|---:|---:|---:|---:|
| offload **on** | 16 | 4.1 | 20.1 | **1,179** |
| offload **off** | 16 | 212.2 | 228.2 | **106** |

**11.2×.** The price: 63 GB/s per node of PCIe — 21% of a 300 GB/s budget, on a link that would otherwise sit idle, consuming essentially no GPU-seconds. This is the clearest result in the paper.

---

## 8. What this says to a system designer

Ranked by measured effect:

1. **Turn on a cascade / shared-prefix attention kernel before doing anything else.** It is worth +20%, and it is a *precondition*: without it, prefix sharing yields exactly zero throughput and cache-aware routing manufactures stragglers. Everything else in this paper is contingent on it.
2. **Offload session KV to node-local DRAM, then NVMe. Never recompute; never pin.** 11× on tokens-per-node. Moving KV is 81× cheaper than making it, and the ratio grows with model size.
3. **Keep KV off the fabric.** Node-local offload is nearly free; cross-node migration contends with the barrier-critical collective and is the one place where affinity genuinely pays.
4. **Price the barrier, don't worship the cache.** Against a warm fetch the barrier is 85–722× more expensive; above ~16 DP ranks it exceeds even a full cold recompute. Load is the objective; hit rate is the tiebreak. This is backwards from every cache-aware router we know of.
5. **Do not build a replication mechanism.** The coverage function gives you the right replication for free.
6. **Size the prefill pool at ~1:3.** Even with a perfect cache, the incremental prefill of each turn's delta against a 100k context is irreducible — and 65% of it is attention *against the cached prefix*. Prefix caching makes prefill cheap, not free.

---

## 9. Limitations — read these before believing anything above

- **These are simulation results from a simulator we wrote.** The hardware constants are first-principles and the sanity anchors (19.4 ms step, 51 tok/s/stream, 1:3 prefill:decode ratio) are plausible, but nothing here has touched a GPU. Nie et al. validated on real A100s within ~10%; we have not. Chen et al. also simulate, so this is par for that sub-literature — which is itself a criticism of the sub-literature.
- **W = 10, the fabric contention weight, is a judgment call and it is load-bearing.** The entire "keep KV off the fabric" conclusion, and much of the case for node-locality over migration, rests on it. If your fabric has headroom (W ≈ 1), cross-node migration becomes 10× cheaper and the calculus shifts toward free migration. **This is the number to measure first.**
- **T_local is modeled as exactly linear in resident KV.** Real kernels have fixed overheads and non-ideal scaling; the linear model is the same one Chen et al. use, and it flatters the barrier story.
- **The workload is synthetic.** Zipf(1.1) repo popularity, log-normal think times, and the specific tree shape are assumptions. The qualitative conclusions (T2, T5) are robust to these; the routing deltas (T3) are not obviously so.
- **KV quantization is not modeled.** Compressing KV to int4 for the offload tiers would change *c* by 4× and shift every bandwidth number in the paper. It probably makes offload look even better.
- **No speculative decoding, no chunked-prefill/decode overlap, no disaggregated attention.** Each would perturb α or the barrier structure.
- **The min-max submodular allocation problem is stated, not solved.** We give the greedy marginal rule and no approximation guarantee. That is the honest gap and the obvious next paper.

---

## 10. Relation to prior work

**Nie, Si & Zhou** give us the capacity functional and the Lyapunov machinery; we show their policy-independence result is the κ ≡ 1 boundary case of a routing-dependent capacity, and that their multi-GPU validation (independent replicas, round-robin) is precisely the topology in which their result is clean and the barrier is absent.

**Chen et al.** give us the barrier and the sticky-assignment structure; we show that their load term is the wrong functional under prefix reuse — it should be a coverage function, not a sum — and that with a cascade kernel the memory constraint bounds their imbalance for free, which is why our measured barrier idle (2–4%) is an order of magnitude below their reported 40%. Their 40% is a *no-sharing, no-cascade* number. **If that is right, the single highest-value intervention on their trace is not a better router. It is a different attention kernel.** We would want to see that ablation.

**Cache-aware routers** (SGLang's router, Preble, Mooncake's scheduler) do prefill-side prefix affinity with a load guardrail. Our T3 says the guardrail is the objective and the affinity is the guardrail. We are not aware of prior work that prices the barrier cost of a placement against the cache cost in a common currency, or that identifies the cascade kernel as a precondition rather than an optimization.

The joint decision variable — **(prefill node, decode node) chosen together**, with the transfer between them priced against the collective it contends with — is, as far as we can tell, unaddressed. That is the routing problem the two papers, read together, imply and neither poses.

---

## Appendix: the one-line versions

- Load is a **union**, not a sum. Everything follows.
- **Cascade or nothing.** Sharing without a cascade kernel is throughput-neutral by construction.
- **Moving KV is 81× cheaper than making it** — and the ratio grows with model size.
- **The barrier is recurring and global; the cache miss is one-time and local.** Above ~16 DP ranks, take the cold recompute.
- **Don't build a replication mechanism.** We built one. It was worse.

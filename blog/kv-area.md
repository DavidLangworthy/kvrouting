# A KV cache is an area, not a size

*Two recent papers ask the same question — how do you use the KV cache well when memory is the
binding constraint? — and answer it in two languages that look unrelated. One, from Nie, Si &
Zhou (ICML 2026), is queueing theory. The other, from Chen, Bu, Song, Lu, Ye & Zhou (an arXiv
preprint), is load balancing. They share an author, and beneath the two formalisms they are
describing a single object. This is what that object is.*

---

## The object: a request's footprint over time

Take one decode request. At each step it appends one token to its KV cache.[^spec] So its
**footprint** — the bytes of KV it occupies — grows by one token per step, from the prompt length
up to prompt-plus-output:

**footprint at step t = prefix length + t**

$$W_i(t) = s + t \qquad t = 0, \dots, o$$

Plot footprint against time and you get a trapezoid: it starts at the prompt size, ramps up as
the model generates, and vanishes when the request finishes. **That trapezoid is the whole
story.** Each paper is one way of collapsing it to a single number.

## Reading one: integrate over time → a capacity ceiling

First question: how many requests can one GPU worker sustain? A worker has a fixed KV budget —
the HBM left over after weights and workspace, call it the **budget**, perhaps 500 GB.

The mistake is to think a request's cost is its *size* (peak footprint) or its *speed*
(tokens/sec). It is neither. The cost is the **area under the trapezoid** — how much memory it
ties up, integrated over how long it ties it up:

**footprint-area ≈ output length × (prefix length + output length / 2)**

$$g(s,o) \approx o\left(s + \tfrac{o}{2}\right)$$

The units give it away: this is **byte·seconds** — a memory×time area, not bytes and not
bytes-per-second. A request's claim on a memory-bound server is *how much it holds for how long*.
That is the one idea you miss if you size a cache by its peak.

Sum that area over the requests sharing a worker, cap the sum at the budget, and you get a
ceiling on the sustainable request rate:

**sustainable rate = budget ÷ (batch factor × average footprint-area)**

$$\mu = \frac{M}{\bar{b}\,\mathbb{E}[g]}$$

Nie, Si & Zhou make this rigorous — for a **single worker** — and prove something sharp: no
scheduling policy beats this ceiling. Being *work-conserving* (never idle while requests wait) is
all it takes; the order and batching you pick cannot move the wall. That is the whole theorem:
one GPU, scheduling only. It says nothing about routing across GPUs — and it does not need to,
because memory is additive, so spreading requests over more workers just sums the same ceiling.
They confirmed the single-worker result on real A100s to within ~10%. Their eight-GPU test is
plain independent replicas — round-robin, no coordination. Call this reading
**memory-as-capacity**, and hold that last point.

## Reading two: max across workers → a barrier

Now put the same trapezoids on a *coordinated* multi-GPU deployment, and a second cost appears
that the first reading cannot see.

Large models are served with their layers split across GPUs. In the layout now standard for big
mixture-of-experts models — attention run **data-parallel**, experts **expert-parallel** across
the same GPUs (the SGLang / DeepSeek arrangement) — the GPUs **synchronize on every decode
step**: each step, an all-to-all collective shuffles tokens out to their experts and back. That
collective is a barrier, and every rank waits for the slowest.

And the slowest rank is the one with the most resident KV to read — attention each step reads the
whole resident cache, so more resident KV means a slower step:

**step time = per-token read cost × busiest rank's resident KV + collective**

$$T_{\text{step}} = \alpha \, \max_g L_g + T_{\text{sync}}$$

The busiest rank sets the pace; every other rank idles for the difference. Chen, Bu, Song, Lu, Ye
& Zhou measure this in a production trace and find the idle exceeds **40% per step**. At that
level, imbalance is not a tax on throughput — it *is* the throughput problem. Call this reading
**memory-as-barrier**.

## The same trapezoid, twice

Here is the payoff, and the reason to read the two papers together:

> **Integrate** a request's footprint over its lifetime → the area → the capacity ceiling.
> *(Nie et al.)*
>
> Take the **max** of footprints across workers each step → the barrier idle. *(Chen et al.)*

One trapezoid. Collapse the **time** axis and you get capacity; collapse the **worker** axis and
you get the barrier. Two formalisms from two overlapping teams — queueing theory and load
balancing — are two projections of the same object. Nie measures memory *summed over time* on one
worker; Chen measures memory *maxed across workers* at one instant. Same shape, read two ways.

## Where both papers stop

Both model one request at a time: it arrives, runs to completion, releases its KV, and departs.
Nothing in either lets a cache outlive the request that made it, reuse one request's KV for
another, or move a cache from one worker to another. That is not a shortcoming — it is a clean
boundary, and it is where the interesting questions start. A companion essay picks them up.

---

**The two papers**

- Chengyi Nie, Nian Si, Zijie Zhou. *A Queueing-Theoretic Framework for Stability Analysis of LLM
  Inference with KV Cache Memory Constraints.* ICML 2026. [arXiv:2605.04595](https://arxiv.org/abs/2605.04595)
- Zixi Chen, Tianci Bu, Chendong Song, Xin Lu, Yinyu Ye, Zijie Zhou. *A Universal Load Balancing
  Principle and Its Application to Large Language Model Serving.* [arXiv:2601.17855](https://arxiv.org/abs/2601.17855)

[^spec]: Speculative decoding appends a variable-length chunk per step rather than one token; it
    dents the "+1 per step" picture but not the shape of the argument. Set it aside for now.

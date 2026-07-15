This started as a practical question about where long-running agent sessions should keep their KV cache between turns. The answer looked like it ought to be a routing problem: keep a session near its cache, avoid the miss, and let prefix reuse do the work.

The analysis below argues that this framing is backwards for agentic coding workloads. Within a single conversation, cache reuse eliminates prefill work, but it does not reduce decode memory pressure. Once the session is waiting on a human, a test run, or another tool, the useful question becomes where the idle KV should live and how aggressively the decode pool should rebalance before the next turn arrives.

Everything below the line is the generated analysis and simulator write-up.

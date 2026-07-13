"""
First-principles hardware constants: 70B-class dense model, TP=8 H100 node.

Nothing here is measured. Every constant is derived from a spec sheet or an
arithmetic identity, and the derivation is in the comment. Change them here
and every number in the papers moves with them.
"""

# ---------------------------------------------------------------- model
N_PARAMS   = 70e9
N_LAYERS   = 80
D_MODEL    = 8192
N_KV_HEADS = 8          # GQA
D_HEAD     = 128
KV_DTYPE_B = 1          # fp8

# KV bytes per token = 2 (K and V) x layers x kv_heads x head_dim x bytes
C_KV = 2 * N_LAYERS * N_KV_HEADS * D_HEAD * KV_DTYPE_B      # 163,840 B ~ 160 KB

# ---------------------------------------------------------------- node (one TP=8 group = one logical worker)
HBM_BW     = 8 * 3.3e12      # 26.4 TB/s aggregate KV-read bandwidth
M_TOK      = 3_200_000       # KV budget in tokens (~520 GB after weights + workspace)
NODE_FLOPS = 6.3e15          # 8 x H100 fp8 dense, ~40% MFU
T_SYNC     = 2e-3            # EP all-to-all per decode step (s)

# seconds of decode-step time per resident token of KV. THE central constant.
ALPHA = C_KV / HBM_BW                                        # 6.06e-9 s

# prefill = linear (FFN/projections) + attention (each new token attends to all preceding)
TH_LIN = 2 * N_PARAMS / NODE_FLOPS                           # 2.22e-5  s / token
TH_ATT = 4 * D_MODEL * N_LAYERS / NODE_FLOPS                 # 4.16e-10 s / token / token-of-context

# ---------------------------------------------------------------- links (per node)
B_PCIE   = 300e9        # 8 x PCIe5 x16, derated for host DRAM controller bandwidth
B_FABRIC = 400e9        # 8 x ConnectX-7
B_SSD    = 70e9         # NVMe array
DRAM_TOK = 12_500_000   # 2 TB host DRAM / C_KV
SSD_TOK  = 187_500_000  # 30 TB NVMe / C_KV

# ===== THE LOAD-BEARING CONSTANT =====================================
# Cross-node KV traffic rides the same fabric as the EP all-to-all, which sits on
# the barrier's critical path EVERY decode step. So a migration does not merely
# consume bandwidth, it delays the collective that every rank is waiting on.
# We charge it W_FABRIC x its raw cost.
#
# This is a judgment call, not a measurement. Every "migrate rather than stall"
# conclusion in the papers depends on it:
#     W = 1   -> migration is ~110x cheaper than the barrier; conclusion strengthens
#     W = 10  -> ~11x cheaper (what the papers assume)
#     W = 100 -> comparable; the conclusion INVERTS and affinity routing returns
# MEASURE THIS FIRST.
W_FABRIC = 10.0
# =====================================================================


def prefill_s(n_new, ctx_cached=0.0):
    """node-seconds to prefill n_new tokens on top of ctx_cached already-cached tokens.

    Cold prefill of s tokens:  prefill_s(s, 0)          -> O(s^2), attention-dominated
    Warm turn with delta d:    prefill_s(d, s)          -> O(d*s)
    """
    return TH_LIN * n_new + TH_ATT * n_new * (ctx_cached + n_new / 2.0)


def fetch_pcie(tok):    return tok * C_KV / B_PCIE
def fetch_ssd(tok):     return tok * C_KV / B_SSD
def fetch_fabric(tok):  return tok * C_KV / B_FABRIC * W_FABRIC   # charged for collective contention


def barrier_cost(G, o_remaining, excess_tokens):
    """node-seconds of cluster-wide idle from making a worker the straggler by
    `excess_tokens`, for `o_remaining` further decode steps.

    Recurring (every step) and global (all G-1 other ranks wait). This is the term
    the whole paper turns on, and the reason a cache hit is worth less than it looks.
    """
    return (G - 1) * o_remaining * ALPHA * max(0.0, excess_tokens)


if __name__ == "__main__":
    print(f"C_KV      {C_KV/1e3:8.1f} KB/token")
    print(f"ALPHA     {ALPHA*1e9:8.2f} ns per resident token per step")
    print(f"step @ M  {ALPHA*M_TOK*1e3:8.1f} ms   ({1/(ALPHA*M_TOK):.0f} tok/s/stream)")
    print(f"TH_LIN    {TH_LIN:8.2e} s/tok   ({1/TH_LIN:,.0f} tok/s prefill, linear term only)")
    print(f"TH_ATT    {TH_ATT:8.2e} s/tok/tok-ctx")
    f = 107_000
    print(f"\n--- materialising a {f//1000}k-token prefix (node-seconds) ---")
    print(f"  HBM hit          {0.0:8.4f}")
    print(f"  node-local DRAM  {fetch_pcie(f):8.4f}")
    print(f"  node-local NVMe  {fetch_ssd(f):8.4f}")
    print(f"  cross-node       {fetch_fabric(f):8.4f}   (W={W_FABRIC})")
    print(f"  cold recompute   {prefill_s(f):8.4f}")
    print(f"  warm turn (1.5k) {prefill_s(1500, f):8.4f}")
    print(f"\n  moving KV is {prefill_s(f)/fetch_pcie(f):.0f}x cheaper than making it")

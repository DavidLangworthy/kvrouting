"""figures/saturation.{png,html}: TTFT p95 vs decode goodput as offered load rises.

The unified picture of the-think-gap.md. Four routing policies traced from light
to heavy load. Cache affinity and pre-staging keep KV node-local (~0.1 s TTFT) but
affinity saturates early; balance / CB-IO sustain throughput but pay a ~0.7 s
migration tax on every turn. Think-gap pre-staging is the lower-right envelope:
balance's throughput at affinity's latency.

Reads results/load_sweep.json (produced by figures/sweep_load.py). Run from anywhere:
    python3 figures/sweep_load.py     # ~2 min, writes the JSON
    python3 figures/plot_saturation.py
"""
import json, math, pathlib
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "results" / "load_sweep.json"))
OUT_PNG = ROOT / "figures" / "saturation.png"
OUT_HTML = ROOT / "figures" / "saturation.html"

# Plotly quirk: on a LOG axis, ANNOTATION y-coords must be log10(value)
# (shapes / hlines / hrects use raw values). L() converts for annotations.
def L(v): return math.log10(v)

COL = {"aff": "#e0872b", "cbio": "#9aa7b5", "bfio": "#3b6fd4", "prestage": "#0f9d76"}
NAME = {"aff": "Cache affinity", "cbio": "CB-IO (priced)",
        "bfio": "Load balance (BF-IO)", "prestage": "Think-gap pre-staging"}
ORDER = ["cbio", "bfio", "aff", "prestage"]   # prestage drawn last = on top
SLO = 0.5

def xk(p): return [g / 1000 for g in d[p]["goodput"]]
def yy(p): return d[p]["ttft95"]

fig = go.Figure()

# faint "acceptable" band below the interactive SLO
fig.add_hrect(y0=0.045, y1=SLO, fillcolor="#0f9d76", opacity=0.07, line_width=0, layer="below")
fig.add_hline(y=SLO, line=dict(color="#475569", width=1.4, dash="dot"))
fig.add_annotation(x=0.012, y=L(SLO), xref="paper", yref="y",
                   text="interactive SLO · TTFT p95 = 0.5 s", showarrow=False,
                   font=dict(size=11.5, color="#475569"), xanchor="left", yshift=11)
fig.add_annotation(x=28.9, y=L(0.058), xref="x", yref="y", text="acceptable-latency zone",
                   showarrow=False, font=dict(size=10.5, color="#0b7a5c"),
                   xanchor="right", yanchor="bottom", opacity=0.9)

for p in ORDER:
    win = p == "prestage"
    fig.add_trace(go.Scatter(
        x=xk(p), y=yy(p), name=NAME[p], mode="lines+markers",
        line=dict(color=COL[p], width=5.5 if win else 2.6, shape="spline", smoothing=0.5),
        marker=dict(size=11 if win else 7, color=COL[p], line=dict(width=1.5, color="white"),
                    symbol="star" if win else "circle"),
        opacity=1.0 if win else 0.85, zorder=win and 10 or 1,
        hovertemplate=f"<b>{NAME[p]}</b><br>goodput %{{x:.1f}}k tok/s<br>"
                      f"TTFT p95 %{{y:.2f}}s<extra></extra>",
    ))

# --- annotations -------------------------------------------------------------
# money comparison: at ~24.2k tok/s, balance = 0.71 s but pre-staging = 0.12 s
xc = 24.2
yb, yp = d["bfio"]["ttft95"][2], d["prestage"]["ttft95"][2]
fig.add_shape(type="line", x0=xc, x1=xc, y0=yp, y1=yb, xref="x", yref="y",
              line=dict(color="#0f172a", width=1.6))
for yy_ in (yp, yb):     # end caps
    fig.add_shape(type="line", x0=xc-0.22, x1=xc+0.22, y0=yy_, y1=yy_,
                  xref="x", yref="y", line=dict(color="#0f172a", width=1.6))
fig.add_annotation(x=xc - 0.5, y=L(0.285), xref="x", yref="y",
                   text="<b>same throughput,<br>~6× lower TTFT</b>", showarrow=False,
                   font=dict(size=12.5, color="#0f172a"), xanchor="right", align="right",
                   bgcolor="rgba(255,255,255,0.82)")

# balance shelf: migration tax  (open space above the blue/grey shelf)
fig.add_annotation(x=12.8, y=L(1.5), text="<b>Balance pays the migration tax</b><br>"
                   "~0.7 s TTFT on every turn — a cross-node<br>KV fetch on the critical path",
                   xref="x", yref="y", showarrow=False, font=dict(size=12.5, color=COL["bfio"]),
                   xanchor="left", align="left")

# low shelf: affinity + prestage  (in open space above the low shelf)
fig.add_annotation(x=8.4, y=L(0.20), text="<b>Affinity &amp; pre-staging keep KV node-local</b><br>"
                   "~0.1 s TTFT — nothing fetched on the critical path",
                   xref="x", yref="y", showarrow=False, font=dict(size=12.5, color="#0b7a5c"),
                   xanchor="left", align="left")

# prestage headline: the win, pointing at the green knee, placed in open space below-right
fig.add_annotation(x=26.23, y=L(0.481), ax=40, ay=70, xref="x", yref="y",
                   text="<b>Pre-staging: balance's throughput<br>at interactive latency</b>",
                   font=dict(size=13, color=COL["prestage"]), align="center", xanchor="center",
                   arrowcolor=COL["prestage"], arrowwidth=1.8, arrowhead=2,
                   bgcolor="rgba(255,255,255,0.82)")

# affinity saturates early: point at the orange curve where it is dying alone
# (same throughput as the green knee, but ~5x the latency)
fig.add_annotation(x=26.5, y=L(2.6), ax=-52, ay=-26, xref="x", yref="y",
                   text="<b>Affinity saturates early</b><br>barrier idle wastes ~10%",
                   font=dict(size=11.5, color="#b5701f"), align="right", xanchor="right",
                   arrowcolor=COL["aff"], arrowwidth=1.4, arrowhead=2)

fig.update_layout(
    template="plotly_white",
    title=dict(text="<b>Think-gap pre-staging escapes the throughput–latency tradeoff</b>"
               "<br><span style='font-size:14px;color:#64748b'>Time-to-first-token (p95) vs decode "
               "goodput as offered load rises · 16 nodes · within-conversation reuse</span>",
               x=0.5, xanchor="center", font=dict(size=22, color="#0f172a")),
    font=dict(family="Inter, Helvetica, Arial, sans-serif", size=14, color="#0f172a"),
    xaxis=dict(title="<b>Decode goodput</b>  (tokens / s, higher is better)  →",
               showgrid=True, gridcolor="#eef2f6", zeroline=False, ticksuffix="k",
               range=[7, 29.5], dtick=5),
    yaxis=dict(title="↓  <b>TTFT p95</b>  (seconds, log · lower is better)", type="log",
               showgrid=True, gridcolor="#eef2f6", range=[-1.35, 1.15],
               tickvals=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
               ticktext=["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10"]),
    legend=dict(x=0.015, y=0.985, bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0", borderwidth=1, font=dict(size=13),
                itemsizing="constant"),
    width=1040, height=690, margin=dict(l=84, r=56, t=92, b=72),
    plot_bgcolor="white", paper_bgcolor="white",
)

fig.write_html(OUT_HTML, include_plotlyjs="cdn", full_html=True)
fig.write_image(OUT_PNG, scale=2)
print("wrote", OUT_PNG, "and", OUT_HTML)
for p in ORDER:
    xs, ys = d[p]["goodput"], d[p]["ttft95"]
    under = max([x for x, y in zip(xs, ys) if y <= SLO] + [0]) / 1000
    print(f"  {NAME[p]:22s} max goodput under {SLO}s p95 SLO: {under:5.1f}k")

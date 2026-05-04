# Q3: Generate three networks of 1000 nodes each using:
#       - Random Network Model       (Erdos-Renyi)
#       - Small World Network Model  (Watts-Strogatz)
#       - Preferential Attachment    (Barabasi-Albert)
#     and compare their characteristics.
#
# No CSV needed — all three networks are generated programmatically.

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

N = 1000
G_er = nx.erdos_renyi_graph(N, 0.006, seed=42)
G_ws = nx.watts_strogatz_graph(N, 6, 0.1, seed=42)
G_ba = nx.barabasi_albert_graph(N, 3, seed=42)

models  = [("Erdos-Renyi\n(Random)",              G_er, 'steelblue'),
           ("Watts-Strogatz\n(Small World)",       G_ws, 'tomato'),
           ("Barabasi-Albert\n(Pref. Attachment)", G_ba, 'seagreen')]
names   = ["ER", "WS", "BA"]
colors  = ['steelblue', 'tomato', 'seagreen']

# ── Compute characteristics ──────────────────────────────────────────────────
props = {}
for (label, G, _), name in zip(models, names):
    deg = [d for _, d in G.degree()]
    lcc = G.subgraph(max(nx.connected_components(G), key=len))
    props[name] = {
        "Mean Degree"     : np.mean(deg),
        "Degree Std"      : np.std(deg),
        "Avg Clustering"  : nx.average_clustering(G),
        "Avg Path Length" : nx.average_shortest_path_length(lcc),
        "Transitivity"    : nx.transitivity(G),
        "Density"         : nx.density(G),
    }

# ── Figure 1: Degree distribution histograms ────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, G, color) in zip(axes, models):
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, bins=30, color=color, edgecolor='black', alpha=0.85)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.axvline(np.mean(degrees), color='black', linestyle='--',
               label=f"mean={np.mean(degrees):.1f}")
    ax.legend(fontsize=9)

fig1.suptitle("Q3: Degree Distributions — Random vs Small World vs Pref. Attachment",
              fontsize=13)
fig1.tight_layout()
fig1.savefig("q3_output.png", dpi=150, bbox_inches='tight')
fig1.show()

# ── Figure 2: Characteristics comparison bar chart ──────────────────────────
char_keys = ["Mean Degree", "Degree Std", "Avg Clustering",
             "Avg Path Length", "Transitivity"]

fig2, axes2 = plt.subplots(1, len(char_keys), figsize=(18, 5))
for ax, key in zip(axes2, char_keys):
    vals = [props[n][key] for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{val:.3f}", ha='center', fontsize=8)
    ax.set_title(key, fontsize=10)
    ax.set_ylabel("Value")

fig2.suptitle("Q3: Network Characteristics Comparison (N=1000)", fontsize=13)
fig2.tight_layout()
fig2.savefig("q3_characteristics.png", dpi=150, bbox_inches='tight')
fig2.show()

# ── Console table ────────────────────────────────────────────────────────────
print(f"\n{'Property':<20} {'ER':>10} {'WS':>12} {'BA':>12}")
print("-" * 56)
for key in ["Mean Degree", "Degree Std", "Avg Clustering",
            "Avg Path Length", "Transitivity", "Density"]:
    row = f"  {key:<18}"
    for n in names:
        row += f" {props[n][key]:>12.4f}"
    print(row)

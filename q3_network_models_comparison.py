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

models = [("Erdos-Renyi (Random)",              G_er, 'steelblue'),
          ("Watts-Strogatz (Small World)",       G_ws, 'tomato'),
          ("Barabasi-Albert (Pref. Attachment)", G_ba, 'seagreen')]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, G, color) in zip(axes, models):
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, bins=30, color=color, edgecolor='black', alpha=0.85)
    ax.set_title(name)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")

plt.suptitle("Q3: Network Model Comparison — Degree Distributions", fontsize=13)
plt.tight_layout()
plt.savefig("q3_output.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'Property':<22} {'ER':>10} {'WS':>12} {'BA':>12}")
print("-" * 58)
props = {}
for G, name in zip([G_er, G_ws, G_ba], ["ER", "WS", "BA"]):
    deg = [d for _, d in G.degree()]
    lcc = G.subgraph(max(nx.connected_components(G), key=len))
    props[name] = {
        "Mean Degree"    : np.mean(deg),
        "Std Degree"     : np.std(deg),
        "Avg Clustering" : nx.average_clustering(G),
        "Avg Path Len"   : nx.average_shortest_path_length(lcc),
        "Transitivity"   : nx.transitivity(G),
        "Density"        : nx.density(G),
    }

for key in ["Mean Degree", "Std Degree", "Avg Clustering", "Avg Path Len", "Transitivity", "Density"]:
    row = f"  {key:<20}"
    for name in ["ER", "WS", "BA"]:
        row += f" {props[name][key]:>12.4f}"
    print(row)

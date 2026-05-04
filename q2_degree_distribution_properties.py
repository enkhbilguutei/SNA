# Q2: Compute and plot degree distribution of a real-world network.
#     Also compute its local and global properties.
#
# DATASET: Put your CSV file path below.
#          Your CSV must have 3 columns: Source, Target, Weight
#          Example row:  1,2,5

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = "your_file.csv"          # <-- change this to your CSV file path

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip().str.lower()

# Directed + weighted graph
G = nx.from_pandas_edgelist(df, source="source", target="target",
                             edge_attr="weight",
                             create_using=nx.DiGraph)

# ── Degree distributions: in-degree and out-degree (correct for directed) ────
in_deg  = [d for _, d in G.in_degree()]
out_deg = [d for _, d in G.out_degree()]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(*np.unique(in_deg,  return_counts=True), color='steelblue', edgecolor='black', alpha=0.85)
axes[0].set_title("In-Degree Distribution")
axes[0].set_xlabel("In-Degree")
axes[0].set_ylabel("Count")

axes[1].bar(*np.unique(out_deg, return_counts=True), color='tomato',    edgecolor='black', alpha=0.85)
axes[1].set_title("Out-Degree Distribution")
axes[1].set_xlabel("Out-Degree")
axes[1].set_ylabel("Count")

# Log-log on in-degree
k, pk = np.unique(in_deg, return_counts=True)
if len(k) > 1:
    axes[2].loglog(k, pk, 'o-', color='seagreen')
    axes[2].set_title("In-Degree Distribution (Log-Log)")
    axes[2].set_xlabel("In-Degree (log)")
    axes[2].set_ylabel("Count (log)")

plt.suptitle("Q2: Directed Weighted Network — Degree Distributions", fontsize=13)
plt.tight_layout()
plt.savefig("q2_output.png", dpi=150, bbox_inches='tight')
plt.show()

# ── Global properties ────────────────────────────────────────────────────────
print("\n--- Global Properties ---")
print(f"Nodes              : {G.number_of_nodes()}")
print(f"Directed Edges     : {G.number_of_edges()}")
print(f"Density            : {nx.density(G):.4f}")
print(f"Is Weighted        : {nx.is_weighted(G)}")

# Diameter & avg path: only safe on largest STRONGLY connected component
scc       = max(nx.strongly_connected_components(G), key=len)
G_scc     = G.subgraph(scc)
scc_frac  = len(scc) / G.number_of_nodes()
print(f"Largest SCC size   : {len(scc)} nodes  ({scc_frac*100:.1f}% of graph)")
if len(scc) > 1:
    print(f"Diameter (SCC)     : {nx.diameter(G_scc)}")
    print(f"Avg Shortest Path  : {nx.average_shortest_path_length(G_scc):.4f}")
else:
    print(f"Diameter (SCC)     : N/A (SCC too small)")
    print(f"Avg Shortest Path  : N/A")

# Reciprocity: fraction of edges that are mutual (directed-specific)
print(f"Reciprocity        : {nx.reciprocity(G):.4f}")

# Transitivity on the underlying undirected graph
G_und = G.to_undirected()
print(f"Transitivity       : {nx.transitivity(G_und):.4f}")

# Assortativity: use try/except — can fail if variance = 0
try:
    assort = nx.degree_assortativity_coefficient(G)
    print(f"Assortativity      : {assort:.4f}")
except Exception:
    print(f"Assortativity      : N/A (undefined for this graph)")

# ── Local properties ─────────────────────────────────────────────────────────
print("\n--- Local Properties (per node) ---")

# Weight-aware betweenness and closeness
betw  = nx.betweenness_centrality(G, weight='weight')
close = nx.closeness_centrality(G)

# Clustering for directed graph (uses nx.clustering with DiGraph)
clust = nx.clustering(G.to_undirected())

print(f"{'Node':<8} {'In-Deg':<9} {'Out-Deg':<10} {'Clustering':<12} {'Betweenness':<13} {'Closeness'}")
print("-" * 65)
for n in G.nodes():
    print(f"{n:<8} {G.in_degree(n):<9} {G.out_degree(n):<10} "
          f"{clust[n]:<12.4f} {betw[n]:<13.4f} {close[n]:.4f}")
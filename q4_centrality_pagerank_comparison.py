# Q4: Compute different centrality measures to identify top-N nodes
#     and compare their ranks with those obtained by PageRank method.
#
# DATASET: Put your CSV file path below.
#          Your CSV must have at least 2 columns: Source, Target
#          Example row:  1,2

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

CSV_FILE = "your_file.csv"          # <-- change this to your CSV file path

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()
G = nx.from_pandas_edgelist(df, source="Source", target="Target")

N = 10   # top-N nodes to compare

degree_c  = nx.degree_centrality(G)
between_c = nx.betweenness_centrality(G)
close_c   = nx.closeness_centrality(G)
eigen_c   = nx.eigenvector_centrality(G)
pagerank  = nx.pagerank(G)

measures = {
    "Degree":      degree_c,
    "Betweenness": between_c,
    "Closeness":   close_c,
    "Eigenvector": eigen_c,
    "PageRank":    pagerank,
}

# Build full rank dictionaries for every node (rank 1 = most central)
def rank_dict(d):
    sorted_nodes = sorted(d, key=d.get, reverse=True)
    return {node: rank+1 for rank, node in enumerate(sorted_nodes)}

ranks = {name: rank_dict(measure) for name, measure in measures.items()}

# Top-N nodes according to PageRank
top_by_pagerank = sorted(pagerank, key=pagerank.get, reverse=True)[:N]

# ── Figure 1: Bar charts — top-N nodes by each measure ──────────────────────
colors = ['steelblue', 'tomato', 'seagreen', 'orange', 'orchid']
fig1, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, (name, measure), color in zip(axes, measures.items(), colors):
    top_nodes = sorted(measure, key=measure.get, reverse=True)[:N]
    scores    = [measure[n] for n in top_nodes]
    ax.bar([str(n) for n in top_nodes], scores, color=color,
           edgecolor='black', alpha=0.85)
    ax.set_title(f"Top-{N}: {name}", fontsize=10)
    ax.set_xlabel("Node")
    ax.set_ylabel("Score")
    ax.tick_params(axis='x', labelsize=8)

fig1.suptitle("Q4: Top-10 Nodes by Each Centrality Measure", fontsize=13)
fig1.tight_layout()
fig1.savefig("q4_output.png", dpi=150, bbox_inches='tight')
fig1.show()

# ── Figure 2: Rank comparison heatmap ───────────────────────────────────────
# Rows = top-N nodes by PageRank, Columns = each centrality measure
# Cell value = rank of that node in that measure
rank_matrix = np.array([
    [ranks[m][node] for m in measures]
    for node in top_by_pagerank
])

fig2, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(rank_matrix, cmap='YlOrRd_r', aspect='auto')

ax.set_xticks(range(len(measures)))
ax.set_xticklabels(list(measures.keys()), fontsize=11)
ax.set_yticks(range(len(top_by_pagerank)))
ax.set_yticklabels([f"Node {n}  (PR rank {i+1})" for i, n in enumerate(top_by_pagerank)],
                   fontsize=9)

for i in range(len(top_by_pagerank)):
    for j in range(len(measures)):
        ax.text(j, i, str(rank_matrix[i, j]), ha='center', va='center',
                fontsize=9, color='black' if rank_matrix[i, j] < rank_matrix.max()*0.6 else 'white')

plt.colorbar(im, ax=ax, label="Rank (1 = most central)")
ax.set_title(f"Q4: Rank Comparison — Top-{N} PageRank Nodes across All Centrality Measures\n"
             f"(Darker = higher ranked)", fontsize=11)
fig2.tight_layout()
fig2.savefig("q4_rank_comparison.png", dpi=150, bbox_inches='tight')
fig2.show()

# ── Spearman correlations with PageRank ──────────────────────────────────────
all_nodes = list(G.nodes())
pr_scores = [pagerank[n] for n in all_nodes]

print("\nSpearman Correlation with PageRank:")
print("-" * 40)
for name, measure in measures.items():
    if name == "PageRank":
        continue
    corr, pval = spearmanr([measure[n] for n in all_nodes], pr_scores)
    print(f"  {name:<14}: r = {corr:.4f}  (p = {pval:.4f})")

print(f"\nTop-{N} nodes by PageRank: {top_by_pagerank}")

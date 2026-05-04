# Q4: Compute different centrality measures to identify top-N nodes
#     and compare their ranks with those obtained by PageRank method.
#
# DATASET: Put your CSV file path below.
#          Your CSV must have at least 2 columns: Source, Target
#          Example row:  1,2

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

CSV_FILE = "your_file.csv"          # <-- change this to your CSV file path

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()
G = nx.from_pandas_edgelist(df, source="Source", target="Target")

N = 10

degree_c  = nx.degree_centrality(G)
between_c = nx.betweenness_centrality(G)
close_c   = nx.closeness_centrality(G)
eigen_c   = nx.eigenvector_centrality(G)
pagerank  = nx.pagerank(G)

def top_n(d, n=N):
    return sorted(d, key=d.get, reverse=True)[:n]

measures = {
    "Degree":      degree_c,
    "Betweenness": between_c,
    "Closeness":   close_c,
    "Eigenvector": eigen_c,
    "PageRank":    pagerank,
}
colors = ['steelblue', 'tomato', 'seagreen', 'orange', 'orchid']

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, (name, measure), color in zip(axes, measures.items(), colors):
    nodes  = top_n(measure)
    scores = [measure[n] for n in nodes]
    ax.bar([str(n) for n in nodes], scores, color=color, edgecolor='black', alpha=0.85)
    ax.set_title(f"Top-{N}: {name}")
    ax.set_xlabel("Node")
    ax.set_ylabel("Score")
    ax.tick_params(axis='x', labelsize=8)

plt.suptitle("Q4: Centrality Measures — Top-10 Nodes", fontsize=13)
plt.tight_layout()
plt.savefig("q4_output.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'Rank':<6} {'Degree':<10} {'Betweenness':<13} {'Closeness':<12} {'Eigenvector':<13} {'PageRank'}")
print("-" * 65)
for i, (d, b, c, e, p) in enumerate(zip(top_n(degree_c), top_n(between_c),
                                         top_n(close_c), top_n(eigen_c), top_n(pagerank)), 1):
    print(f"{i:<6} {d:<10} {b:<13} {c:<12} {e:<13} {p}")

nodes = list(G.nodes())
corr, pval = spearmanr([degree_c[n] for n in nodes], [pagerank[n] for n in nodes])
print(f"\nSpearman Correlation (Degree vs PageRank): {corr:.4f}  (p = {pval:.4f})")

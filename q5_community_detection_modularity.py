# Q5: Apply community detection algorithms on a small real-world network
#     (Karate Club) and compare modularity using bar plot.
#     Also plot the communities revealed with different colors.
#
# DATASET: Karate Club is used by default (as specified in the question).
#          To use your own CSV instead, comment out the karate line and uncomment:
#   import pandas as pd
#   df = pd.read_csv("your_file.csv")
#   df.columns = df.columns.str.strip()
#   G = nx.from_pandas_edgelist(df, source="Source", target="Target")

import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()          # Zachary's Karate Club (34 nodes, 78 edges)

c_louvain = nx.community.louvain_communities(G, seed=42)
c_greedy  = nx.community.greedy_modularity_communities(G)
c_label   = nx.community.label_propagation_communities(G)
c_girvan  = list(next(nx.community.girvan_newman(G)))

algorithms = {
    "Louvain":           c_louvain,
    "Greedy Modularity": c_greedy,
    "Label Propagation": c_label,
    "Girvan-Newman":     c_girvan,
}

modularities = {name: nx.community.modularity(G, comms)
                for name, comms in algorithms.items()}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ── Bar plot: modularity comparison ─────────────────────────────────────────
names  = list(modularities.keys())
scores = list(modularities.values())
bars   = axes[0].bar(names, scores,
                     color=['steelblue', 'tomato', 'seagreen', 'orchid'],
                     edgecolor='black', alpha=0.85)
for bar, score in zip(bars, scores):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{score:.3f}", ha='center', fontsize=11, fontweight='bold')
axes[0].set_title("Modularity Score by Algorithm", fontsize=11)
axes[0].set_ylabel("Modularity")
axes[0].set_ylim(0, max(scores) * 1.25)

# ── Network plot: communities in different colors (best algorithm = Louvain) ─
pos    = nx.spring_layout(G, seed=42)
cmap   = plt.colormaps.get_cmap('tab10')
node_colors = {}
for i, community in enumerate(c_louvain):
    for node in community:
        node_colors[node] = cmap(i)

nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color='gray', alpha=0.5, width=1.2)
for i, community in enumerate(c_louvain):
    nx.draw_networkx_nodes(G, pos, nodelist=list(community), ax=axes[1],
                           node_color=[cmap(i)], node_size=300,
                           label=f"Community {i+1}")
nx.draw_networkx_labels(G, pos, ax=axes[1], font_size=7)
axes[1].legend(loc='upper left', fontsize=8)
axes[1].set_title(f"Louvain: {len(c_louvain)} Communities (highest modularity = {modularities['Louvain']:.3f})",
                  fontsize=10)
axes[1].axis('off')

fig.suptitle("Q5: Community Detection — Karate Club Network", fontsize=13)
fig.tight_layout()
fig.savefig("q5_output.png", dpi=150, bbox_inches='tight')
fig.show()

print("\nModularity Scores:")
print("-" * 35)
for name, score in modularities.items():
    print(f"  {name:<25}: {score:.4f}")
print(f"\nBest algorithm: {max(modularities, key=modularities.get)}")

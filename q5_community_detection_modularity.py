# Q5: Apply community detection algorithms on a small real-world network
#     and compare modularity using bar plot.
#     Also plot the communities revealed with different colors.
#
# DATASET: Put your CSV file path below.
#          Your CSV must have at least 2 columns: Source, Target
#          Example row:  1,2

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

CSV_FILE = "your_file.csv"          # <-- change this to your CSV file path

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()
G = nx.from_pandas_edgelist(df, source="Source", target="Target")

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

names  = list(modularities.keys())
scores = list(modularities.values())
bars   = axes[0].bar(names, scores,
                     color=['steelblue', 'tomato', 'seagreen', 'orchid'],
                     edgecolor='black', alpha=0.85)
for bar, score in zip(bars, scores):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{score:.3f}", ha='center', fontsize=10)
axes[0].set_title("Modularity Score by Algorithm")
axes[0].set_ylabel("Modularity")
axes[0].set_ylim(0, max(scores) * 1.2)

pos    = nx.spring_layout(G, seed=42)
cmap   = plt.colormaps.get_cmap('tab10')
colors = {}
for i, community in enumerate(c_louvain):
    for node in community:
        colors[node] = cmap(i)

nx.draw_networkx(G, pos, ax=axes[1],
                 node_color=[colors[n] for n in G.nodes()],
                 node_size=250, edge_color='gray',
                 font_size=7, alpha=0.9)
axes[1].set_title(f"Louvain Communities ({len(c_louvain)} detected)")
axes[1].axis('off')

plt.suptitle("Q5: Community Detection", fontsize=13)
plt.tight_layout()
plt.savefig("q5_output.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nModularity Scores:")
for name, score in modularities.items():
    print(f"  {name:<25}: {score:.4f}")

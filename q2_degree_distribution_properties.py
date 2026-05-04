# Q2: Compute and plot degree distribution of a real-world network.
#     Also compute its local and global properties.
#
# DATASET: Put your CSV file path below.
#          Your CSV must have at least 2 columns: Source, Target
#          A Weight column is optional.
#          Example row:  1,2

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = "your_file.csv"          # <-- change this to your CSV file path

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()
G = nx.from_pandas_edgelist(df, source="Source", target="Target")

degrees = [d for _, d in G.degree()]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(*np.unique(degrees, return_counts=True), color='steelblue', edgecolor='black')
axes[0].set_xlabel("Degree")
axes[0].set_ylabel("Count")
axes[0].set_title("Degree Distribution (Linear)")

k, pk = np.unique(degrees, return_counts=True)
axes[1].loglog(k, pk, 'o-', color='tomato')
axes[1].set_xlabel("Degree (log)")
axes[1].set_ylabel("Count (log)")
axes[1].set_title("Degree Distribution (Log-Log)")

plt.suptitle("Q2: Degree Distribution", fontsize=13)
plt.tight_layout()
plt.savefig("q2_output.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- Global Properties ---")
print(f"Nodes:               {G.number_of_nodes()}")
print(f"Edges:               {G.number_of_edges()}")
print(f"Density:             {nx.density(G):.4f}")
print(f"Diameter:            {nx.diameter(G)}")
print(f"Avg Shortest Path:   {nx.average_shortest_path_length(G):.4f}")
print(f"Avg Clustering:      {nx.average_clustering(G):.4f}")
print(f"Transitivity:        {nx.transitivity(G):.4f}")
print(f"Assortativity:       {nx.degree_assortativity_coefficient(G):.4f}")

print("\n--- Local Properties (per node) ---")
print(f"{'Node':<6} {'Degree':<8} {'Clustering':<12} {'Betweenness':<12} {'Closeness'}")
betw  = nx.betweenness_centrality(G)
clust = nx.clustering(G)
close = nx.closeness_centrality(G)
for n in G.nodes():
    print(f"{n:<6} {G.degree(n):<8} {clust[n]:<12.4f} {betw[n]:<12.4f} {close[n]:.4f}")

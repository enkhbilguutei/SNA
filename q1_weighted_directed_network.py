# Q1: Plot a weighted directed network such that node size and edge width
#     is proportional to their degree and edge weight respectively.
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

# Build a WEIGHTED DIRECTED graph
G = nx.from_pandas_edgelist(df, source="source", target="target",
                             edge_attr="weight",
                             create_using=nx.DiGraph)

pos = nx.spring_layout(G, seed=42)

# Node size ∝ total degree (in-degree + out-degree) in a directed graph
total_degree = dict(G.degree())
max_deg      = max(total_degree.values()) if total_degree else 1
node_sizes   = [100 + (total_degree[n] / max_deg) * 1200 for n in G.nodes()]

# Edge width ∝ weight
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_w        = max(edge_weights) if edge_weights else 1
edge_widths  = [0.5 + (w / max_w) * 5 for w in edge_weights]

plt.figure(figsize=(12, 9))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='steelblue', alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=7, font_color='white')
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray',
                       arrows=True, arrowsize=15, alpha=0.7,
                       connectionstyle='arc3,rad=0.1',
                       min_source_margin=10, min_target_margin=10)

plt.title("Q1: Weighted Directed Network\nNode size ∝ Degree  |  Edge width ∝ Weight")
plt.axis('off')
plt.tight_layout()
plt.savefig("q1_output.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Nodes: {G.number_of_nodes()}, Directed Edges: {G.number_of_edges()}")
print(f"Is directed: {G.is_directed()}, Is weighted: {nx.is_weighted(G)}")
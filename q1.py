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
df.columns = df.columns.str.strip()                        # clean column names
G = nx.from_pandas_edgelist(df, source="Source",
                             target="Target",
                             edge_attr="Weight",
                             create_using=nx.DiGraph())

pos = nx.spring_layout(G, seed=42)
degrees = dict(G.degree())
node_sizes = [100 + degrees[n] * 120 for n in G.nodes()]
edge_weights = [G[u][v]['Weight'] for u, v in G.edges()]
edge_widths = [w * 0.25 for w in edge_weights]

plt.figure(figsize=(12, 9))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='steelblue', alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=7, font_color='white')
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray',
                       arrows=True, arrowsize=12, alpha=0.7,
                       connectionstyle='arc3,rad=0.1')
plt.title("Q1: Weighted Directed Network\nNode size ∝ Degree  |  Edge width ∝ Weight")
plt.axis('off')
plt.tight_layout()
plt.savefig("q1_output.png", dpi=150, bbox_inches='tight')
plt.show()

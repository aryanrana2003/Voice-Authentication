import networkx as nx
import matplotlib.pyplot as plt

# Create the corrected Cartesian product of C_4 and K_3
G_corrected = nx.Graph()

# Add nodes and edges for the Cartesian product of C_4 and K_3
for i in range(4):  # C_4 has 4 vertices
    for j in range(3):  # K_3 has 3 vertices
        G_corrected.add_node((i, j))
        # Connect within the K_3 part
        for k in range(j + 1, 3):
            G_corrected.add_edge((i, j), (i, k))
        # Connect to the next and previous vertices in the cycle C_4 (modulo 4)
        G_corrected.add_edge((i, j), ((i + 1) % 4, j))
        G_corrected.add_edge((i, j), ((i - 1) % 4, j))

# Draw the corrected graph
pos = nx.spring_layout(G_corrected, seed=42)
plt.figure(figsize=(12, 12))
nx.draw(G_corrected, pos, with_labels=True, node_size=700, node_color='lightblue', font_weight='bold', edge_color='gray')
plt.title('Corrected Graph of C4 x K3')
plt.show()

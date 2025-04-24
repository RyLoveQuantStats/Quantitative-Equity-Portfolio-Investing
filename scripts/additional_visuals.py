
## Flow Chart for Portfolio Construction Process

import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes for each key step with descriptive labels
G.add_node("A", label="Nasdaq-100 Constituents")
G.add_node("B", label="Factor Screening\n(Beta_RMW & Expected Return)")
G.add_node("C", label="Portfolio Optimization\n(Monte Carlo Simulation)")
G.add_node("D", label="Covered Calls Overlay")
G.add_node("E", label="Dynamic Collar Hedging\n(High Volatility)")

# Add edges to represent the process flow
G.add_edge("A", "B")
G.add_edge("B", "C")
G.add_edge("C", "D")
G.add_edge("D", "E")

# Define fixed positions for each node 
pos = {
    "A": (0, 0),
    "B": (4, 0),
    "C": (8, 0),
    "D": (12, 0),
    "E": (16, 0)
}

# Create a figure for the flowchart
plt.figure(figsize=(20, 4))  # Adjusted figsize to accommodate extra spacing
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_nodes(G, pos, node_color='#A0CBE2', node_size=30000)
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=40, edge_color='gray', width=2)

# Remove axis and display the plot
plt.axis('off')
plt.title("Portfolio Construction Process Flowchart", fontsize=14)
plt.savefig("flowchart.png", format="png", bbox_inches="tight")

import sqlite3
import json
import matplotlib.pyplot as plt

# Connect to the SQLite database (adjust the path if needed)
conn = sqlite3.connect('database/data.db')
cursor = conn.cursor()

# Query a random record from the 'optimized_hybrid_portfolios' table
query = "SELECT analysis_date, weights FROM optimized_hybrid_portfolios ORDER BY RANDOM() LIMIT 1"
cursor.execute(query)
result = cursor.fetchone()

if result:
    analysis_date, weights_json = result
    print(f"Random Portfolio Date: {analysis_date}")

    # Parse the JSON string from the weights column into a Python dictionary
    portfolio_weights = json.loads(weights_json)

    # Extract ticker labels and their corresponding weights
    labels = list(portfolio_weights.keys())
    sizes = list(portfolio_weights.values())

    # Create a pie chart to visualize the portfolio allocation
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f"Portfolio Weights on {analysis_date}")
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.savefig("portfolio_weights.png", format="png", bbox_inches="tight")
else:
    print("No portfolio data found.")

conn.close()
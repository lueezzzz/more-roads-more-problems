# OD Demand is distributed by creating a virtual "Centroid" node for each zone and connecting it to major physical nodes in that zone with high-capacity, low-travel-time links. 

import pandas as pd
import networkx as nx
import math
import scipy.optimize
import numpy as np
import time

nodes_df = pd.read_csv('intersections.csv')
edges_df = pd.read_csv('roads_updated.csv')
edges_df['oneway'] = edges_df['oneway'].astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 't'])
zones_df = pd.read_csv('zones.csv')

od_df = pd.read_excel('od matrix.xlsx', sheet_name='OD Matrix')
od_df.columns = od_df.columns.astype(str) 

# Create directed graph and add nodes with attributes
G = nx.DiGraph()
for _, row in nodes_df.iterrows():
    G.add_node(str(row['id']).strip(), name=row['name'])

# Add edges with attributes (including capacity based on lane count)
CAP_PER_LANE = 1100
for _, row in edges_df.iterrows():
    u, v = str(row['A']).strip(), str(row['B']).strip()
    if u not in G.nodes or v not in G.nodes: continue
    
    # Convert fft in hours
    dist_km = float(row['distance']) / 1000.0
    speed = float(row.get('speed', 40.0))
    fft = dist_km / speed 
    
    base_name = str(row.get('name', '')).strip()
    link_attrs = {
        'fft': fft, 
        'flow': 0.0, 
        'weight': fft, 
        'name': base_name, 
        'alpha': 0.15, 
        'beta': 4.0
    }
    
    cap = int(row['lanes']) * CAP_PER_LANE
    
    if row['oneway']:
        G.add_edge(u, v, capacity=cap, **link_attrs)
    else:
        # Forward direction
        G.add_edge(u, v, capacity=cap, **link_attrs)
        
        # Reverse direction
        rev_attrs = link_attrs.copy()
        rev_attrs['name'] = f"{base_name} (Rev)"
        G.add_edge(v, u, capacity=cap, **rev_attrs)

# Map zones and parse OD demand (Internal Zones only)
zone_to_nodes = {str(i): [] for i in range(1, 16)}
for _, row in zones_df.iterrows():
    node_id, zone_id = str(row['id']).strip(), str(row['zone']).strip()
    if zone_id in zone_to_nodes and node_id in G.nodes:
        zone_to_nodes[zone_id].append(node_id)

CONNECTOR_TIME = 0.01  # 36 seconds to reach the main road network
CONNECTOR_CAP = 999999 # Infinite capacity to prevent artificial queuing

# Define strategic gates for each zone 
strategic_gates = {
    "1": ["Proper - 6", "Proper - 11", "Proper - 7"],
    "2": ["Proper - 18", "Proper - 19", "Proper - 21"],
    "3": ["Proper - 1", "Molo - 3", "Proper - 2"],
    "4": ["villa - 2", "villa - 3", "villa - 6"],
    "5": ["mandurriao - 8"],
    "6": ["mandurriao - 1", "mandurriao - 7"],
    "7": ["mandurriao - 2", "mandurriao - 3", "mandurriao - 4"],
    "8": ["mandurriao - 12", "mandurriao - 9", "mandurriao - 10"],
    "9": ["lapuz - 4", "lapuz - 5", "lapaz - 9"],
    "10": ["lapuz - 7", "lapuz - 8"],
    "11": ["jaro - 19"],
    "12": ["jaro - 6", "jaro - 7", "jaro - 8"],
    "13": ["jaro - 9"],
    "14": ["jaro - 15", "jaro - 11", "jaro - 12"],
    "15": ["jaro - 21", "jaro - 22", "jaro - 20"]
}

# Create the virtual nodes for each zone (SPLIT INTO ORIGIN AND DESTINATION)
for zone_id, node_list in strategic_gates.items():
    orig_id = f"Origin - {zone_id}"
    dest_id = f"Destination - {zone_id}"
    
    G.add_node(orig_id, name=f"Zone {zone_id} Outbound")
    G.add_node(dest_id, name=f"Zone {zone_id} Inbound")
    
    for physical_node in node_list:
        # Outbound: From Origin -> Main Road
        G.add_edge(orig_id, physical_node, 
                   capacity=CONNECTOR_CAP, 
                   fft=CONNECTOR_TIME, 
                   alpha=0.0, 
                   beta=4.0, 
                   flow=0.0, 
                   weight=CONNECTOR_TIME, 
                   name="Centroid-Connector-Out")
            
        # Inbound: From Main Road -> Destination
        G.add_edge(physical_node, dest_id, 
                   capacity=CONNECTOR_CAP, 
                   fft=CONNECTOR_TIME, 
                   alpha=0.0,
                   beta=4.0, 
                   flow=0.0, 
                   weight=CONNECTOR_TIME, 
                   name="Centroid-Connector-In")

print(f"Centroid setup complete: 30 Virtual Nodes (15 Orig, 15 Dest) added.")

# Prepare routing demands with peak factor and spatial disaggregation
routing_demands = []
PEAK_FACTOR = 0.10

for index, row in od_df.iterrows():
    orig_zone = str(row['ALL']).strip()
    if not orig_zone.isdigit() or int(orig_zone) > 15: 
        continue 
    
    for dest_zone in [str(i) for i in range(1, 16)]:
        raw_volume = row[dest_zone]
        
        # Skip intra-zonal and empty cells
        if orig_zone == dest_zone or pd.isna(raw_volume) or raw_volume <= 0: 
            continue
        
        # Route strictly from Origin node to Destination node
        routing_demands.append({
            'origin': f"Origin - {orig_zone}", 
            'destination': f"Destination - {dest_zone}", 
            'volume': raw_volume * PEAK_FACTOR
        })

print(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
print(f"Processing {len(routing_demands)} active OD routes...")

def bpr_cost(fft, alpha, flow, capacity, beta):
    # BPR function with a safeguard for near-zero capacities to prevent numerical issues
    if capacity < 1e-3: return np.finfo(np.float32).max
    return fft * (1 + alpha * math.pow((flow / capacity), beta))

def update_travel_times(graph):
    # Updates the 'weight' attribute of each edge based on current flow using the BPR function
    for u, v, d in graph.edges(data=True):
        d['weight'] = bpr_cost(d['fft'], d['alpha'], d['flow'], d['capacity'], d['beta'])

def load_aon(graph, demands):
    # Performs an All-or-Nothing assignment based on current edge weights
    x_bar = {edge: 0.0 for edge in graph.edges()} # Auxiliary flow variable for the All-or-Nothing assignment
    SPTT = 0.0 # Shortest Path Travel Time (SPTT)
    dropped_vol = 0.0 # Tracks commuters trapped by cut-edges
    
    for demand in demands:
        # Compute the shortest path for the current demand using Dijkstra's algorithm with edge weights
        try:
            # Find the shortest path based on current weights (travel times)
            path = nx.shortest_path(graph, demand['origin'], demand['destination'], weight='weight')
            # Calculate the total travel time for this path and accumulate it into SPTT
            path_cost = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            SPTT += path_cost * demand['volume']
            
            for i in range(len(path)-1):
            # Increment the auxiliary flow variable for each edge in the path by the demand volume
                x_bar[(path[i], path[i+1])] += demand['volume']
        except nx.NetworkXNoPath:
            # If no path exists, it means this demand is dropped due to a cut-edge. Accumulate the dropped volume.
            dropped_vol += demand['volume']
            
    return SPTT, x_bar, dropped_vol

def find_alpha(graph, x_bar):
    def df(alpha):
        # Computes the derivative of the objective function with respect to alpha for the line search in Frank-Wolfe
        sum_deriv = 0.0
        for u, v, d in graph.edges(data=True):
            tmp_flow = alpha * x_bar.get((u, v), 0.0) + (1 - alpha) * d['flow']
            tmp_cost = bpr_cost(d['fft'], d['alpha'], tmp_flow, d['capacity'], d['beta'])
            sum_deriv += (x_bar.get((u, v), 0.0) - d['flow']) * tmp_cost
        return sum_deriv
    
    # Boundary catchers to prevent SciPy crashes
    val_0 = df(0.0)
    val_1 = df(1.0)
    if val_0 >= 0.0: return 0.0 
    if val_1 <= 0.0: return 1.0 
    
    # Use SciPy's root_scalar to find the alpha that minimizes the objective function along the line defined by x_bar and current flow
    sol = scipy.optimize.root_scalar(df, x0=0.5, bracket=(0.0, 1.0))
    return max(0.0, min(1.0, sol.root))

def solve_equilibrium(graph, demands, accuracy=0.0001, max_iter=50):
    # Runs the Frank-Wolfe algorithm to find the user equilibrium flow pattern for the given graph and demands
    
    # Initialize all flows to zero before starting the iterations
    for u, v, d in graph.edges(data=True): d['flow'] = 0.0
    gap = np.inf
    iteration = 1
    TSTT = 0.0
    

    while gap > accuracy and iteration <= max_iter:
        # Update the travel times based on current flows to reflect congestion effects
        update_travel_times(graph)
        SPTT, x_bar, dropped_vol = load_aon(graph, demands)
        
        # Immediate abort if network is disconnected (cut-edge scenario), as this would invalidate the equilibrium solution
        if dropped_vol > 0: 
            return float('inf'), dropped_vol 
        
        # Perform line search to find the optimal alpha for the convex combination of current flow and auxiliary flow from All-or-Nothing assignment
        alpha = 1.0 if iteration == 1 else find_alpha(graph, x_bar)

        # Update flows on each edge based on the convex combination of current flow and auxiliary flow, using the alpha found from the line search
        for u, v, d in graph.edges(data=True):
            d['flow'] = alpha * x_bar.get((u, v), 0.0) + (1 - alpha) * d['flow']
        
        # Recalculate travel times and total system travel time (TSTT) after flow update to evaluate convergence
        update_travel_times(graph)
        SPTT, _, _ = load_aon(graph, demands)
        TSTT = sum(d['flow'] * d['weight'] for u, v, d in graph.edges(data=True))
        
        # Calculate the relative gap between the current TSTT and the SPTT from the All-or-Nothing assignment to check for convergence
        gap = (TSTT / SPTT) - 1.0 if SPTT > 0 else 0.0
        iteration += 1
    
    return TSTT, 0.0

print("\n--- Calculating Baseline User Equilibrium ---")
# Calculate the baseline TSTT for the original network before any modifications, and measure the time taken for this initial equilibrium calculation
start_time = time.time()
baseline_tstt, _ = solve_equilibrium(G, routing_demands)
print(f"Baseline TSTT: {baseline_tstt:.2f} hours ({baseline_tstt * 60:.1f} minutes)")
print(f"Time taken: {time.time()-start_time:.1f} seconds")

baseline_data = []
for u, v, d in G.edges(data=True):
    # Filter out virtual centroid connectors
    if 'Origin' not in str(u) and 'Destination' not in str(u) and 'Origin' not in str(v) and 'Destination' not in str(v):
        flow = d['flow']
        capacity = d['capacity']
        vc_ratio = flow / max(capacity, 1) # Prevent division by zero
        
        baseline_data.append({
            'From Node': u,
            'To Node': v,
            'Road Name': d.get('name', ''),
            'Baseline Flow': round(flow, 2),
            'Capacity': capacity,
            'V/C Ratio': round(vc_ratio, 3)
        })

df_baseline = pd.DataFrame(baseline_data)
df_baseline.to_csv('Setup 6 Baseline Network Flows.csv', index=False)
print("Saved baseline flows to 'Setup 6 Baseline Network Flows.csv'")

candidate_links = [
    (u, v, d['name'], d.get('flow', 0.0)) 
    for u, v, d in G.edges(data=True) 
    if "Centroid-Connector" not in d.get('name', '')
]

results = []

for i, (u, v, road_name, base_flow) in enumerate(candidate_links):
    # Save and remove FORWARD edge
    edge_data = G[u][v].copy()
    G.remove_edge(u, v)

    # Run the altered network
    new_tstt, dropped_vol = solve_equilibrium(G, routing_demands, max_iter=25, accuracy=0.01) 
    
    # Restore both edges immediately
    G.add_edge(u, v, **edge_data)

    # Analyze Results
    if dropped_vol > 0:
        results.append({
            'Road Name': road_name,
            'From Node': u,
            'To Node': v,
            'Baseline Flow': round(base_flow, 2),
            'Status': 'CRITICAL CUT-EDGE',
            'Baseline TSTT (Hours)': round(baseline_tstt, 2),
            'New TSTT (Hours)': 'DISCONNECTED',
            'Change (Minutes)': f"Dropped {round(dropped_vol)} vol",
            'Braess Link?': 'N/A'
        })
        continue # Move to next loop only after saving

    # Convert time difference to minutes for readability
    time_difference_mins = (new_tstt - baseline_tstt) * 60

    # If the new time is LOWER than the baseline, the network IMPROVED after deletion 
    is_paradox = time_difference_mins < -0.5 
    
    results.append({
        'Road Name': road_name,
        'From Node': u,
        'To Node': v,
        'Baseline Flow': round(base_flow, 2),
        'Status': 'Active' if base_flow > 1.0 else 'Inactive',
        'Baseline TSTT (Hours)': round(baseline_tstt, 2),
        'New TSTT (Hours)': round(new_tstt, 2),
        'Change (Minutes)': round(time_difference_mins, 2),
        'Braess Link?': 'Yes' if is_paradox else 'No'
    })

df_results = pd.DataFrame(results)

# Save to CSV
df_results.to_csv('Setup 6 Results.csv', index=False)
print("\nData successfully saved to 'Setup 6 Results.csv'")

# Print the top 10 valid active findings to console
print("\n--- TOP 10 FINDINGS ---")
# Filter out cut-edges just for the console print
valid_results = df_results[df_results['Status'] == 'Active'].sort_values(by='Change (Minutes)', ascending=True)
print(valid_results.head(10).to_string(index=False))

# # Save new network flows for each Braess link found, and then restore the original network before the next iteration
# braess_only = [res for res in results if res.get('Braess Link?', 'No') == 'Yes' and res.get('Status') == 'Active']

# print(f"Total links tested: {len(results)}")
# print(f"Braess links found: {len(braess_only)}")

# for result in braess_only:
#     u_node = result['From Node']
#     v_node = result['To Node']
#     road_name = result['Road Name']

#     # Save and remove FORWARD edge
#     edge_data_fwd = G[u_node][v_node].copy()
#     G.remove_edge(u_node, v_node)

#     new_tstt, dropped_vol = solve_equilibrium(G, routing_demands, accuracy=0.0001, max_iter=50)

#     # Export the New Network Flows
#     intervention_data = []
#     for u, v, d in G.edges(data=True):
#         # Filter out virtual centroid connectors so the map is clean
#         if 'Origin' not in str(u) and 'Destination' not in str(u) and 'Origin' not in str(v) and 'Destination' not in str(v):
#             flow = d['flow']
#             capacity = d['capacity']
#             vc_ratio = flow / max(capacity, 1) # Prevent division by zero
            
#             intervention_data.append({
#                 'From Node': u,
#                 'To Node': v,
#                 'Road Name': d.get('name', ''),
#                 'New Flow': round(flow, 2),
#                 'Capacity': capacity,
#                 'V/C Ratio': round(vc_ratio, 3)
#             })

#     df_intervention = pd.DataFrame(intervention_data)
#     output_filename = f"Setup 6 {road_name} Flows.csv"
#     df_intervention.to_csv(output_filename, index=False)
#     print(f"\nSUCCESS! Intervention flows saved to '{output_filename}'")

#     G.add_edge(u_node, v_node, **edge_data_fwd)
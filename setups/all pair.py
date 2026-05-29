# This script performs an exhaustive all-pairs audit of the road network to identify potential Braess paradox scenarios.
# It tests both unidirectional and bidirectional closures for each candidate edge and records any cases

import pandas as pd
import networkx as nx
import math
import scipy.optimize
import numpy as np
import os

def bpr_cost(fft, alpha, flow, capacity, beta):
    # UE cost function
    if capacity < 1e-3: return np.finfo(np.float32).max
    return fft * (1 + alpha * math.pow((flow / capacity), beta))

def marginal_cost(fft, alpha, flow, capacity, beta):
    # SO marginal cost function
    if capacity < 1e-3: return np.finfo(np.float32).max
    return fft * (1 + alpha * (beta + 1) * math.pow((flow / capacity), beta))

def update_travel_times(graph, is_so=False):
    # Update edge weights based on current flows
    for u, v, d in graph.edges(data=True):
        if is_so:
            d['weight'] = marginal_cost(d['fft'], d['alpha'], d['flow'], d['capacity'], d['beta'])
        else:
            d['weight'] = bpr_cost(d['fft'], d['alpha'], d['flow'], d['capacity'], d['beta'])

def load_aon(graph, demands):
    # All-or-Nothing assignment to find shortest paths and calculate Shortest Path Travel Time (SPTT)
    x_bar = {edge: 0.0 for edge in graph.edges()} 
    SPTT = 0.0 
    dropped_vol = 0.0 
    
    for demand in demands:
        try:
            path = nx.shortest_path(graph, demand['origin'], demand['destination'], weight='weight')
            path_cost = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            SPTT += path_cost * demand['volume']
            for i in range(len(path)-1):
                x_bar[(path[i], path[i+1])] += demand['volume']
        except nx.NetworkXNoPath:
            dropped_vol += demand['volume']
            
    return SPTT, x_bar, dropped_vol

def find_alpha(graph, x_bar, is_so=False):
    # Find the optimal alpha for convex combination of current flow and AON flow using root finding
    def df(alpha):
        sum_deriv = 0.0
        for u, v, d in graph.edges(data=True):
            tmp_flow = alpha * x_bar.get((u, v), 0.0) + (1 - alpha) * d['flow']
            tmp_cost = marginal_cost(d['fft'], d['alpha'], tmp_flow, d['capacity'], d['beta']) if is_so else bpr_cost(d['fft'], d['alpha'], tmp_flow, d['capacity'], d['beta'])
            sum_deriv += (x_bar.get((u, v), 0.0) - d['flow']) * tmp_cost
        return sum_deriv
    
    val_0, val_1 = df(0.0), df(1.0)
    if val_0 >= 0.0: return 0.0 
    if val_1 <= 0.0: return 1.0 
    
    sol = scipy.optimize.root_scalar(df, x0=0.5, bracket=(0.0, 1.0))
    return max(0.0, min(1.0, sol.root))

def solve_equilibrium(graph, demands, is_so=False, accuracy=0.001, max_iter=50):
    # Frank-Wolfe algorithm to solve for UE or SO equilibrium
    for u, v, d in graph.edges(data=True): d['flow'] = 0.0
    gap = np.inf
    iteration = 1
    TSTT = 0.0
    
    # Main loop of Frank-Wolfe
    while gap > accuracy and iteration <= max_iter:
        update_travel_times(graph, is_so)
        SPTT, x_bar, dropped_vol = load_aon(graph, demands)
        if dropped_vol > 0: return float('inf'), dropped_vol 
        
        alpha = 1.0 if iteration == 1 else find_alpha(graph, x_bar, is_so)
        for u, v, d in graph.edges(data=True):
            d['flow'] = alpha * x_bar.get((u, v), 0.0) + (1 - alpha) * d['flow']
        
        update_travel_times(graph, is_so)
        SPTT, _, _ = load_aon(graph, demands)
        TSTT = sum(d['flow'] * bpr_cost(d['fft'], d['alpha'], d['flow'], d['capacity'], d['beta']) for u, v, d in graph.edges(data=True))
        gap = (TSTT / SPTT) - 1.0 if SPTT > 0 else 0.0
        iteration += 1
    
    return TSTT, 0.0

print("Loading CSV network data...")

# Load nodes and edges from CSV files, construct the directed graph, and set attributes for BPR cost function
nodes_df = pd.read_csv('intersections.csv')
edges_df = pd.read_csv('roads_updated.csv')
edges_df['oneway'] = edges_df['oneway'].astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 't'])

G = nx.DiGraph()
CAP_PER_LANE = 1100 
for _, row in nodes_df.iterrows():
    G.add_node(str(row['id']).strip(), name=row['name'])

for _, row in edges_df.iterrows():
    u, v = str(row['A']).strip(), str(row['B']).strip()
    if u not in G.nodes or v not in G.nodes: continue
    
    dist_km = float(row['distance']) / 1000.0
    speed = float(row.get('speed', 40.0))
    fft = dist_km / speed 
    cap = int(row.get('lanes', 1)) * CAP_PER_LANE
    base_name = str(row.get('name', '')).strip()
    link_attrs = {'fft': fft, 'flow': 0.0, 'weight': fft, 'name': base_name, 'alpha': 0.15, 'beta': 4.0, 'capacity': cap}
    
    if row['oneway']:
        G.add_edge(u, v, **link_attrs)
    else:
        G.add_edge(u, v, **link_attrs)
        rev_attrs = link_attrs.copy()
        rev_attrs['name'] = f"{base_name} (Rev)"
        G.add_edge(v, u, **rev_attrs)

node_list = list(G.nodes)
total_nodes = len(node_list)
total_pairs = total_nodes * (total_nodes - 1)

print(f"\nINITIATING ALL-PAIRS CHECKING...")
print(f"Total possible Origin-Destination combinations to test: {total_pairs}")

output_filename = "All_Pairs_Result.csv"
volumes_to_test = np.arange(500, 10500, 500) 
candidate_edges = [(u, v, d.get('name', 'Unnamed Road')) for u, v, d in G.edges(data=True)]

if not os.path.exists(output_filename):
    pd.DataFrame(columns=['Origin Node', 'Origin Name', 'Destination Node', 'Destination Name', 'Critical Volume', 'Max PoA', 'Road Name', 'Closed Edge From', 'Closed Edge To', 'Closure Type', 'Baseline TSTT (hrs)', 'New TSTT (hrs)', 'Time Saved (mins)']).to_csv(output_filename, index=False)
pair_count = 0
found_braess_count = 0


# Iterate through all unique origin-destination pairs in the graph and perform the Braess paradox analysis for each pair
for origin_node in node_list:
    for destination_node in node_list:
        if origin_node == destination_node: continue
        pair_count += 1
        
        if pair_count % 10 == 0:
            print(f"--- Progress: Tested {pair_count} / {total_pairs} pairs. Found {found_braess_count} Braess traps so far. ---")
            
        if not nx.has_path(G, origin_node, destination_node): continue

        origin_name = G.nodes[origin_node].get('name', 'N/A')
        dest_name = G.nodes[destination_node].get('name', 'N/A')

        # PoA Volume Sweep
        poa_results = []
        network_failed = False

        for vol in volumes_to_test:
            single_demand = [{'origin': origin_node, 'destination': destination_node, 'volume': vol}]
            
            # Calculate UE
            tstt_ue, dropped = solve_equilibrium(G, single_demand, is_so=False)
            if dropped > 0:
                network_failed = True
                break 
                
            # Calculate SO
            tstt_so, _ = solve_equilibrium(G, single_demand, is_so=True)
            
            poa = tstt_ue / tstt_so if tstt_so > 0 else 1.0
            poa_results.append({'Volume': vol, 'Price of Anarchy': poa})

        if not poa_results: continue # Skip if failed on 500 volume
        
        df_poa = pd.DataFrame(poa_results)
        highest_vol_tested = df_poa['Volume'].max()
        max_poa_row = df_poa.loc[df_poa['Price of Anarchy'].idxmax()]
        optimal_volume = max_poa_row['Volume']
        max_poa_value = max_poa_row['Price of Anarchy']

        # print(f"Pair {pair_count}/{total_pairs} | {origin_node} -> {destination_node} | Max PoA: {max_poa_value:.4f} | Max Volume: {highest_vol_tested}")

        # If PoA is basically 1.0 everywhere, skip to save hours of processing time
        if max_poa_value < 1.01:
            continue

        locked_demand = [{'origin': origin_node, 'destination': destination_node, 'volume': optimal_volume}]
        baseline_tstt_ph2, _ = solve_equilibrium(G, locked_demand, is_so=False)
        
        pair_results = []
        tested_bidirectional = set() # Prevents testing the same two-way street twice
        
        for u, v, road_name in candidate_edges:
            edge_data_fwd = G[u][v].copy()
            
            # Unidirectional Closure
            G.remove_edge(u, v)
            new_tstt_uni, dropped_vol_uni = solve_equilibrium(G, locked_demand, is_so=False)
            G.add_edge(u, v, **edge_data_fwd) 
                
            time_diff_mins_uni = (new_tstt_uni - baseline_tstt_ph2) * 60
            if dropped_vol_uni == 0 and time_diff_mins_uni < -0.5: 
                print(f"  [!] ONE-WAY BRAESS: Closing {road_name} ({u}->{v}) saves {abs(time_diff_mins_uni):.2f} mins!")
                pair_results.append({
                    'Origin Node': origin_node, 'Origin Name': origin_name,
                    'Destination Node': destination_node, 'Destination Name': dest_name,
                    'Critical Volume': optimal_volume, 'Max PoA': round(max_poa_value, 4),
                    'Road Name': road_name, 'Closed Edge From': u, 'Closed Edge To': v,
                    'Closure Type': 'Unidirectional', # NEW COLUMN
                    'Baseline TSTT (hrs)': round(baseline_tstt_ph2, 2),
                    'New TSTT (hrs)': round(new_tstt_uni, 2),
                    'Time Saved (mins)': round(abs(time_diff_mins_uni), 2)
                })

            # Bidirectional Closure
            if G.has_edge(v, u) and (u, v) not in tested_bidirectional and (v, u) not in tested_bidirectional:
                edge_data_rev = G[v][u].copy()
                
                # Remove both directions
                G.remove_edge(u, v)
                G.remove_edge(v, u)
                
                new_tstt_bi, dropped_vol_bi = solve_equilibrium(G, locked_demand, is_so=False)
                
                # Restore both directions
                G.add_edge(u, v, **edge_data_fwd)
                G.add_edge(v, u, **edge_data_rev)
                
                # Mark as tested so we don't repeat this when the loop reaches the reverse edge
                tested_bidirectional.add((u, v))
                tested_bidirectional.add((v, u))
                
                time_diff_mins_bi = (new_tstt_bi - baseline_tstt_ph2) * 60
                if dropped_vol_bi == 0 and time_diff_mins_bi < -0.5:
                    print(f"  [!] TWO-WAY BRAESS: Full closure of {road_name} ({u}<->{v}) saves {abs(time_diff_mins_bi):.2f} mins!")
                    pair_results.append({
                        'Origin Node': origin_node, 'Origin Name': origin_name,
                        'Destination Node': destination_node, 'Destination Name': dest_name,
                        'Critical Volume': optimal_volume, 'Max PoA': round(max_poa_value, 4),
                        'Road Name': road_name, 'Closed Edge From': u, 'Closed Edge To': v,
                        'Closure Type': 'Bidirectional', 
                        'Baseline TSTT (hrs)': round(baseline_tstt_ph2, 2),
                        'New TSTT (hrs)': round(new_tstt_bi, 2),
                        'Time Saved (mins)': round(abs(time_diff_mins_bi), 2)
                    })

        # Append immediately to CSV
        if pair_results:
            found_braess_count += len(pair_results)
            pd.DataFrame(pair_results).to_csv(output_filename, mode='a', header=False, index=False)

print(f"\nEXHAUSTIVE AUDIT COMPLETE.")
print(f"Total Braess Candidates Found: {found_braess_count}")
print(f"Results saved to {output_filename}")
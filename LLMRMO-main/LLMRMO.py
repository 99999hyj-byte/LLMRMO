# -*- coding: utf-8 -*-
"""
MONLCD Large Language Model Based Modularity Optimization for Community Detection
Implementing Top k PageRank Candidate Filtering and Two Stage Collaborative Framework
"""

import networkx as nx
import numpy as np
from openai import OpenAI

# Initialize OpenAI client 
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def llm_local_decision(v, candidates, G):
    """
    Phase 1 LLM based local decision
    Utilizing semantic reasoning to evaluate candidate node affiliation
    """
    v_degree = G.degree(v)
    
    # Construct structured prompts 
    prompt = (
        "### Task Description\n"
        "In network community detection identify potential community center nodes Decision logic as follows\n"
        "- Centrality Community centers are usually nodes with higher degrees\n"
        "- Affiliation Candidates with more common neighbors with the source node represent a stronger likelihood of belonging to the same community\n"
        "Select the most suitable target node as the community center based on node information\n"
        "If no neighbor is more suitable select the source node itself as the center\n\n"
        f"Source node {v} degree {v_degree} candidate neighbor count {len(candidates)}\n"
        "Candidate neighbor detailed information\n"
    )

    for nn in candidates:
        nn_degree = G.degree(nn)
        common_neighbors = set(G.neighbors(v)) & set(G.neighbors(nn))
        prompt += (
            f"Node {nn} degree {nn_degree} "
            f"Common neighbors with source node {len(common_neighbors)} "
            f"Represents strong community affiliation with source node\n"
        )

    prompt += "\nOutput requirement Only return a single number representing the target node ID"

    # Use the specified LLM component 
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1 
    )
    
    try:
        return int(response.choices[0].message.content.strip())
    except:
        return v

def build_llm_hierarchy(G, k=30):
    """
    Local clustering stage compute PageRank and filter Top k candidate nodes
    """
    # Compute PageRank values typically with damping factor 0 85 
    pr_scores = nx.pagerank(G, alpha=0.85)
    
    D = nx.DiGraph()
    D.add_nodes_from(G)
    
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if not neighbors:
            continue
        
        # Filter Top k neighbors as candidate centers 
        candidates = sorted(neighbors, key=lambda x: pr_scores[x], reverse=True)[:k]
        
        # LLM based decision making 
        center_node = llm_local_decision(v, candidates, G)
        
        # If LLM identifies neighbor as center establish directed edge to that center 
        if center_node != v and center_node in candidates:
            D.add_edge(v, center_node)
            
    # Transform DAG structure into initial sub community set 
    root_to_node = {}
    for node in D.nodes():
        curr = node
        visited = {curr}
        while D.out_degree(curr) > 0:
            next_node = list(D.successors(curr))[0]
            if next_node in visited: break
            curr = next_node
            visited.add(curr)
        root_to_node.setdefault(curr, []).append(node)
        
    return [set(nodes) for nodes in root_to_node.values()]

def compute_modularity_gain(G, comm_i, comm_j, u=0.8):
    """
    Phase 2 improved modularity gain calculation
    Introducing similarity constraint to balance merging of different sized communities 
    """
    m = G.number_of_edges()
    if m == 0: return 0
    
    # Calculate internal edges and total degrees 
    E_ij = sum(1 for node_i in comm_i for node_j in comm_j if G.has_edge(node_i, node_j))
    sum_ki = sum(G.degree(n) for n in comm_i)
    sum_kj = sum(G.degree(n) for n in comm_j)
    
    # Community similarity function 1 minus min divided by max 
    r_ij = 1 - min(len(comm_i), len(comm_j)) / max(len(comm_i), len(comm_j))
    
    # Modified modularity variation formula 
    gain = (E_ij / m) - (1 - u * r_ij) * (sum_ki * sum_kj) / (2 * m**2)
    return gain

def merge_communities(G, communities, target_num=None, u=0.8):
    """
    Global optimization stage greedy merging 
    Supporting specified community count or modularity maximization 
    """
    while len(communities) > 1:
        # Check if target community number is reached 
        if target_num is not None and len(communities) <= target_num:
            break
        
        best_gain = -np.inf
        best_pair = None
        
        # Only consider neighbor community merging to ensure topological connectivity 
        num_comm = len(communities)
        for i in range(num_comm):
            for j in range(i + 1, num_comm):
                # Topology connectivity check 
                if any(G.has_edge(node_i, node_j) for node_i in communities[i] for node_j in communities[j]):
                    gain = compute_modularity_gain(G, communities[i], communities[j], u)
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, j)
        
        # Termination mechanism based on modularity variation 
        if target_num is None and best_gain <= 0:
            break
            
        if best_pair is None: break
        
        # Execute merge and update community set 
        i, j = best_pair
        new_comm = communities[i].union(communities[j])
        communities = [c for idx, c in enumerate(communities) if idx not in [i, j]]
        communities.append(new_comm)
        
    return communities

def monlcd_algorithm(G, target_comm_num=None, u=0.8, k=30):
    """
    Full MONLCD workflow implementation 
    """
    # Phase 1 LLM driven local clustering 
    initial_communities = build_llm_hierarchy(G, k=k)
    
    # Phase 2 modified modularity driven global merging 
    final_communities = merge_communities(G, initial_communities, target_comm_num, u)
    
    return final_communities
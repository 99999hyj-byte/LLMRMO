# -*- coding: utf-8 -*-
"""
Community Detection Evaluation Metrics Implementation
Calculating Normalized Mutual Information and Pairwise F1 Score
"""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

def calculate_nmi(G, pred_communities, true_communities):
    """
    Calculate Normalized Mutual Information NMI between partitions
    Directly computes similarity between detected and ground truth communities
    """
    all_nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    pred_labels = np.zeros(len(all_nodes), dtype=int)
    true_labels = np.zeros(len(all_nodes), dtype=int)
    
    # Assign community labels for each node in prediction
    for idx, comm in enumerate(pred_communities):
        for node in comm:
            if node in node_to_idx:
                pred_labels[node_to_idx[node]] = idx
                
    # Assign community labels for each node in ground truth
    for idx, comm in enumerate(true_communities):
        for node in comm:
            if node in node_to_idx:
                true_labels[node_to_idx[node]] = idx
    
    # NMI measures consistency between partitions from an information theoretic perspective
    return normalized_mutual_info_score(true_labels, pred_labels)

def calculate_pairwise_f1(G, pred_communities, true_communities):
    """
    Calculate Pairwise F1 score for community detection
    Evaluates accuracy based on node pair classification within communities
    """
    all_nodes = list(G.nodes())
    n = len(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    pred_labels = np.full(n, -1)
    true_labels = np.full(n, -1)
    
    for idx, comm in enumerate(pred_communities):
        for node in comm:
            if node in node_to_idx:
                pred_labels[node_to_idx[node]] = idx
                
    for idx, comm in enumerate(true_communities):
        for node in comm:
            if node in node_to_idx:
                true_labels[node_to_idx[node]] = idx
                
    tp = 0 # True Positives same community in both partitions
    fp = 0 # False Positives same in prediction but different in truth
    fn = 0 # False Negatives different in prediction but same in truth
    
    # Iterate through all unique node pairs to evaluate boundaries
    for i in range(n):
        for j in range(i + 1, n):
            # Skip nodes not belonging to any community
            if pred_labels[i] == -1 or pred_labels[j] == -1 or \
               true_labels[i] == -1 or true_labels[j] == -1:
                continue
                
            pred_same = (pred_labels[i] == pred_labels[j])
            true_same = (true_labels[i] == true_labels[j])
            
            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif not pred_same and true_same:
                fn += 1
                
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 score balances precision and recall of local community boundaries
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1
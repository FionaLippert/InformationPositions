#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import networkx as nx
import numpy as np
import os
import glob

def sample_nodes(G, num_nodes):
    probs = {}
    vals, counts = np.unique(sorted([d for n, d in G.degree()]), return_counts=True)
    probs = { vals[i] : 1/counts[i] for i in range(vals.size)}
    degree_sequence = sorted([(d,n) for n, d in G.degree()])
    prob_seq = [probs[d] for d, n in degree_sequence]
    prob_seq /= np.sum(prob_seq)
    node_samples = np.random.choice([n for d, n in degree_sequence], p=prob_seq, size=num_nodes)
    print(np.array([G.degree(n) for n in node_samples]))
    return node_samples

"""
G = nx.read_gpickle('networkData/ER/ER_k=2.0_N=1000_v0.gpickle')
g = G.copy()
g.remove_node(333)
nx.write_gpickle(g, f'networkData/ER/ER_k=2.0_N=1000_v0_without_333.gpickle', 2)
"""

n = 20
for filepath in glob.iglob('networkData/ER/ER_k=2.0_N=1000_v0.gpickle'):
    G = nx.read_gpickle(filepath)
    nodes = list(G)
    path = filepath.strip('.gpickle')
    np.save(path + '_nodes.npy', np.array(nodes))

    with open(path + f'_sample_nodes_weighted_{n}.txt', 'w') as f:
        #sample_nodes = np.random.choice(nodes, size=10, replace=False)
        samples = sample_nodes(G, n)
        for node in samples:
            f.write(f'{node}\n')

"""
G = nx.read_gpickle('networkData/unweighted_criminal_after_2012.gpickle')
nodes = list(G)
np.save('networkData/unweighted_criminal_after_2012_nodes.npy', np.array(nodes))


with open('networkData/unweighted_criminal_after_2012_sample_nodes_weighted.txt', 'w') as f:
    #sample_nodes = np.random.choice(nodes, size=10, replace=False)
    samples = sample_nodes(G, 10)
    for n in samples:
        f.write(f'{n}\n')
"""

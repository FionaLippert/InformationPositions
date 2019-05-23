#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import networkx as nx
import numpy as np
import os
import glob
import argparse

parser = argparse.ArgumentParser(description='sample a subset of nodes with a representative degree distribution')
parser.add_argument('graph', type=str, help='path to graph')
parser.add_argument('n', type=int, help='number of nodes to sample')
parser.add_argument('--add', action="store_true", help='add nodes to existing node samples to get n unique nodes')

def sample_nodes(G, num_nodes):
    probs = {}
    vals, counts = np.unique(sorted([d for n, d in G.degree()]), return_counts=True)
    probs = { vals[i] : 1/counts[i] for i in range(vals.size)}
    degree_sequence = sorted([(d,n) for n, d in G.degree()])
    prob_seq = [probs[d] for d, n in degree_sequence]
    prob_seq /= np.sum(prob_seq)
    node_samples = np.random.choice([n for d, n in degree_sequence], p=prob_seq, size=num_nodes, replace=False)
    print(np.array([G.degree(n) for n in node_samples]))
    return node_samples

def add_nodes(G, nodes, target_num):
    unique_nodes = np.unique(nodes)
    extended_nodes = np.zeros(target_num, dtype=int)
    extended_nodes[:unique_nodes.size] = unique_nodes
    for i in range(target_num - unique_nodes.size):
        new = sample_nodes(G, 1)[0]
        while new in extended_nodes[:unique_nodes.size + i]:
            new = sample_nodes(G, 1)[0]
        print(f'add new node: {new}')
        extended_nodes[unique_nodes.size + i] = new
    return extended_nodes

if __name__ == '__main__':

    args = parser.parse_args()

    """
    G = nx.read_gpickle('networkData/ER/ER_k=2.0_N=1000_v0.gpickle')
    g = G.copy()
    g.remove_node(333)
    nx.write_gpickle(g, f'networkData/ER/ER_k=2.0_N=1000_v0_without_333.gpickle', 2)
    """

    ensemble = [g for g in glob.iglob(f'networkData/{args.graph}*.gpickle')]

    for filepath in ensemble: #filepath in glob.iglob(args.graph):
        print(filepath)
        G = nx.read_gpickle(filepath)
        nodes = list(G)
        path = filepath.strip('.gpickle')
        np.save(path + '_nodes.npy', np.array(nodes))

        if args.add:
            samples = np.load(path + f'_sample_nodes_weighted_{args.n}.npy')
            print(f'old: {samples}')
            samples = add_nodes(G, samples, args.n)
            print(f'new: {samples}')
        else:
            samples = sample_nodes(G, args.n)

        np.save(path + f'_sample_nodes_weighted_{args.n}.npy', samples)


        with open(path + f'_sample_nodes_weighted_{args.n}.txt', 'w') as f:
            #sample_nodes = np.random.choice(nodes, size=10, replace=False)
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

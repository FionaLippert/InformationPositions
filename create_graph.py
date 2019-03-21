#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import networkx as nx, itertools, scipy,\
        os, pickle, h5py, sys, multiprocessing as mp, json,\
        datetime, sys
import time
from numpy import *

def create_undirected_tree(z, depth, path=None):
    graph = nx.balanced_tree(z, depth)

    if path is not None: nx.write_gpickle(graph, \
                f'{path}/undirected_tree_z={z}_depth={depth}.gpickle', 2)

    return graph

def create_directed_tree(z, depth, path=None):
    graph = nx.DiGraph()
    graph = nx.balanced_tree(z, depth, create_using=graph)

    if path is not None: nx.write_gpickle(graph, \
                f'{path}/directed_tree_z={z}_depth={depth}.gpickle', 2)

    return graph

def create_cayley_tree(z, depth, path=None):
    subtrees = [(nx.balanced_tree(z,depth-1), 0) for _ in range(z+1)]
    graph = nx.join(subtrees)

    if path is not None: nx.write_gpickle(graph, \
                f'{path}/cayley_tree_z={z}_depth={depth}.gpickle', 2)

    return graph


def create_erdos_renyi_graph(N, avg_deg=2., path=None):

    p = avg_deg/N

    graph = nx.erdos_renyi_graph(N, p)
    connected_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(connected_nodes)
    #N = len(graph)
    if path is not None: nx.write_gpickle(graph,
                f'{path}/ER_k={avg_deg}_N={N}.gpickle', 2)

def create_powerlaw_graph(N, gamma=1.6, path=None):
    seq = nx.utils.powerlaw_sequence(N, gamma)
    graph = nx.expected_degree_graph(seq, selfloops = False)
    graph = sorted(nx.connected_component_subgraphs(graph), key=len, reverse=True)[0]
    #N = len(graph)
    if path is not None: nx.write_gpickle(graph,
                f'{path}/scaleFree_gamma={gamma}_N={N}.gpickle')

def create_2D_grid(L, path=None):

    graph = nx.grid_2d_graph(L, L, periodic=True)
    if path is not None: nx.write_gpickle(graph,
                f'{path}/2D_grid_L={L}.gpickle', 2)




if __name__ == '__main__':

    now = time.time()
    targetDirectory = f'{os.getcwd()}/networkData'

    undirected_2 = create_undirected_tree(2, 6, targetDirectory)
    undirected_3 = create_undirected_tree(3, 6, targetDirectory)
    undirected_4 = create_undirected_tree(4, 6, targetDirectory)
    undirected_5 = create_undirected_tree(5, 6, targetDirectory)

    directed_2 = create_directed_tree(2, 6, targetDirectory)
    directed_3 = create_directed_tree(3, 6, targetDirectory)
    directed_4 = create_directed_tree(4, 6, targetDirectory)
    directed_5 = create_directed_tree(5, 6, targetDirectory)

    cayley_2 = create_cayley_tree(2, 6, targetDirectory)
    cayley_3 = create_cayley_tree(3, 6, targetDirectory)
    cayley_4 = create_cayley_tree(4, 6, targetDirectory)
    cayley_5 = create_cayley_tree(5, 6, targetDirectory)

    ER_N100_k15 = create_erdos_renyi_graph(100, 1.5, targetDirectory)
    ER_N100_k2 = create_erdos_renyi_graph(100, 2.0, targetDirectory)
    ER_N100_k25 = create_erdos_renyi_graph(100, 2.5, targetDirectory)
    ER_N100_k3 = create_erdos_renyi_graph(100, 3.0, targetDirectory)

    ER_N500_k15 = create_erdos_renyi_graph(500, 1.5, targetDirectory)
    ER_N500_k2 = create_erdos_renyi_graph(500, 2.0, targetDirectory)
    ER_N500_k25 = create_erdos_renyi_graph(500, 2.5, targetDirectory)
    ER_N500_k3 = create_erdos_renyi_graph(500, 3.0, targetDirectory)

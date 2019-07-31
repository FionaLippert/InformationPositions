#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import networkx as nx, itertools, scipy,\
        os, pickle, h5py, sys, multiprocessing as mp, json,\
        datetime, sys
import time
import numpy as np

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

def create_undirected_star(z, path=None):
    graph = nx.star_graph(z)

    if path is not None: nx.write_gpickle(graph, \
                f'{path}/undirected_star_z={z}.gpickle', 2)

    return graph

def create_directed_star_path_graph(z, depth, path=None):
    graph = nx.DiGraph()
    graph.add_star(range(z+1))
    for node in range(1, z+1):
        path_nodes = [node]
        path_nodes.extend(range(len(graph), len(graph)+depth))
        graph.add_path(path_nodes)

    if path is not None: nx.write_gpickle(graph, \
                f'{path}/directed_star_path_graph_z={z}_depth={depth}.gpickle', 2)

    return graph

def create_undirected_star_path_graph(z, depth, path=None):
    graph = nx.Graph()
    graph.add_star(range(z+1))
    for node in range(1, z+1):
        path_nodes = [node]
        path_nodes.extend(range(len(graph), len(graph)+depth))
        graph.add_path(path_nodes)

    if path is not None: nx.write_gpickle(graph, \
                f'{path}/undirected_star_path_graph_z={z}_depth={depth}.gpickle', 2)

    return graph

def create_regular_graph(N, d, path=None):
    graph = nx.random_regular_graph(d, N)
    connected_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(connected_nodes)

    if path is not None: nx.write_gpickle(graph,
                f'{path}/regular_graph_d={d}_N={N}.gpickle', 2)
    return graph


def create_erdos_renyi_graph(N, avg_deg=2., path=None, version=''):

    p = avg_deg/N

    graph = nx.erdos_renyi_graph(N, p)
    connected_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(connected_nodes)
    #N = len(graph)
    if path is not None: nx.write_gpickle(graph,
                f'{path}/ER_k={avg_deg:.2f}_N={N}{version}.gpickle', 2)
    return graph

def create_erdos_renyi_graph_exactN(N_target, N_start, avg_deg, max_trials=int(1e5), path=None, version=''):

    p = avg_deg/N_start
    print(p)

    for trial in range(max_trials):
        G = nx.erdos_renyi_graph(N_start, p=p)
        connected_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(connected_nodes)
        if len(G) == N_target: break

    k = np.mean([d for n,d in nx.degree(G)])
    print(f'avg degree = {k}')
    if path is not None: nx.write_gpickle(G,
                f'{path}/ER_k={avg_deg:.2f}_N={N_target}{version}.gpickle', 2)

def create_watts_strogatz(N, k, beta, path=None, version=''):
    graph = nx.watts_strogatz_graph(N, k, beta)
    connected_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(connected_nodes)
    clustering = nx.average_clustering(graph)
    print(clustering)
    if path is not None: nx.write_gpickle(graph,
                f'{path}/WS_k={k}_beta={beta}_N={N}{version}.gpickle', 2)
    return graph

def create_powerlaw_graph(N, gamma=1.6, path=None, version=''):
    seq = nx.utils.powerlaw_sequence(N, gamma)
    graph = nx.expected_degree_graph(seq, selfloops = False)
    graph = sorted(nx.connected_component_subgraphs(graph), key=len, reverse=True)[0]
    #N = len(graph)
    if path is not None: nx.write_gpickle(graph,
                f'{path}/SF_gamma={gamma}_N={N}{version}.gpickle')
    return graph


def create_barabasi_albert_graph(N, m, path=None, version=''):
    graph=nx.barabasi_albert_graph(N, m)
    graph = sorted(nx.connected_component_subgraphs(graph), key=len, reverse=True)[0]
    #N = len(graph)
    if path is not None: nx.write_gpickle(graph,
                f'{path}/BA_m={m}_N={N}{version}.gpickle')
    return graph


def create_2D_grid(L, path=None, version=''):

    G = nx.grid_2d_graph(L, L, periodic=True)
    mapping = {n : idx for idx, n in enumerate(list(G))}
    edges = [(mapping[e[0]], mapping[e[1]]) for e in G.edges]

    graph = nx.Graph()
    graph.add_nodes_from(list(mapping.values()))
    graph.add_edges_from(edges)
    if path is not None: nx.write_gpickle(graph,
                f'{path}/2D_grid_L={L}{version}.gpickle', 2)
    return graph

def create_path_graph(L, path=None):
    graph = nx.path_graph(L)
    if path is not None: nx.write_gpickle(graph,
                f'{path}/path_graph_L={L}.gpickle', 2)
    return graph




if __name__ == '__main__':

    N_target = 70
    k = 0.05*1.2*N_target

    for i in range(10):
        targetDirectory = f'{os.getcwd()}/networkData/small_graphs/N={N_target}_p=0.05/ER_k={k:.2f}_N={N_target}_v{i}'
        os.makedirs(targetDirectory, exist_ok=True)
        #create_erdos_renyi_graph_exactN(N_target, int(1.2*N_target), k, path=targetDirectory, version=f'_v{i}')
        create_erdos_renyi_graph_exactN(N_target, N_target, k, path=targetDirectory, version=f'_v{i}')

    #create_path_graph(100, targetDirectory)

    #nx.write_gpickle(nx.krackhardt_kite_graph(), f'{targetDirectory}/small_graphs/kitegraph.gpickle', 2)

    """
    for N_target in [2, 5, 10, 20, 30, 40, 50]:
        create_erdos_renyi_graph_exactN(N_target, 2*N_target, 2.0, path=targetDirectory)
    """

    """
    targetDirectory = f'{os.getcwd()}/networkData/trees'
    os.makedirs(targetDirectory, exist_ok=True)

    for depth in [6, 8]:
        for z in [2, 3, 4, 5]:
            undirected_tree = create_undirected_tree(z, depth, targetDirectory)


    for z in range(2,6):
        cayley_tree = create_cayley_tree(z, 8, targetDirectory)
    """

    """
    targetDirectory = f'{os.getcwd()}/networkData/star_graph'
    os.makedirs(targetDirectory, exist_ok=True)
    for z in range(3,8):
        create_undirected_star_path_graph(z, 10, targetDirectory)
    """

    """
    targetDirectory = f'{os.getcwd()}/networkData/regular_graphs'
    os.makedirs(targetDirectory, exist_ok=True)
    #create_regular_graph(1000, 2, targetDirectory)
    #create_regular_graph(10000, 3, targetDirectory)
    #create_regular_graph(10000, 4, targetDirectory)
    create_regular_graph(10000, 5, targetDirectory)
    """


    beta = 0.2
    k = 4
    targetDirectory = f'{os.getcwd()}/networkData/WS/WS_k={k}_beta={beta}_N=1000'
    os.makedirs(targetDirectory, exist_ok=True)
    for i in range(10):
        create_watts_strogatz(1000, k, beta, targetDirectory, f'_v{i}')


    """
    targetDirectory = f'{os.getcwd()}/networkData/ER'
    for N in [250, 500, 750, 1000]:
        for k in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            for i in range(10):
                create_erdos_renyi_graph(N, k, targetDirectory, f'_v{i}')
    """

    """
    targetDirectory = f'{os.getcwd()}/networkData/BA'
    os.makedirs(targetDirectory, exist_ok=True)
    N=1000
    m=3
    for i in range(10):
        G = create_barabasi_albert_graph(N, m, targetDirectory, f'_v{i}')
        nodes = list(G)
        np.save(f'{targetDirectory}/BA_m={m}_N={N}_v{i}_nodes.npy', np.array(nodes))
        with open(f'{targetDirectory}/BA_m={m}_N={N}_v{i}_sample_nodes.txt', 'w') as f:
            sample_nodes = np.random.choice(nodes, size=10, replace=False)
            for n in sample_nodes:
                f.write(f'{n}\n')
    """

    """
    targetDirectory = f'{os.getcwd()}/networkData/SF'
    os.makedirs(targetDirectory, exist_ok=True)
    N=1000
    gamma=2.2
    for i in range(10):
        G = create_powerlaw_graph(N, gamma, targetDirectory, f'_v{i}')
        nodes = list(G)
        np.save(f'{targetDirectory}/SF_gamma={gamma}_N={N}_v{i}_nodes.npy', np.array(nodes))
        with open(f'{targetDirectory}/SF_gamma={gamma}_N={N}_v{i}_sample_nodes.txt', 'w') as f:
            sample_nodes = np.random.choice(nodes, size=10, replace=False)
            for n in sample_nodes:
                f.write(f'{n}\n')
    """

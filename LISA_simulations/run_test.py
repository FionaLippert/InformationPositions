#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Casper van Elteren'
"""
Created on Mon Jun 11 09:06:57 2018

@author: casper
"""

from Models import fastIsing
from Toolbox import infcy
from Utils import IO, plotting as plotz
from Utils.IO import SimulationResult
import networkx as nx, itertools, scipy,\
        os, pickle, h5py, sys, multiprocessing as mp, json,\
        datetime, sys
import time
import timeit
from timeit import default_timer as timer
from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from functools import partial
from scipy import sparse
from threading import Thread
close('all')


def plot_avgMI(MI, degrees, diameter, title, path):
    fig, ax = subplots(figsize=(8,5))
    #[ax.errorbar(range(1,diameter+1), MI[i,:,0], MI[i,:,1], label=degrees[i], ls='--', marker='o', capsize=5) for i in range(nodes.size)]
    [ax.errorbar(range(1,diameter+1), np.nanmean(MI[i,:,:], axis=1), np.nanstd(MI[i,:,:], axis=1), label=degrees[i], ls='--', marker='o', capsize=5) for i in range(MI.shape[0])]
    ax.set_xlabel('node distance')
    ax.set_ylabel('<MI>')
    #ax.set_title('erdos_renyi_graph, N={}, T={}'.format(N,T))
    ax.set_title(title)
    ax.legend()
    savefig(path)

def plot_avgMI_fit(MI, degrees, diameter, title, path):
    func = lambda x, a, b, c:  a / (1 + exp(b * (x - c)))

    fig, ax = subplots(figsize=(8,5))
    [ax.errorbar(range(1,diameter+1), np.nanmean(MI[i,:,:], axis=1), np.nanstd(MI[i,:,:], axis=1), label=degrees[i], ls='--', marker='o', capsize=5) for i in range(MI.shape[0])]
    for i in range(MI.shape[0].size):
        mi = MI[i,:,0][np.where(np.isfinite(MI[i,:,0]))]
        ax.plot(range(1,mi.size+1), mi, 'o', label=degrees[i])
        a, b = scipy.optimize.curve_fit(func, range(3,mi.size+1), mi[2:], p0=[0.5, 3, 7])
        xx = linspace(3, mi.size+1, 1000)
        ax.plot(xx, func(xx, *a))
    ax.set_xlabel('node distance')
    ax.set_ylabel('<MI>')
    #ax.set_title('erdos_renyi_graph, N={}, T={}'.format(N,T))
    ax.set_title(title)
    ax.legend()
    savefig(path)

if __name__ == '__main__':

    # 2e4 steps with non-single updates and 32x32 grid --> serial-time = parallel-time

    temps         = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    trials        = 1
    repeats       = 8
    burninSamples = int(1e4) # int(1e6)
    nSamples      = int(50) #int(1e6)
    distSamples   = int(1e3)
    magSide       = '' # which sign should the overall magnetization have (''--> doesn't matter, 'neg' --> flip states if <M> > 0, 'pos' --> flip if <M> < 0)
    updateType    = 'async'


    #graph = nx.grid_2d_graph(16, 16, periodic=True)
    avg_deg = 2.5
    N = 1000
    p = avg_deg/N

    """
    graph = nx.erdos_renyi_graph(N, p)
    connected_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(connected_nodes)
    nx.write_gpickle(graph, f'ER_avgDeg={avg_deg}_N={N}.gpickle', 2)
    print("avg degree = {}".format(np.mean([d for k, d in graph.degree()])))
    """
    graph = nx.read_gpickle("networkData/ER_avgDeg=1.5_N=100.gpickle")
    #graph = nx.read_gpickle("networkData/ER_avgDeg4.gpickle")


    N = len(graph)
    print(N)

    #graph = nx.grid_2d_graph(100, 100, periodic=True)

    diameter = nx.diameter(graph)
    #diameter = 34
    print("diameter = {}".format(diameter))

    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)

    print(targetDirectory)

    # graph = nx.barabasi_albert_graph(10, 3)
    modelSettings = dict(\
                         graph       = graph,\
                         temperature = 1.0,\
                         updateType  = updateType,\
                         magSide     = magSide
                         )
    model = fastIsing.Ising(**modelSettings)
    updateType = model.updateType


    node_deg_ranks = range(10) #[1, 2, 500, 501, 2000, 2001, 3000, 3001] # highest degree, lowest and some intermediate ones
    #node_deg_ranks = [1, 2, 500, 501, 2000, 2001] # highest degree, lowest and some intermediate ones
    all_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    selected_nodes = [all_nodes[i][0] for i in node_deg_ranks]
    nodes = np.array([model.mapping[n] for n in selected_nodes], dtype=np.intc)

    settings = dict(
        T                = temps,\
        nSamples         = nSamples,\
        distSamples      = distSamples,\
        burninSamples    = burninSamples,\
        repeats          = repeats,\
        trials           = trials,\
        updateMethod     = updateType,\
        nNodes           = graph.number_of_nodes(),\
        selectedGraphNodes    = selected_nodes
        )
    IO.saveSettings(targetDirectory, settings)


    #print([len(nx.ego_graph(graph, all_nodes[0][0], i)) - len(nx.ego_graph(graph, all_nodes[0][0], i-1)) for i in range(1,10)])
    #print([len(nx.ego_graph(graph, all_nodes[100][0], i)) - len(nx.ego_graph(graph, all_nodes[100][0], i-1)) for i in range(1,10)])
    #infcy.getSnapShots(model, nSamples=16, step = 10,\
    #                  burninSamples = int(0))
    #infcy.getSnapShotsLargeNetwork(model, nSamples=100, step = 1000,\
    #                   burninSamples = int(0), nodeSubset = model.neighboursAtDist(0, 5)
    #past = timer()
    #infcy.test(model, node=0, dist=2, nSnapshots=int(1e3), nStepsToSnapshot=int(1e2),
    #              nSamples=int(50), distSamples=int(1e2), nRunsSampling=int(1e2))
    #print(f'time = {timer()-past} sec')

    """
    mixingTimes, corrTimes = infcy.mixingTimePerTemp(model, \
                            nInitialConfigs = 20, \
                            temps = np.array([model.t]), \
                            stepSizeBurnin = 10, \
                            nStepsRegress = int(1e3), \
                            thresholdReg = 0.05, \
                            nStepsCorr = int(1e4), \
                            thresholdCorr = 0.05)

    distSamples = corrTimes[0]
    print(f'correlation time = {distSamples}')
    """
    distSamples = 3000
    #print(f'correlation time = {distSamples}')

    node = 0
    maxDist = 5
    neighbours = model.neighboursAtDist(node, maxDist)
    """
    snapshots = infcy.getSnapshotsPerDist(model, nSamples=int(1e2), nSteps=int(distSamples), node=node, maxDist=maxDist, threshold=1e-4)
    #model.reset()
    #model.seed += 1
    #snapshots2 = infcy.getSnapshotsPerDist(model, nSamples=int(1e3), nSteps=int(distSamples), node=node, maxDist=maxDist, threshold=1e-4)
    d = 4

    sortedSnapshots = sorted(snapshots[d].items(), key=lambda x: x[1], reverse=True)
    top100 = sortedSnapshots[:10]
    #sortedSnapshots2 = sorted(snapshots2[d].items(), key=lambda x: x[1], reverse=True)
    #top1002 = sortedSnapshots2[:10]

    dictTop = dict(top100)
    #dictTop2 = dict(top1002)

    tmp = dict(snapshots = snapshots)
    with open(f'{targetDirectory}/snapshots_node={node}.pickle', 'wb') as f:
        pickle.dump(snapshots, f)
    """
    with open(f'Data/1551108283.320843/snapshots_node={node}.pickle', 'rb') as f:
        snapshots = pickle.load(f)
    #for k in dictTop:
    #    if k in list(dictTop2.keys()):
    #        print(dictTop[k], dictTop2[k])
    #    else:
    #        print(dictTop[k])



    # how many samples are needed to obtain stable estimate ?
    d = 5
    states = list(snapshots[d-1].keys())[5]
    print(np.fromstring(states))
    model.fixedNodes = neighbours[d]
    conds = []
    for i in tqdm(range(10)):
        model.reset()
        model.seed += 1
        probCond = infcy.monteCarloFixedNeighboursSeq(model, states, node, \
                   neighbours[d], 0, \
                   nSamples = int(1e3), distSamples = distSamples)
        conds.append(probCond[0])
        print(conds)
    print(np.std(conds))

    # TODO how many snapshots need to be considered to get stable estimate of MI?
    """
    MIs = np.zeros(maxDist)
    for d in range(1, maxDist+1):
        _, _, MI = infcy.runNeighbourhoodMI(model, node, neighbours[d], snapshots[d-1], \
                  nBurnin=int(distSamples), nSamples=int(1e3), distSamples=int(distSamples))
        MIs[d-1] = MI
    fig, ax = subplots(figsize=(8,5))
    ax.plot(range(1,maxDist+1), MIs, ls='--', marker='o')
    ax.set_xlabel('distance')
    ax.set_ylabel('MI')
    fig.savefig(f'{targetDirectory}/MIperDist.png')
    np.save(f'{targetDirectory}/MI_T={model.t}_{time.time()}.npy', MIs)
    """

    """
    for T in temps:
        model.t = T
        for trial in range(trials):
            snapshots, MI, degrees = infcy.runMI(model, repeats, burninSamples, nSamples, distSamples, nodes=nodes, distMax=diameter, targetDirectory=targetDirectory)
            np.save(f'{targetDirectory}/snapshots_T={T}_{time.time()}.npy', snapshots)
            np.save(f'{targetDirectory}/MI_T={T}_{time.time()}.npy', MI)
            plot_avgMI(MI, degrees, diameter, 'erdos_renyi_graph, N={}, T={}'.format(N,T), f'{targetDirectory}/avgMIperDist_T{T}_{time.time()}.png')
    """

    #MI_switch, degrees = infcy.runMI(model, repeats=16, burninSamples=int(1e4), nSamples=int(20), distSamples=int(1e3), nodes=nodes, distMax=diameter, magThreshold=-0.025)
    #MI_normal, degrees = infcy.runMI(model, repeats=16, burninSamples=int(1e4), nSamples=int(20), distSamples=int(1e3), nodes=nodes, distMax=diameter, magThreshold=0.025)

    #np.save(f'{targetDirectory}/MI_switch.npy', MI_switch)
    #np.save(f'{targetDirectory}/MI_normal.npy', MI_normal)


    #degrees = [11,11,7,7,4,4]
    #MI = np.load("Data/1549365170.2915149/MI.npy")


    #plot_avgMI(MI_switch, degrees, diameter, 'erdos_renyi_graph, |<M>| < 0.025, N={}, T={}'.format(N,T), f'{targetDirectory}/avgMIperDist_switch.png')
    #plot_avgMI(MI_normal, degrees, diameter, 'erdos_renyi_graph, |<M>| >= 0.025, N={}, T={}'.format(N,T), f'{targetDirectory}/avgMIperDist_normal.png')



    """
    fig, ax = subplots(figsize=(8,5))
    [ax.scatter(range(1,diameter+1), MI[0,:,i], alpha = .2) for i in range(MI.shape[2])]
    ax.set_xlabel('node distance')
    ax.set_ylabel('MI')
    ax.set_title('erdos_renyi_graph, N={}, T={}'.format(N,T))
    #ax.legend()
    savefig(f'{targetDirectory}/scatterMIperDist.png')

    fig, ax = subplots(figsize=(8,5))
    [ax.plot(range(1,diameter+1), MI[i,:,2], label=degrees[i], ls='--', marker='o') for i in range(nodes.size)]
    ax.set_xlabel('node distance')
    ax.set_ylabel('sum(MI)')
    ax.set_title('erdos_renyi_graph, N={}, T={}'.format(N,T))
    ax.legend()
    savefig(f'{targetDirectory}/summedMIperDist.png')
    """
    print(targetDirectory)

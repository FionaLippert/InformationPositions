#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


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
from scipy import sparse, stats
from threading import Thread
close('all')


def computeMI_joint(jointSnapshots, maxDist, Z):
    MIs = []
    for d in tqdm(range(maxDist)):
        P_XY = np.array(sorted([v/Z2 for s in jointSnapshots[d].keys() for v in jointSnapshots[d][s].values()]))
        P_X = np.array([sum(list(dict_s.values()))/Z for dict_s in jointSnapshots[d].values()])
        all_keys = set.union(*[set(dict_s.keys()) for dict_s in jointSnapshots[d].values()])
        P_Y = np.array([jointSnapshots[d][1][k]/Z if k in jointSnapshots[d][1] else 0 for k in all_keys]) + \
                np.array([jointSnapshots[d][-1][k]/Z if k in jointSnapshots[d][-1] else 0 for k in all_keys])

        MI = stats.entropy(P_X, base=2) + stats.entropy(P_Y, base=2) - stats.entropy(P_XY, base=2)
        MIs.append(MI)
    return MIs


def computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings):
    MIs = []
    subgraph_nodes = [node]
    for d in range(1, maxDist+1):
        # get subgraph and outer neighbourhood at distance d
        if d in neighbours_G.keys():
            subgraph_nodes.extend(neighbours_G[d])
            subgraph = graph.subgraph(subgraph_nodes)

            if d >= minDist:
                print(f'------------------- distance d={d}, num neighbours = {len(neighbours_G[d])}, num states = {len(snapshots[d-1])} -----------------------')
                model_subgraph = fastIsing.Ising(subgraph, **modelSettings)
                # determine correlation time for subgraph Ising model
                mixingTime_subgraph, distSamples_subgraph, _ = infcy.determineCorrTime(model_subgraph, **corrTimeSettings)
                #distSamples_subgraph = 1000
                print(f'correlation time = {distSamples_subgraph}')
                print(f'mixing time      = {mixingTime_subgraph}')

                _, _, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, threads=7)

                #_, _, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                #          nSamples=nSamples, distSamples=distSamples_subgraph)
                MIs.append(MI)
    print(MIs)
    np.save(f'{targetDirectory}/MI_cond_T={model.t}_nSamples={nSamples*nTrials}.npy', np.array(MIs))
    return MIs



if __name__ == '__main__':

    # create data directory
    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)
    print(targetDirectory)

    # setup Ising model with nNodes spin flip attempts per simulation step
    # set temp to np.infty --> completely random
    modelSettings = dict( \
        temperature     = 0.5, \
        updateType      = 'async' ,\
        magSide         = ''
    )
    IO.saveSettings(targetDirectory, modelSettings, 'model')

    # load network
    maxDist = 20
    minDist = 11
    nGraphs = 5
    MIruns = np.zeros(nGraphs)

    for i, n in enumerate(range(10, 11)):
        graph = nx.DiGraph()
        #graph = nx.star_graph(int(n), create_using=graph)
        graph.add_star(range(int(n)))
        for node in range(1, int(n)):
            path_nodes = [node]
            path_nodes.extend(range(len(graph), len(graph)+maxDist))
            graph.add_path(path_nodes)
        print(graph.edges())

        model = fastIsing.Ising(graph, **modelSettings)


        # determine mixing/correlation time
        mixingTimeSettings = dict( \
            nInitialConfigs = 10, \
            burninSteps  = 10, \
            nStepsRegress   = int(1e3), \
            nStepsCorr      = int(1e4), \
            thresholdReg    = 0.05, \
            thresholdCorr   = 0.05
        )
        mixingTime, distSamples, mags = infcy.determineCorrTime(model, **mixingTimeSettings)
        #mixingTime = 100
        #distSamples = 100
        print(f'correlation time = {distSamples}')
        print(f'mixing time      = {mixingTime}')


        corrTimeSettings = dict( \
            nInitialConfigs = 10, \
            burninSteps  = mixingTime, \
            nStepsCorr      = int(1e4), \
            thresholdCorr   = 0.05, \
            checkMixing     = 0
        )

        node = list(graph)[0]


        nSnapshots = 100
        snapshotSettingsCond = dict( \
            nSamples    = nSnapshots, \
            burninSamples = mixingTime, \
            maxDist     = maxDist
        )

        snapshots, neighbours_G, neighbours_idx = infcy.getSnapshotsPerDist2(model, node, **snapshotSettingsCond)
        print(neighbours_G)

        nTrials = 100
        nSamples = 100

        computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings)[0]

    #np.save(f'{targetDirectory}/MI_cond_T={model.t}_nSamples={nSamples*nTrials}.npy', np.array(MIruns))


    print(targetDirectory)

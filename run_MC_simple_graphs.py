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
import itertools
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
        P_XY = np.array(sorted([v/Z for s in jointSnapshots[d].keys() for v in jointSnapshots[d][s].values()]))
        P_X = np.array([sum(list(dict_s.values()))/Z for dict_s in jointSnapshots[d].values()])
        all_keys = set.union(*[set(dict_s.keys()) for dict_s in jointSnapshots[d].values()])
        P_Y = np.array([jointSnapshots[d][1][k]/Z if k in jointSnapshots[d][1] else 0 for k in all_keys]) + \
                np.array([jointSnapshots[d][-1][k]/Z if k in jointSnapshots[d][-1] else 0 for k in all_keys])

        #print(P_XY, P_Y, P_X)

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
                mixingTime_subgraph, meanMag, distSamples_subgraph, _ = infcy.determineCorrTime(model_subgraph, **corrTimeSettings)
                #distSamples_subgraph = 50
                print(f'correlation time = {distSamples_subgraph}')
                print(f'mixing time      = {mixingTime_subgraph}')

                _, _, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, threads=1)

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

    # load network
    graph = nx.DiGraph()
    #graph = nx.path_graph(20, create_using=graph)
    #graph = nx.path_graph(20)
    #graph = nx.DiGraph()
    graph = nx.balanced_tree(2,6, create_using=graph)
    #graph = nx.balanced_tree(2,6)
    #graph.remove_edge(2,1)
    #graph.add_edge(2,1)
    N = len(graph)
    #maxDist = N-1
    maxDist = 6

    #print(graph.edges())

    #theory = np.zeros(maxDist)
    #for d in tqdm(range(maxDist)):
    #    theory[d] = infcy.MI_tree_theory(d, 1.0, 2)
    #    print(theory[d])
    #np.save(f'{targetDirectory}/MI_bin_tree_theory.npy', theory)

    path = 'nx.balanced_tree(2,10)'
    #path = 'nx.path_graph'

    networkSettings = dict( \
        path = path, \
        nNodes = N
    )


    # setup Ising model with nNodes spin flip attempts per simulation step
    # set temp to np.infty --> completely random
    modelSettings = dict( \
        temperature     = 1.0, \
        updateType      = 'async' ,\
        magSide         = ''
    )
    IO.saveSettings(targetDirectory, modelSettings, 'model')
    model = fastIsing.Ising(graph, **modelSettings)


    # determine mixing/correlation time
    mixingTimeSettings = dict( \
        nInitialConfigs = 10, \
        burninSteps  = 10, \
        nStepsRegress   = int(1e3), \
        nStepsCorr      = int(1e4), \
        thresholdReg    = 0.05, \
        thresholdCorr   = 0.01
    )
    IO.saveSettings(targetDirectory, mixingTimeSettings, 'mixingTime')
    mixingTime, meanMag, distSamples, mags = infcy.determineCorrTime(model, **mixingTimeSettings)
    print(f'correlation time = {distSamples}')
    print(f'mixing time      = {mixingTime}')
    #distSamples = 100

    for key, values in mags.items():
        np.save(f'{targetDirectory}/magSeries_{key}.npy', np.array(values))


    corrTimeSettings = dict( \
        nInitialConfigs = 10, \
        burninSteps  = mixingTime, \
        nStepsCorr      = int(1e4), \
        thresholdCorr   = 0.01, \
        checkMixing     = 0
    )
    IO.saveSettings(targetDirectory, corrTimeSettings, 'corrTime')

    #mixingTime2 = infcy.mixing2(model, nInitialConfigs=1000, nSteps=100, threshold = 0.005)
    #print(mixingTime2)
    #mixingTime2 = infcy.mixing2(model, nInitialConfigs=10, nSteps=100, threshold = 5e-4)
    #print(mixingTime2)

    node = list(graph)[0]
    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(model.mapping[node], maxDist)
    
    """
    snapshotSettingsJoint = dict( \
        nSamples    = int(1e3), \
        burninSamples = mixingTime, \
        maxDist     = maxDist, \
        nBins       = 50, \
        threshold   = 0.001
    )
    IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')

    jointSnapshots, avgSnapshots, Z = infcy.getJointSnapshotsPerDist(model, node, allNeighbours_idx, **snapshotSettingsJoint, threads=1)
    print(f'Z={Z}')

    with open(f'{targetDirectory}/jointSnapshots_node={node}.pickle', 'wb') as f:
        pickle.dump(jointSnapshots, f)
    with open(f'{targetDirectory}/avgSnapshots_node={node}.pickle', 'wb') as f:
        pickle.dump(avgSnapshots, f)
    """

    """
    with open(f'Data/jointSnapshots_node={node}.pickle', 'rb') as f:
        jointSnapshots = pickle.load(f)
    with open(f'Data/avgSnapshots_node={node}.pickle', 'rb') as f:
        avgSnapshots = pickle.load(f)

    Z2=6000.0
    """
    """
    MIs = computeMI_joint(jointSnapshots, maxDist, Z)
    np.save(f'{targetDirectory}/MI_joint_T={model.t}.npy', np.array(MIs))
    MIs_avg = computeMI_joint(avgSnapshots, maxDist, Z)
    np.save(f'{targetDirectory}/MI_avg_T={model.t}.npy', np.array(MIs_avg))
    print(MIs)
    print(MIs_avg)
    """
    """
    pairwiseMISettings = dict( \
        repeats    = 16, \
        burninSamples = mixingTime, \
        nSamples     = int(1e3), \
        distSamples   = distSamples, \
        distMax = maxDist
    )
    IO.saveSettings(targetDirectory, pairwiseMISettings, 'pairwise')


    _, MI, degrees = infcy.runMI(model, nodes = np.array([node]), **pairwiseMISettings)
    MIs_pairwise = np.array([np.nanmean(MI[i,:,:], axis=1) for i in range(MI.shape[0])])
    print(MIs_pairwise)
    np.save(f'{targetDirectory}/MI_pairwise_T={model.t}.npy', MIs_pairwise[0])
    """

    nSnapshots = 100
    snapshotSettingsCond = dict( \
        nSamples    = nSnapshots, \
        burninSamples = mixingTime, \
        maxDist     = maxDist
    )
    IO.saveSettings(targetDirectory, snapshotSettingsCond, 'snapshots')

    snapshots, neighbours_idx = infcy.getSnapshotsPerDist2(model, node, allNeighbours_idx, **snapshotSettingsCond, threads=7)

    with open(f'{targetDirectory}/snapshots_node={node}_nSamples={nSnapshots}.pickle', 'wb') as f:
        pickle.dump(snapshots, f)
    with open(f'{targetDirectory}/neighboursG_node={node}.pickle', 'wb') as f:
        pickle.dump(allNeighbours_G, f)
    """
    with open(f'Data/1551775429.3397434/snapshots_node={node}_nSamples={nSnapshots}.pickle', 'rb') as f:
        snapshots = pickle.load(f)
    with open(f'Data/1551775429.3397434/neighboursG_node={node}.pickle', 'rb') as f:
        neighbours_G = pickle.load(f)
    """

    #maxDist = 10
    minDist = 1
    nTrials = 100
    nSamples = 100
    rep = 3
    MIruns = np.zeros((rep, maxDist))
    for i in range(rep):
        MIruns[i,:] = computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings)
    np.save(f'{targetDirectory}/MI_cond_T={model.t}_nSamples={nSamples*nTrials}_{rep}_repetitions.npy', np.array(MIruns))


    print(targetDirectory)

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
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph)

                #_, _, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                #          nSamples=nSamples, distSamples=distSamples_subgraph)
                MIs.append(MI)
    print(MIs)
    np.save(f'{targetDirectory}/MI_cond_T={model.t}_nSamples={nSamples*nTrials}.npy', np.array(MIs))



if __name__ == '__main__':

    # create data directory
    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)
    print(targetDirectory)

    # load network
    graph_path = "networkData/ER_avgDeg=1.5_N=1000.gpickle"
    graph = nx.read_gpickle(graph_path)
    N = len(graph)
    print(f'number of nodes = {N}')
    diameter = nx.diameter(graph)
    print("diameter = {}".format(diameter))

    maxDist = 15

    networkSettings = dict( \
        path = graph_path, \
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
        thresholdCorr   = 0.05
    )
    IO.saveSettings(targetDirectory, mixingTimeSettings, 'mixingTime')
    mixingTime, distSamples, mags = infcy.determineCorrTime(model, **mixingTimeSettings)
    print(f'correlation time = {distSamples}')
    print(f'mixing time      = {mixingTime}')
    #distSamples = 1000

    corrTimeSettings = dict( \
        nInitialConfigs = 10, \
        burninSteps  = mixingTime, \
        nStepsCorr      = int(1e4), \
        thresholdCorr   = 0.05, \
        checkMixing     = 0
    )
    IO.saveSettings(targetDirectory, corrTimeSettings, 'corrTime')

    #mixingTime2 = infcy.mixing2(model, nInitialConfigs=1000, nSteps=100, threshold = 0.005)
    #print(mixingTime2)
    #mixingTime2 = infcy.mixing2(model, nInitialConfigs=10, nSteps=100, threshold = 5e-4)
    #print(mixingTime2)

    node = list(graph)[0]
    print(node, graph.degree(node))

    snapshotSettingsJoint = dict( \
        nSamples    = int(1e3), \
        burninSamples = mixingTime, \
        maxDist     = maxDist, \
        nBins       = 50, \
        threshold   = 0.001
    )
    IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')

    jointSnapshots, avgSnapshots, Z2 = infcy.getJointSnapshotsPerDist(model, node, **snapshotSettingsJoint)
    print(f'Z2={Z2}')

    with open(f'{targetDirectory}/jointSnapshots_node={node}.pickle', 'wb') as f:
        pickle.dump(jointSnapshots, f)
    with open(f'{targetDirectory}/avgSnapshots_node={node}.pickle', 'wb') as f:
        pickle.dump(avgSnapshots, f)


    """
    with open(f'Data/jointSnapshots_node={node}.pickle', 'rb') as f:
        jointSnapshots = pickle.load(f)
    with open(f'Data/avgSnapshots_node={node}.pickle', 'rb') as f:
        avgSnapshots = pickle.load(f)

    Z2=6000.0
    """

    MIs = computeMI_joint(jointSnapshots, maxDist, Z2)
    np.save(f'{targetDirectory}/MI_joint_T={model.t}.npy', np.array(MIs))
    MIs_avg = computeMI_joint(avgSnapshots, maxDist, Z2)
    np.save(f'{targetDirectory}/MI_avg_T={model.t}.npy', np.array(MIs_avg))
    print(MIs)
    print(MIs_avg)
    """

    pairwiseMISettings = dict( \
        repeats    = 16, \
        burninSamples = mixingTime, \
        nSamples     = int(1e4), \
        distSamples   = distSamples, \
        distMax = maxDist
    )
    IO.saveSettings(targetDirectory, pairwiseMISettings, 'pairwise')
    """
    """
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

    snapshots, neighbours_G, neighbours_idx = infcy.getSnapshotsPerDist2(model, node, **snapshotSettingsCond)
    print(neighbours_G)

    with open(f'{targetDirectory}/snapshots_node={node}_nSamples={nSnapshots}.pickle', 'wb') as f:
        pickle.dump(snapshots, f)
    with open(f'{targetDirectory}/neighboursG_node={node}.pickle', 'wb') as f:
        pickle.dump(neighbours_G, f)
    """
    with open(f'Data/snapshots_node={node}_nSamples=200.pickle', 'rb') as f:
        snapshots = pickle.load(f)
    with open(f'Data/neighboursG_node={node}.pickle', 'rb') as f:
        neighbours_G = pickle.load(f)
    """
    # how many samples are needed to obtain stable estimate ?
    """
    states = list(snapshots[d-1].keys())[0]
    print(np.fromstring(states))
    model.fixedNodes = neighbours[d]
    conds = []
    for i in tqdm(range(10)):
        model.reset()
        model.seed += 1
        probCond = infcy.monteCarloFixedNeighboursSeq(model, states, node, \
                   neighbours[d], burninSamples=0, \
                   nSamples = int(1e3), distSamples = distSamples)
        conds.append(probCond[0])
        print(conds)
    print(f'stability of conditional probs: {np.mean(conds)}, {np.std(conds)}')
    """
    maxDist = 3
    minDist = 3
    nTrials = 100
    for nSamples in [int(1e3)]: #[int(1e2), int(1e3), int(1e4)]: #int(5e3), int(1e4), int(5e4)]:
        nSamples /= nTrials
        computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings)
    """
    for n in [int(5e3), int(1e4), int(5e4)]:
        MIs = []
        subgraph_nodes = [node]
        for d in range(1, maxDist+1):
            #sortedSnapshots = sorted(snapshots[d-1].items(), key=lambda x: x[1], reverse=True)
            #sortedSnapshots_counts = np.array([c for s,c in sortedSnapshots])
            #np.save(f'{targetDirectory}/snapshot_counts.npy', sortedSnapshots_counts)
            #print(f'num states: {len(snapshots[d-1])}')

            # get subgraph and outer neighbourhood at distance d
            if d in neighbours_G.keys():
                print(f'------------------- distance d={d} -----------------------')
                subgraph_nodes.extend(neighbours_G[d])
                subgraph = graph.subgraph(subgraph_nodes)
                model_subgraph = fastIsing.Ising(subgraph, **modelSettings)

                # determine correlation time for subgraph Ising model
                mixingTime_subgraph, distSamples_subgraph, _ = infcy.determineCorrTime(model_subgraph, **corrTimeSettings)
                print(f'correlation time = {distSamples_subgraph}')
                print(f'mixing time      = {mixingTime_subgraph}')

                #numSnapshots = 100
                #topSnapshots = sortedSnapshots[:numSnapshots]
                #topSnapshots = [sortedSnapshots[i] for i in np.random.choice(np.arange(len(sortedSnapshots)), numSnapshots)]
                #sum = np.sum([v for k,v in topSnapshots])
                #print(f'{sum} out of {Z1} states')
                #topSnapshots = {k: v/sum for k,v in topSnapshots}
                #topSnapshots = dict(sortedSnapshots)
                #print(topSnapshots.values())
                _, _, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                          nSamples=n, distSamples=distSamples_subgraph)
                #print(MI)
                MIs.append(MI)
        print(MIs)
        np.save(f'{targetDirectory}/MI_cond_T={model.t}_nSamples={n}.npy', np.array(MIs))
    """


    """
    MIs = np.zeros(maxDist)
    for d in range(1, maxDist+1):
        _, _, MI = infcy.runNeighbourhoodMI(model, node, neighbours[d], snapshots[d-1], \
                  nBurnin=distSamples, nSamples=int(1e3), distSamples=distSamples)
        MIs[d-1] = MI
    fig, ax = subplots(figsize=(8,5))
    ax.plot(range(1,maxDist+1), MIs, ls='--', marker='o')
    ax.set_xlabel('distance')
    ax.set_ylabel('MI')
    fig.savefig(f'{targetDirectory}/MIperDist.png')
    np.save(f'{targetDirectory}/MI_T={model.t}_{time.time()}.npy', MIs)
    """


    print(targetDirectory)
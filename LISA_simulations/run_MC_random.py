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
        P_XY = np.array(sorted([v/Z for s in jointSnapshots[d].keys() for v in jointSnapshots[d][s].values()]))
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
    graph_path = "networkData/ER_avgDeg=1.5_N=100.gpickle"
    graph = nx.read_gpickle(graph_path)
    N = len(graph)
    print(f'number of nodes = {N}')
    diameter = nx.diameter(graph)
    print("diameter = {}".format(diameter))

    maxDist = 10

    networkSettings = dict( \
        path = graph_path, \
        nNodes = N
    )


    # setup Ising model with nNodes spin flip attempts per simulation step
    # set temp to np.infty --> completely random
    modelSettings = dict( \
        temperature     = np.infty, \
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


    thresholds = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000001]
    allMIs = np.zeros((len(thresholds), maxDist))
    allZs = np.zeros(len(thresholds))

    for idx, t in enumerate(thresholds):

        snapshotSettingsJoint = dict( \
            nSamples    = int(1e2), \
            burninSamples = mixingTime, \
            maxDist     = maxDist, \
            nBins       = 500, \
            threshold   = t
        )
        IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')
        jointSnapshots, avgSnapshots, Z = infcy.getJointSnapshotsPerDist(model, node, **snapshotSettingsJoint)
        print(f'Z={Z}')
        allZs[idx] = Z
        with open(f'{targetDirectory}/jointSnapshots_node={node}_Z={Z}.pickle', 'wb') as f:
            pickle.dump(jointSnapshots, f)
        with open(f'{targetDirectory}/avgSnapshots_node={node}_Z={Z}.pickle', 'wb') as f:
            pickle.dump(avgSnapshots, f)

        #MIs = computeMI_joint(jointSnapshots, maxDist, Z)
        #np.save(f'{targetDirectory}/MI_joint_T=inf_Z={Z}.npy', np.array(MIs))
        MIs_avg = computeMI_joint(avgSnapshots, maxDist, Z)
        allMIs[idx,:] = MIs_avg
        #np.save(f'{targetDirectory}/MI_avg_T=inf_Z={Z}.npy', np.array(MIs_avg))
        #print(MIs)
        print(MIs_avg)
    np.save(f'{targetDirectory}/MI_avg_T=inf.npy', np.array(allMIs))
    np.save(f'{targetDirectory}/nSnapshots.npy', np.array(allZs))

        #bias = np.array([len(set(d[-1].keys()).union(set(d[1].keys()))) for d in jointSnapshots])/(2*Z*np.log(2))
        #print(MIs-bias)



    """
    for n in [100, 1000, 10000]:
        pairwiseMISettings = dict( \
            repeats    = 16, \
            burninSamples = mixingTime, \
            nSamples     = n, \
            distSamples   = distSamples, \
            distMax = maxDist
        )
        IO.saveSettings(targetDirectory, pairwiseMISettings, 'pairwise')


        _, MI, degrees = infcy.runMI(model, nodes = np.array([node]), **pairwiseMISettings)
        MIs_pairwise = np.array([np.nanmean(MI[i,:,:], axis=1) for i in range(MI.shape[0])])
        print(MIs_pairwise)
        np.save(f'{targetDirectory}/MI_pairwise_T=inf_nSamples={n}.npy', MIs_pairwise[0])

    """
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
    # how many samples are needed to obtain stable estimate ?
    maxDist = 5
    minDist = 5
    nTrials = 100
    for nSamples in [int(1e2), int(5e2), int(1e3), int(5e3), int(1e4)]: #int(5e3), int(1e4), int(5e4)]:
        nSamples /= nTrials
        computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings)
    """


    print(targetDirectory)

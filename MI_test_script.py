#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infcy
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
    for d in range(maxDist):
        P_XY = jointSnapshots[d].flatten()/Z
        P_X = np.sum(jointSnapshots[d], axis=1)/Z # sum over all bins
        P_Y = np.sum(jointSnapshots[d], axis=0)/Z # sum over all spin states

        MI = stats.entropy(P_X, base=2) + stats.entropy(P_Y, base=2) - stats.entropy(P_XY, base=2)
        MIs.append(MI)
    return MIs


def computeMI_cond(model, node, minDist, maxDist, neighboursG, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings):
    MIs = []
    subgraph_nodes = [node]
    for d in range(1, maxDist+1):
        # get subgraph and outer neighbourhood at distance d
        if d in neighboursG.keys():
            subgraph_nodes.extend(neighboursG[d])
            subgraph = graph.subgraph(subgraph_nodes)


            if d >= minDist:
                print(f'------------------- distance d={d}, num neighbours = {len(neighboursG[d])}, num states = {len(snapshots[d-1])}, size subgraph = {len(subgraph)} -----------------------')
                print(neighboursG[d])
                #print(list(subgraph))
                #print(subgraph_nodes)
                model_subgraph = fastIsing.Ising(subgraph, **modelSettings)
                #print(model_subgraph.mapping)
                # determine correlation time for subgraph Ising model
                mixingTime_subgraph, meanMag, distSamples_subgraph, _ = infcy.determineCorrTime(model_subgraph, **corrTimeSettings)
                print(f'correlation time = {distSamples_subgraph}')
                print(f'mixing time      = {mixingTime_subgraph}')
                distSamples_subgraph = min(distSamples_subgraph, 100)

                _, probs, MI, states = infcy.neighbourhoodMI(model_subgraph, node, neighboursG[d], snapshots[d-1], \
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, threads=7)
                #print(probs)
                np.save(f'{targetDirectory}/states_d={d}.npy', states)
                MIs.append(MI)
    return MIs



if __name__ == '__main__':

    # create data directory
    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)
    #print(targetDirectory)

    # load network
    graph_path = "networkData/ER_avgDeg=1.5_N=100.gpickle"
    graph_path = "networkData/2D_grid/2D_grid_L=60.gpickle"
    graph = nx.read_gpickle(graph_path)
    #graph = nx.path_graph(50)
    #graph = nx.DiGraph()
    #graph = nx.balanced_tree(2,3, create_using=graph)
    #graph.add_edge(2,4)
    #graph.remove_edge(1,4)
    #graph_path = 'balanced_tree'

    N = len(graph)
    print(f'number of nodes = {N}')
    #diameter = nx.diameter(graph)
    #print("diameter = {}".format(diameter))

    maxDist = 30

    networkSettings = dict( \
        path = graph_path, \
        nNodes = N
    )


    # setup Ising model with nNodes spin flip attempts per simulation step
    # set temp to np.infty --> completely random
    modelSettings = dict( \
        temperature     = 2.3, \
        updateType      = 'async' ,\
        magSide         = ''
    )
    #IO.saveSettings(targetDirectory, modelSettings, 'model')
    model = fastIsing.Ising(graph, **modelSettings)
    node = 1230 #list(graph)[0]
    nodeIdx = model.mapping[node]
    allNeighboursG, allNeighboursIdx = model.neighboursAtDist(node, maxDist)
    nNeighbours = np.array([len(allNeighboursG[d]) for d in sorted(allNeighboursG.keys())])


    # determine mixing/correlation time
    mixingTimeSettings = dict( \
        nInitialConfigs = 10, \
        burninSteps  = 10, \
        nStepsRegress   = int(1e3), \
        nStepsCorr      = int(1e4), \
        thresholdReg    = 0.1, \
        thresholdCorr   = 0.1
    )

    print('Determining mixing time...')
    mixingTime, meanMag, distSamples, mags = infcy.determineCorrTime(model, **mixingTimeSettings)
    print(f'correlation time = {distSamples}')
    print(f'mixing time      = {mixingTime}')
    print(f'magnetization    = {meanMag}')

    corrTimeSettings = dict( \
        nInitialConfigs = 10, \
        burninSteps  = mixingTime, \
        nStepsCorr      = int(1e4), \
        thresholdCorr   = 0.1, \
        checkMixing     = 0, \
        nodeG         = node
    )


    """
    avg neighbour spin MI approach
    """
    snapshotSettingsJoint = dict( \
        nSamples    = int(1e3), \
        repeats     = 10, \
        burninSamples = mixingTime, \
        distSamples   = distSamples, \
        maxDist     = maxDist, \
        nBins       = 100
    )

    """
    one node
    """

    #avgSnapshots, Z = infcy.getJointSnapshotsPerDist2(model, node, allNeighboursG, **snapshotSettingsJoint, threads=7)
    #MIs_avg = computeMI_joint(avgSnapshots, maxDist, Z)
    #print(MIs_avg)

    """
    all nodes
    """
    #avgSnapshotsAllNodes, Z, neighboursGAllNodes = infcy.getJointSnapshotsPerDistNodes(model, np.array(list(graph)), **snapshotSettingsJoint, threads=7)
    #for n in range(avgSnapshotsAllNodes.shape[0]):
#        MIs_avg = computeMI_joint(avgSnapshotsAllNodes[n], maxDist, Z)
#        print(MIs_avg)


    """
    fixed neighbours MI approach
    """

    nSnapshots = 100
    snapshotSettingsCond = dict( \
        nSamples    = nSnapshots, \
        burninSamples = mixingTime, \
        maxDist     = maxDist
    )
    minDist = 1
    maxDist = 16
    nTrials = 1
    nSamples = int(1e3)

    """
    for one node
    """
    snapshots, neighbours_idx, spins = infcy.getSnapshotsPerDist2(model, node, allNeighboursG, **snapshotSettingsCond, threads=7)
    np.save(f'{targetDirectory}/states.npy', spins)
    MIs = computeMI_cond(model, node, minDist, maxDist, allNeighboursG, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings)
    print(MIs)

    """
    for all nodes
    """
    """
    allNodes = np.array(list(graph))
    snapshotsAllNodes, neighboursGAllNodes = infcy.getSnapshotsPerDistNodes(model, allNodes, **snapshotSettingsCond, threads=7)
    for n in range(len(snapshotsAllNodes)):
        node = allNodes[n]
        corrTimeSettings = dict( \
            nInitialConfigs = 10, \
            burninSteps  = mixingTime, \
            nStepsCorr      = int(1e4), \
            thresholdCorr   = 0.1, \
            checkMixing     = 0, \
            nodeG         = node
        )
        MIs = computeMI_cond(model, node, minDist, maxDist, neighboursGAllNodes[n], snapshotsAllNodes[n], nTrials, nSamples, modelSettings, corrTimeSettings)
        print(MIs)
    """

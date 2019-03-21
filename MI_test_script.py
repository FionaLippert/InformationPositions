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
    for d in tqdm(range(maxDist)):
        P_XY = np.array(sorted([v/Z for s in jointSnapshots[d].keys() for v in jointSnapshots[d][s].values()]))
        P_X = np.array([sum(list(dict_s.values()))/Z for dict_s in jointSnapshots[d].values()])
        all_keys = set.union(*[set(dict_s.keys()) for dict_s in jointSnapshots[d].values()])
        P_Y = np.array([jointSnapshots[d][1][k]/Z if k in jointSnapshots[d][1] else 0 for k in all_keys]) + \
                np.array([jointSnapshots[d][-1][k]/Z if k in jointSnapshots[d][-1] else 0 for k in all_keys])

        MI = stats.entropy(P_X, base=2) + stats.entropy(P_Y, base=2) - stats.entropy(P_XY, base=2)
        MIs.append(MI)

        state = list(all_keys)[0]
        #print(state)
        #if type(state) == bytes:
        #    print(np.fromstring(state))
        #    print(jointSnapshots[d][1][state]/Z if state in jointSnapshots[d][1] else 0, jointSnapshots[d][-1][state]/Z if state in jointSnapshots[d][-1] else 0)
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
                print(f'------------------- distance d={d}, num neighbours = {len(neighbours_G[d])}, num states = {len(snapshots[d-1])}, size subgraph = {len(subgraph)} -----------------------')
                model_subgraph = fastIsing.Ising(subgraph, **modelSettings)
                # determine correlation time for subgraph Ising model
                mixingTime_subgraph, meanMag, distSamples_subgraph, _ = infcy.determineCorrTime(model_subgraph, **corrTimeSettings)
                #distSamples_subgraph = 1000
                print(f'correlation time = {distSamples_subgraph}')
                print(f'mixing time      = {mixingTime_subgraph}')
                #distSamples_subgraph = distSamples_subgraph

                _, _, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, threads=7)

                #_, _, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                #          nSamples=nSamples, distSamples=distSamples_subgraph)
                MIs.append(MI)
                #np.save(f'{targetDirectory}/MI_cond_T={model.t}_nSamples={nSamples*nTrials}.npy', np.array(MIs))
    print(MIs)
    #np.save(f'{targetDirectory}/MI_cond_T={model.t}_nSamples={nSamples*nTrials}.npy', np.array(MIs))



if __name__ == '__main__':

    # create data directory
    #now = time.time()
    #targetDirectory = f'{os.getcwd()}/Data/{now}'
    #os.mkdir(targetDirectory)
    #print(targetDirectory)

    # load network
    graph_path = "networkData/ER_avgDeg=1.5_N=100.gpickle"
    graph = nx.read_gpickle(graph_path)
    #graph = nx.DiGraph()
    #graph = nx.balanced_tree(2,3, create_using=graph)
    #graph.add_edge(2,4)
    #graph.remove_edge(1,4)
    #graph_path = 'balanced_tree'

    N = len(graph)
    print(f'number of nodes = {N}')
    #diameter = nx.diameter(graph)
    #print("diameter = {}".format(diameter))

    maxDist = 2

    networkSettings = dict( \
        path = graph_path, \
        nNodes = N
    )

    node = list(graph)[0]


    # setup Ising model with nNodes spin flip attempts per simulation step
    # set temp to np.infty --> completely random
    modelSettings = dict( \
        temperature     = 2.0, \
        updateType      = 'async' ,\
        magSide         = ''
    )
    #IO.saveSettings(targetDirectory, modelSettings, 'model')
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
    #IO.saveSettings(targetDirectory, mixingTimeSettings, 'mixingTime')
    mixingTime, meanMag, distSamples, mags = infcy.determineCorrTime(model, **mixingTimeSettings)
    print(f'correlation time = {distSamples}')
    print(f'mixing time      = {mixingTime}')
    print(f'magnetization    = {meanMag}')

    #for key, values in mags.items():
    #    np.save(f'{targetDirectory}/magSeries_{key}.npy', np.array(values))


    corrTimeSettings = dict( \
        nInitialConfigs = 50, \
        burninSteps  = mixingTime, \
        nStepsCorr      = int(1e4), \
        thresholdCorr   = 0.01, \
        checkMixing     = 0, \
        node            = node
    )
    #IO.saveSettings(targetDirectory, corrTimeSettings, 'corrTime')


    snapshotSettingsJoint = dict( \
        nSamples    = int(1e3), \
        burninSamples = mixingTime, \
        maxDist     = maxDist, \
        nBins       = 100, \
        threshold   = 0.01
    )
    #IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')

    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(model.mapping[node], maxDist)
    nNeighbours = np.array([len(allNeighbours_G[d]) for d in sorted(allNeighbours_G.keys())])
    print(nNeighbours)
    #np.save(f'{targetDirectory}/nNeighbours.npy', nNeighbours)


    jointSnapshots, avgSnapshots, Z = infcy.getJointSnapshotsPerDist(model, node, allNeighbours_idx, **snapshotSettingsJoint, threads=7)
    print(f'Z={Z}')


    MIs = computeMI_joint(jointSnapshots, maxDist, Z)
    #np.save(f'{targetDirectory}/MI_joint_T={model.t}.npy', np.array(MIs))
    MIs_avg = computeMI_joint(avgSnapshots, maxDist, Z)
    #np.save(f'{targetDirectory}/MI_avg_T={model.t}.npy', np.array(MIs_avg))
    print(MIs)
    print(MIs_avg)



    nSnapshots = 200
    snapshotSettingsCond = dict( \
        nSamples    = nSnapshots, \
        burninSamples = mixingTime, \
        maxDist     = maxDist
    )
    #IO.saveSettings(targetDirectory, snapshotSettingsCond, 'snapshots')

    snapshots, neighbours_idx = infcy.getSnapshotsPerDist2(model, node, allNeighbours_idx, **snapshotSettingsCond, threads=7)

    #maxDist = 20
    minDist = 1
    nTrials = 100
    nSamples = int(1e3)
    nSamples /= nTrials
    computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infcy
from Utils import IO
import networkx as nx, itertools, scipy, time, \
        os, pickle, sys, multiprocessing as mp
import itertools
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from scipy import stats


nthreads = mp.cpu_count() - 1 # leave one thread for coordination tasks
#nthreads = 1



def computeMI_cond(model, node, dist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings, distSamples):
    MIs = []
    corrTimes = []
    subgraph_nodes = [node]
    for d in range(1, dist+1):
        # get subgraph and outer neighbourhood at distance d
        if d in neighbours_G.keys():
            subgraph_nodes.extend(neighbours_G[d])
            subgraph = graph.subgraph(subgraph_nodes)


    #for r in range(reps):
    print(f'------------------- distance d={d}, num neighbours = {len(neighbours_G[d])}, num states = {len(snapshots[d-1])} -----------------------')
    model_subgraph = fastIsing.Ising(subgraph, **modelSettings)
    model_subgraph.reset()
    print(f'mean mag = {np.mean(model_subgraph.states)}')

    threads = nthreads if len(subgraph_nodes) > 20 or distSamples > 100 else 1

    snapshotsDict, pCond, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[dist], snapshots[dist-1], \
              nTrials=nTrials, burninSamples=corrTimeSettings['burninSteps'], nSamples=nSamples, distSamples=distSamples, threads=nthreads)

    return MI


def computeMI_joint(jointSnapshots, d, Z):
    P_XY = np.array(sorted([v/Z for s in jointSnapshots[d].keys() for v in jointSnapshots[d][s].values()]))
    P_X = np.array([sum(list(dict_s.values()))/Z for dict_s in jointSnapshots[d].values()])
    all_keys = set.union(*[set(dict_s.keys()) for dict_s in jointSnapshots[d].values()])
    P_Y = np.array([jointSnapshots[d][1][k]/Z if k in jointSnapshots[d][1] else 0 for k in all_keys]) + \
            np.array([jointSnapshots[d][-1][k]/Z if k in jointSnapshots[d][-1] else 0 for k in all_keys])

    MI = stats.entropy(P_X, base=2) + stats.entropy(P_Y, base=2) - stats.entropy(P_XY, base=2)
    return MI



if __name__ == '__main__':

    start = timer()

    # create data directory
    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)
    print(targetDirectory)


    # load network
    z = 2
    maxDist = 5
    #graph = nx.DiGraph()
    #graph = nx.balanced_tree(z,maxDist, create_using=graph)
    graph = nx.balanced_tree(2,6)
    path = f'nx.balanced_tree({z},{maxDist})'

    #path = f'{os.getcwd()}/networkData/ER_k=2.5_N=100.gpickle'
    #graph = nx.read_gpickle(path)


    N = len(graph)


    #theory = np.zeros(maxDist)
    #for d in tqdm(range(maxDist)):
    #    theory[d] = infcy.MI_tree_theory(d, 1.0, 2)
    #    print(theory[d])
    #np.save(f'{targetDirectory}/MI_bin_tree_theory.npy', theory)




    networkSettings = dict( \
        path = path, \
        nNodes = N
    )


    # setup Ising model with nNodes spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = 0.7, \
        updateType      = 'async' ,\
        magSide         = ''
    )
    IO.saveSettings(targetDirectory, modelSettings, 'model')
    model = fastIsing.Ising(graph, **modelSettings)

    # central node and its neighbour shells
    node = list(graph)[0]
    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(model.mapping[node], maxDist)


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
    print(f'mean mags        = {meanMag}')

    for key, values in mags.items():
        np.save(f'{targetDirectory}/magSeries_{key}.npy', np.array(values))


    corrTimeSettings = dict( \
        nInitialConfigs = 10, \
        burninSteps  = mixingTime, \
        nStepsCorr      = int(1e4), \
        thresholdCorr   = 0.01, \
        checkMixing     = 0, \
        node            = node
    )
    IO.saveSettings(targetDirectory, corrTimeSettings, 'corrTime')


    nBins = np.linspace(2,100,20).astype(int)
    np.save(f'{targetDirectory}/nBins.npy', nBins)

    mixingTime = min(mixingTime, 5000)
    distSamples = min(distSamples, 100)


    #for i, nBins in enumerate(nBins_range):
    #    print(f'------------- nBins = {nBins} -------------')
    # 1e4, 100
    snapshotSettingsJoint = dict( \
        nSamples    = int(1e2), \
        repeats     = 100, \
        burninSamples = mixingTime, \
        distSamples   = distSamples, \
        maxDist     = maxDist
    )
    IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')

    reps = 1
    MIs = np.zeros((reps, nBins.size))
    for r in range(reps):

        avgSnapshots, Z = infcy.getJointSnapshotsPerDistNodes(model, np.array([0,1,2]), **snapshotSettingsJoint, nBins=10, threads = nthreads)
        avgSnapshots, Z = infcy.getJointSnapshotsPerDistBins(model, node, allNeighbours_idx, **snapshotSettingsJoint, nBins=nBins, threads = nthreads)
        #Z = snapshotSettingsJoint['nSamples'] * snapshotSettingsJoint['repeats']

        #print(avgSnapshots)

        MIs_avg = [computeMI_joint(avgSnapshots[bins], maxDist-1, Z) for bins in nBins]
        print(MIs_avg)
        MIs[r] = np.array(MIs_avg)

    np.save(f'{targetDirectory}/MI_avg_T={model.t}.npy', MIs)

    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)

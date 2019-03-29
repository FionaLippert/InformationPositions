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



def computeMI_cond(model, node, dist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings, threshold):
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

    mixingTime, meanMag, distSamples, mags = infcy.determineCorrTime(model, **corrTimeSettings, thresholdCorr=threshold)
    print(f'distSamples = {distSamples}')

    threads = nthreads if len(subgraph_nodes) > 20 or distSamples > 100 else 1

    snapshotsDict, pCond, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[dist], snapshots[dist-1], \
              nTrials=nTrials, burninSamples=corrTimeSettings['burninSteps'], nSamples=nSamples, distSamples=distSamples, threads=nthreads)

    return pCond



if __name__ == '__main__':

    start = timer()

    # create data directory
    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)
    print(targetDirectory)


    # load network
    z = 2
    maxDist = 6
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
        temperature     = 1.0, \
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
        thresholdReg    = 0.1, \
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
        checkMixing     = 0, \
        node            = node
    )
    IO.saveSettings(targetDirectory, corrTimeSettings, 'corrTime')


    # collect neighbourhood snapshots
    nSnapshots = 50
    snapshotSettingsCond = dict( \
        nSamples    = nSnapshots, \
        burninSamples = mixingTime, \
        maxDist     = maxDist
    )
    IO.saveSettings(targetDirectory, snapshotSettingsCond, 'snapshots')




    snapshots, _ = infcy.getSnapshotsPerDist2(model, node, allNeighbours_idx, **snapshotSettingsCond, threads=nthreads)

    #state = list(snapshots[maxDist-1].keys())[0]
    #mixingTime, meanMag, distSamples, _ = infcy.determineCorrTime(model, **corrTimeSettings)
    #print(f'correlation time = {distSamples}')
    #print(f'mixing time      = {mixingTime}')

    nTrials = 1 # 10
    nSamples = 1000
    reps = 10

    thresholds = np.linspace(0.9, 0.01, 10)
    np.save(f'{targetDirectory}/corrThresholds.npy', np.array(thresholds))

    #for dist in [1, 2, 3, 4]:
    for dist in [5,6]:

        numStates = len(snapshots[dist-1])

        #distances = np.logspace(0,3,13).astype(int)

        p = np.zeros((thresholds.size, reps, numStates))

        for i, t in enumerate(thresholds):
            print(f'------------- threshold = {t} -------------')

            for rep in range(reps):
                #snapshots, _ = infcy.getSnapshotsPerDist2(model, node, allNeighbours_idx, **snapshotSettingsCond, threads=nthreads)
                p[i, rep,:] = computeMI_cond(model, node, dist, allNeighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings, t)[:,0]
            np.save(f'{targetDirectory}/p_cond_T={model.t}_d={dist}.npy', np.array(p))

    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)

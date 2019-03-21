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
nthreads = 1



def computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings, rep=1):
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

                snapshotsDict, pCond, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, threads=nthreads)

                MIs.append(MI)
                np.save(f'{targetDirectory}/snapshots_nSamples={nSamples*nTrials}_d={d}_rep={rep}.npy', np.array(list(snapshotsDict.keys())))
                np.save(f'{targetDirectory}/pCond_nSamples={nSamples*nTrials}_d={d}_rep={rep}.npy', pCond)
    print(MIs)
    np.save(f'{targetDirectory}/MI_cond_T={model.t}_nSamples={nSamples*nTrials}_rep={rep}.npy', np.array(MIs))
    return MIs



if __name__ == '__main__':

    start = timer()

    # create data directory
    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)
    print(targetDirectory)


    # load network
    z = 2
    maxDist = 3
    graph = nx.DiGraph()
    graph = nx.balanced_tree(z,maxDist, create_using=graph)
    #graph = nx.balanced_tree(2,6)
    N = len(graph)


    #theory = np.zeros(maxDist)
    #for d in tqdm(range(maxDist)):
    #    theory[d] = infcy.MI_tree_theory(d, 1.0, 2)
    #    print(theory[d])
    #np.save(f'{targetDirectory}/MI_bin_tree_theory.npy', theory)

    path = f'nx.balanced_tree({z},{maxDist})'

    networkSettings = dict( \
        path = path, \
        nNodes = N
    )


    # setup Ising model with nNodes spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = np.infty, \
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


    # collect neighbourhood snapshots
    nSnapshots = 1
    snapshotSettingsCond = dict( \
        nSamples    = nSnapshots, \
        burninSamples = mixingTime, \
        maxDist     = maxDist
    )
    IO.saveSettings(targetDirectory, snapshotSettingsCond, 'snapshots')

    #with open(f'{targetDirectory}/neighboursG_node={node}.pickle', 'wb') as f:
    #    pickle.dump(allNeighbours_G, f)

    """
    minDist = 1
    maxDist = 3
    nTrials = 10
    nSamples = 100
    rep = 5
    #MIruns = np.zeros((rep, maxDist-minDist+1))
    for i in range(rep):
        snapshots, _ = infcy.getSnapshotsPerDist2(model, node, allNeighbours_idx, **snapshotSettingsCond, threads=nthreads)

        #with open(f'{targetDirectory}/snapshots_node={node}_nSamples={nSnapshots}_{i}.pickle', 'wb') as f:
        #    pickle.dump(snapshots, f)

        for nSamples in np.logspace(0, 4, 13).astype(int):
            computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings, i)
    #np.save(f'{targetDirectory}/MI_cond_T={model.t}_nSamples={nSamples*nTrials}_{rep}_repetitions.npy', np.array(MIruns))
    """


    snapshots, _ = infcy.getSnapshotsPerDist2(model, node, allNeighbours_idx, **snapshotSettingsCond, threads=nthreads)
    state = list(snapshots[maxDist-1].keys())[0]
    mixingTime, meanMag, distSamples, _ = infcy.determineCorrTime(model, **corrTimeSettings)
    print(f'correlation time = {distSamples}')
    print(f'mixing time      = {mixingTime}')

    nTrials = 10
    rep = 100

    for nSamples in np.logspace(0, 4, 13).astype(int):
        print(f'------------- {nSamples} samples -------------')
        allProbs = np.zeros(rep)
        for i in tqdm(range(rep)):
            probs = infcy.monteCarloFixedNeighboursSeq(model, state, node, \
                           allNeighbours_idx[maxDist], nTrials, mixingTime, \
                           nSamples, distSamples)
            allProbs[i] = probs[0]
        np.save(f'{targetDirectory}/pCond_nSamples={nSamples*nTrials}_T=inf_rep={rep}.npy', allProbs)

    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)

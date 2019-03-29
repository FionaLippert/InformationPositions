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

nthreads = mp.cpu_count() - 1
#nthreads = 1


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




if __name__ == '__main__':

    start = timer()

    T = float(sys.argv[1])
    targetDirectory = sys.argv[2]
    path = sys.argv[3]

    # load network
    #avg_deg = 1.5
    maxDist = 20
    #N = 500
    #p = avg_deg/N
    #graph = nx.erdos_renyi_graph(N, p)
    #connected_nodes = max(nx.connected_components(graph), key=len)
    #graph = graph.subgraph(connected_nodes)
    #nx.write_gpickle(graph, f'{targetDirectory}/ER_avgDeg={avg_deg}_N={N}.gpickle', 2)
    graph = nx.read_gpickle(path)
    N = len(graph)

    #path = f'nx.erdos_renyi_graph({N},{p})'
    
    node = list(graph)[0]
    deg = graph.degree(node)

    networkSettings = dict( \
        path = path, \
        nNodes = N, \
        node = node, \
        degree = deg
    )
    IO.saveSettings(targetDirectory, networkSettings, 'network')


    # setup Ising model with nNodes spin flip attempts per simulation step
    # set temp to np.infty --> completely random
    modelSettings = dict( \
        temperature     = T, \
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
    print(f'mag level        = {meanMag}')
    #distSamples = 100

    mixingTime = min(mixingTime, 10000)

    mixingResults = dict(\
        mixingTime = mixingTime, \
        corrTime = distSamples, \
        magLevel = meanMag
    )
    IO.saveSettings(targetDirectory, mixingResults, 'mixingResults')

    #for key, values in mags.items():
    #    np.save(f'{targetDirectory}/magSeries_{key}.npy', np.array(values))


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

    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(model.mapping[node], maxDist)
    
    
    snapshotSettingsJoint = dict( \
        nSamples    = int(1e3), \
        burninSamples = mixingTime, \
        maxDist     = maxDist, \
        nBins       = 50, \
        threshold   = 0.001
    )
    IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')

    for r in range(10):

        jointSnapshots, avgSnapshots, Z = infcy.getJointSnapshotsPerDist(model, node, allNeighbours_idx, **snapshotSettingsJoint, threads=nthreads)
        print(f'Z={Z}')

        #with open(f'{targetDirectory}/jointSnapshots_node={node}.pickle', 'wb') as f:
        #    pickle.dump(jointSnapshots, f)
        #with open(f'{targetDirectory}/avgSnapshots_node={node}.pickle', 'wb') as f:
        #    pickle.dump(avgSnapshots, f)
    
    
        MIs = computeMI_joint(jointSnapshots, maxDist, Z)
        np.save(f'{targetDirectory}/MI_joint_T={model.t}_rep={r}.npy', np.array(MIs))
        MIs_avg = computeMI_joint(avgSnapshots, maxDist, Z)
        np.save(f'{targetDirectory}/MI_avg_T={model.t}_rep={r}.npy', np.array(MIs_avg))
        print(MIs)
        print(MIs_avg)
    

    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infcy
from Utils import IO
import networkx as nx, itertools, scipy, time, \
                os, pickle, sys, argparse, multiprocessing as mp
import itertools
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from scipy import stats

nthreads = mp.cpu_count() - 1
#nthreads = 1


parser = argparse.ArgumentParser(description='run MC chain and compute MI based on conditional PDF of the central node with neighbour states fixed')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('node', type=int, help='central node ID')
parser.add_argument('maxDist', type=int, help='max distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')
parser.add_argument('--maxCorrTime', type=int, default=-1, help='max distance between two samples in the MC')
parser.add_argument('--snapshots', type=int, default=100, help='number of neighbourhood snapshots')




def computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings, maxCorrTime=-1):
    MIs = []
    subgraph_nodes = [node]
    for d in range(1, maxDist+1):
        # get subgraph and outer neighbourhood at distance d
        if d in neighbours_G.keys():
            subgraph_nodes.extend(neighbours_G[d])
            subgraph = graph.subgraph(subgraph_nodes)

            if d >= minDist:
                print(f'------------------- distance d={d}, num neighbours = {len(neighbours_G[d])}, subgraph size = {len(subgraph_nodes)}, num states = {len(snapshots[d-1])} -----------------------')
                model_subgraph = fastIsing.Ising(subgraph, **modelSettings)
                # determine correlation time for subgraph Ising model
                mixingTime_subgraph, meanMag, distSamples_subgraph, _ = infcy.determineCorrTime(model_subgraph, nodeG=node, **corrTimeSettings)
                #distSamples_subgraph = max(distSamples_subgraph, 10)
                if maxCorrTime > 0: distSamples_subgraph = min(distSamples_subgraph, maxCorrTime)
                print(f'correlation time = {distSamples_subgraph}')
                print(f'mixing time      = {mixingTime_subgraph}')

                threads = nthreads if len(subgraph_nodes) > 20 else 1

                _, _, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, threads=threads)

                #_, _, MI = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], \
                #          nSamples=nSamples, distSamples=distSamples_subgraph)
                MIs.append(MI)
    print(MIs)
    return MIs



if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    maxDist = args.maxDist

    # load network
    #z = 2
    #maxDist = 8
    #subtrees = [(nx.balanced_tree(z,maxDist-1), 0) for _ in range(z+1)]
    #graph = nx.join(subtrees)
    #graph = nx.balanced_tree(z, maxDist)
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    node = args.node
    deg = graph.degree[node]

    #path = f'nx.balanced_tree({z},{maxDist})'
    #path = 'nx.path_graph'

    networkSettings = dict( \
        path = args.graph, \
        nNodes = N, \
        node = node, \
        degree = deg
    )


    # setup Ising model with nNodes spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = ''
    )
    IO.saveSettings(targetDirectory, modelSettings, 'model')
    model = fastIsing.Ising(graph, **modelSettings)

    try:
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']

    except:
        raise Exception('No mixing results found! Please run the mixing script first to determine the mixing time of the model.')

        """
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
    

        mixingResults = dict(\
            mixingTime = mixingTime, \
            corrTime = distSamples, \
            magLevel = meanMag
        )
        IO.saveResults(targetDirectory, mixingResults, 'mixingResults')

        mixingTime = min(mixingTime, 5000)


        corrTimeSettings = dict( \
            nInitialConfigs = 10, \
            burninSteps  = mixingTime, \
            nStepsCorr      = int(1e4), \
            thresholdCorr   = 0.01, \
            checkMixing     = 0
        )
        IO.saveSettings(targetDirectory, corrTimeSettings, 'corrTime')
        """


    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)
    
    
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

    nSnapshots = args.snapshots
    snapshotSettingsCond = dict( \
        nSamples    = nSnapshots, \
        burninSamples = burninSteps, \
        maxDist     = maxDist
    )
    IO.saveSettings(targetDirectory, snapshotSettingsCond, 'snapshots')

    #snapshots, neighbours_idx = infcy.getSnapshotsPerDist2(model, node, allNeighbours_G, **snapshotSettingsCond, threads=nthreads)

    #with open(f'{targetDirectory}/snapshots_node={node}_nSamples={nSnapshots}.pickle', 'wb') as f:
    #    pickle.dump(snapshots, f)
    with open(f'{targetDirectory}/neighboursG_node={node}.pickle', 'wb') as f:
        pickle.dump(allNeighbours_G, f)
    """
    with open(f'Data/1551775429.3397434/snapshots_node={node}_nSamples={nSnapshots}.pickle', 'rb') as f:
        snapshots = pickle.load(f)
    with open(f'Data/1551775429.3397434/neighboursG_node={node}.pickle', 'rb') as f:
        neighbours_G = pickle.load(f)
    """


    minDist = 1
    nTrials = 10
    nSamples = 1000
   
    #result_dir = f'{targetDirectory}/MI_cond'
    #if not os.path.isdir(result_dir):
    #    os.mkdir(result_dir)

    for i in range(args.runs):
        threads = nthreads if len(model.graph) > 20 else 1
        snapshots, _ = infcy.getSnapshotsPerDist2(model, node, allNeighbours_G, **snapshotSettingsCond, threads=threads)
        #with open(f'{targetDirectory}/snapshots_node={node}_nSamples={nSnapshots}_{i}.pickle', 'wb') as f:
        #    pickle.dump(snapshots, f)
        MI = computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings, maxCorrTime=args.maxCorrTime)

        now = time.time()
        np.save(f'{targetDirectory}/MI_cond_{now}.npy', np.array(MI))

    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)

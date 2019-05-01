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
parser.add_argument('--minDist', type=int, default=1, help='min distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')
parser.add_argument('--maxCorrTime', type=int, default=-1, help='max distance between two samples in the MC')
parser.add_argument('--minCorrTime', type=int, default=1, help='min distance between two samples in the MC')
parser.add_argument('--snapshots', type=int, default=100, help='number of neighbourhood snapshots')
parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate MI')
parser.add_argument('--numSamples', type=int, default=1000, help='number of samples per MC run with fixed neighbour states')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=1, help='initial system state')
parser.add_argument('--getStates', type=int, default=0, help='get system states instead of MI')




def computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, spinProbs, nTrials, nSamples, modelSettings, corrTimeSettings):
    MIs = []
    HXs = []
    all_states = {}
    all_mappings = {}
    subgraph_nodes = [node]
    for d in range(1, maxDist+1):
        # get subgraph and outer neighbourhood at distance d
        if d in neighbours_G.keys():
            subgraph_nodes.extend(neighbours_G[d])
            subgraph = graph.subgraph(subgraph_nodes)

            if d >= minDist:
                print(f'------------------- distance d={d}, num neighbours = {len(neighbours_G[d])}, subgraph size = {len(subgraph_nodes)}, num states = {len(snapshots[d-1])} -----------------------')

                model_subgraph = fastIsing.Ising(subgraph, **modelSettings)
                all_mappings[d] = model_subgraph.mapping

                # determine correlation time for subgraph Ising model
                if args.maxCorrTime == args.minCorrTime:
                    distSamples_subgraph = args.maxCorrTime
                    mixingTime_subgraph = corrTimeSettings['burninSteps']
                else:
                    mixingTime_subgraph, meanMag, distSamples_subgraph, _ = infcy.determineCorrTime(model_subgraph, nodeG=node, **corrTimeSettings)
                    if args.maxCorrTime > 0: distSamples_subgraph = min(distSamples_subgraph, args.maxCorrTime)
                    distSamples_subgraph = max(distSamples_subgraph, args.minCorrTime)
                print(f'correlation time = {distSamples_subgraph}')
                print(f'mixing time      = {mixingTime_subgraph}')

                threads = nthreads if len(subgraph_nodes) > 20 else 1

                if args.getStates:
                    _, states = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], spinProbs, \
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, \
                          threads=threads, initStateIdx=args.initState, getStates=1)
                    all_states[d] = states

                else:
                    _, _, MI, HX = infcy.neighbourhoodMI(model_subgraph, node, neighbours_G[d], snapshots[d-1], spinProbs, \
                              nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, threads=threads, initStateIdx=args.initState)

                    MIs.append(MI)
                    HXs.append(HX)
                    print(MIs)

    if args.getStates:
        return all_states, all_mappings
    else:
        return MIs, HXs



if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir


    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    node = args.node
    deg = graph.degree[node]

    if args.maxDist > 0:
        maxDist = args.maxDist
    else:
        maxDist = nx.diameter(graph)

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
        magSide         = args.magSide if args.magSide in ['pos', 'neg'] else ''
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

    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)

    nSnapshots = args.snapshots
    snapshotSettingsCond = dict( \
        nSamples    = nSnapshots, \
        burninSamples = burninSteps, \
        maxDist     = maxDist
    )
    IO.saveSettings(targetDirectory, snapshotSettingsCond, 'snapshots')

    modelSettingsCond = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = ''
    )


    with open(f'{targetDirectory}/neighboursG_node={node}.pickle', 'wb') as f:
        pickle.dump(allNeighbours_G, f)

    minDist = args.minDist
    nTrials = args.repeats
    nSamples = args.numSamples


    for i in range(args.runs):
        now = time.time()

        threads = nthreads if len(model.graph) > 20 else 1
        snapshots, _ , system_states, spinProbs = infcy.getSnapshotsPerDist2(model, node, allNeighbours_G, **snapshotSettingsCond, threads=threads, initStateIdx=args.initState)
        with open(f'{targetDirectory}/snapshots_{now}.pickle', 'wb') as f:
            pickle.dump(snapshots, f)

        #np.save(f'{targetDirectory}/system_states_{now}.npy', system_states)

        if args.getStates:
            states, mappings = computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, spinProbs, nTrials, nSamples, modelSettingsCond, corrTimeSettings)
            with open(f'{targetDirectory}/subsystem_states_{now}.pickle', 'wb') as f:
                pickle.dump(states, f)
            with open(f'{targetDirectory}/subsystem_mappings_{now}.pickle', 'wb') as f:
                pickle.dump(mappings, f)
        else:
            MI, HX = computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, spinProbs, nTrials, nSamples, modelSettingsCond, corrTimeSettings)
            np.save(f'{targetDirectory}/MI_cond_{now}.npy', np.array(MI))
            np.save(f'{targetDirectory}/HX_{now}.npy', np.array(HX))

    print(f'time elapsed: {timer()-start : .2f} seconds')

    #print(targetDirectory)

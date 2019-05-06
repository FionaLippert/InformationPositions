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
#parser.add_argument('nodes', type=str, help='path to nodes for which MI should be computed')
parser.add_argument('maxDist', type=int, help='max distance to central node')
parser.add_argument('--minDist', type=int, default=1, help='min distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')
parser.add_argument('--maxCorrTime', type=int, default=-1, help='max distance between two samples in the MC')
parser.add_argument('--minCorrTime', type=int, default=1, help='min distance between two samples in the MC')
parser.add_argument('--snapshots', type=int, default=100, help='number of neighbourhood snapshots')
parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate MI')
parser.add_argument('--numSamples', type=int, default=1000, help='number of samples per MC run with fixed neighbour states')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=-1, help='initial system state')
parser.add_argument('--uniformPDF', action="store_true", help='assume uniform distribution over neighbourhood snapshots')




def computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings):
    MIs             = []
    HXs             = []
    all_HXgiveny    = []
    all_keys        = []

    subgraph_nodes = [node]
    for d in range(minDist, maxDist+1):
        # get subgraph and outer neighbourhood at distance d
        if d in neighbours_G.keys():
            #subgraph_nodes.extend(neighbours_G[d])
            #subgraph = graph.subgraph(subgraph_nodes)
            #print(subgraph.edges())
            subgraph = nx.ego_graph(model.graph, node, d)

            print(f'------------------- distance d={d}, num neighbours = {len(neighbours_G[d])}, subgraph size = {len(subgraph)}, num states = {len(snapshots[d-1])} -----------------------')

            model_subgraph = fastIsing.Ising(subgraph, **modelSettings)

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


            threads = nthreads if len(subgraph) > 20 else 1

            _, _, MI, HX, HXgiveny, keys, probs = infcy.neighbourhoodMI(model_subgraph, node, \
                            d, neighbours_G, snapshots[d-1], nTrials=nTrials, \
                            burninSamples=mixingTime_subgraph, nSamples=nSamples, \
                            distSamples=distSamples_subgraph, threads=threads, \
                            initStateIdx=args.initState, uniformPDF=args.uniformPDF, out='MI')

            MIs.append(MI)
            HXs.append(HX)
            all_keys.append(keys)
            all_HXgiveny.append(HXgiveny)

    return MIs, HXs, all_HXgiveny, all_keys



if __name__ == '__main__':

    start = timer()
    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    #nodes = np.load(args.nodes)
    nodes = np.array([283, 642])

    if args.maxDist > 0:
        maxDist = args.maxDist
    else:
        maxDist = nx.diameter(graph)

    networkSettings = dict( \
        path = args.graph, \
        nNodes = N, \
        nodes = list(nodes)
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
        raise Exception('No mixing results found! Please run the mixing script \
                        first to determine the mixing time of the model.')


    snapshotSettingsCond = dict( \
        nSamples    = args.snapshots, \
        burninSamples = burninSteps, \
        maxDist     = maxDist
    )
    IO.saveSettings(targetDirectory, snapshotSettingsCond, 'snapshots')

    modelSettingsCond = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = ''
    )


    #with open(f'{targetDirectory}/neighboursG_node={node}.pickle', 'wb') as f:
    #    pickle.dump(allNeighbours_G, f)

    minDist = args.minDist
    nTrials = args.repeats
    nSamples = args.numSamples


    for i in range(args.runs):

        all_MIs = {}
        all_HX  = {}

        now = time.time()

        #threads = nthreads
        threads = 10 # run 10 MC chains in parallel to collect snapshots
        # if initState = -1: the chains are alternatingly started in "all +1" and "all -1"
        snapshots, allNeighbours_G = infcy.getSnapshotsPerDistNodes(model, \
                                nodes, **snapshotSettingsCond, \
                                threads=threads, initStateIdx=args.initState)

        with open(f'{targetDirectory}/snapshots_nodes_{now}.pickle', 'wb') as f:
            pickle.dump(snapshots, f)

        for i, node in enumerate(nodes):
            print(f'start conditional sampling for node {node}')
            MI, HX, H_XgivenY, keys = computeMI_cond(model, node, minDist, maxDist, \
                                allNeighbours_G[i], snapshots[i], nTrials, \
                                nSamples, modelSettingsCond, corrTimeSettings)
            print(f'MI = {MI}')
            all_MIs[node]   = np.array(MI)
            all_HX[node]    = np.array(HX)

        with open(f'{targetDirectory}/MI_vector_nodes_{now}.pickle', 'wb') as f:
            pickle.dump(all_MIs, f)
        with open(f'{targetDirectory}/HX_vector_nodes_{now}.pickle', 'wb') as f:
            pickle.dump(all_HX, f)

    print(f'time elapsed: {timer()-start : .2f} seconds')

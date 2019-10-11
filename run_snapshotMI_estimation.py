# NOTE: #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infoTheory, simulation
from Utils import IO

import itertools, scipy, time, subprocess, \
    os, pickle, sys, argparse
import multiprocessing as mp
import numpy as np
import networkx as nx
from tqdm import tqdm
from timeit import default_timer as timer
from scipy import stats

# use as many concurrent threads as there are cores availble
nthreads = mp.cpu_count()


parser = argparse.ArgumentParser(description='run MCMC simulations and estimate \
            the snapshot-based MI between the given nodes and their neighbourhood \
            shells at distances 1 to maxDist')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--nodes', type=str, default='all',
            help='path to numpy array containing node IDs')
parser.add_argument('--maxDist', type=int, default=-1,
            help='max distance to central node. If -1, use diameter, if -2 use \
            distance where max neighbours are reached')
parser.add_argument('--minDist', type=int, default=1, help='min distance to central node')
parser.add_argument('--threshold', type=float, default=0,
            help='when MI drops below threshold, stop simulating for larger distances')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')
parser.add_argument('--maxCorrTime', type=int, default=-1,
            help='maximal number of simulation steps between two samples')
parser.add_argument('--minCorrTime', type=int, default=1,
            help='minimal number of simulation steps between two samples')
parser.add_argument('--snapshots', type=int, default=100,
            help='number of neighbourhood snapshots')
parser.add_argument('--trials', type=int, default=10,
            help='number of repeated MCMC simulations with fixed neighbourhood \
            state used to estimate the conditional PDF of the central node')
parser.add_argument('--numSamples', type=int, default=int(1e3),
            help='number of samples per MCMC simulation with fixed neighbourhood state')
parser.add_argument('--magSide', type=str, default='',
            help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=1,
            help='initial system state (given as index to model.agentStates). \
            If -1, repeated simulations start from random agent states')
parser.add_argument('--uniformPDF', action="store_true",
            help='assume uniform distribution over neighbourhood snapshots')




def computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots,
                    nTrials, nSamples, modelSettings, corrTimeSettings):

    all_MI          = np.full(maxDist - minDist + 1, np.nan)
    all_HX          = np.full(maxDist - minDist + 1, np.nan)

    stop = False

    subgraph_nodes = [node]
    for idx, d in enumerate(minDist, maxDist+1):

        # get subgraph and outer neighbourhood at distance d
        if len(neighbours_G[d]) > 0:

            subgraph = nx.ego_graph(model.graph, node, d)

            print(f'------------------- distance d={d}, num neighbours = \
                {len(neighbours_G[d])}, subgraph size = {len(subgraph)}, \
                num states = {len(snapshots[d-1])} -----------------------')

            model_subgraph = fastIsing.Ising(subgraph, **modelSettings)

            # determine correlation time for subgraph Ising model
            if args.maxCorrTime == args.minCorrTime:
                distSamples_subgraph = args.maxCorrTime
                mixingTime_subgraph = corrTimeSettings['burninSteps']
            else:
                mixingTime_subgraph, meanMag, distSamples_subgraph, _ = \
                    simulation.determineCorrTime(model_subgraph, nodeG=node, **corrTimeSettings)
                if args.maxCorrTime > 0:
                    distSamples_subgraph = min(distSamples_subgraph, args.maxCorrTime)
                distSamples_subgraph = max(distSamples_subgraph, args.minCorrTime)
            print(f'correlation time = {distSamples_subgraph}')
            print(f'mixing time      = {mixingTime_subgraph}')


            threads = nthreads if len(subgraph) > 20 else 1

            _, _, MI, HX, HXgiveny, keys, probs = \
                    simulation.neighbourhoodMI(model_subgraph, node, \
                        d, neighbours_G, snapshots[d-1], nTrials=nTrials, \
                        burninSteps=mixingTime_subgraph, nSamples=nSamples, \
                        distSamples=distSamples_subgraph, threads=threads, \
                        initStateIdx=args.initState, uniformPDF=args.uniformPDF, out='MI')

            all_MI[idx] = MI
            all_HX[idx] = HX

            if MI < args.threshold: break


    return all_MI, all_HX



if __name__ == '__main__':

    args = parser.parse_args()
    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    maxDist = args.maxDist if args.maxDist > 0 else nx.diameter(graph)
    nodes = np.array(list(graph), dtype=int) if args.nodes == 'all' else np.load(args.nodes)
    networkSettings = dict( \
        path = args.graph, \
        size = N, \
        nodes = nodes
    )


    # setup Ising model with N=networkSize spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = args.magSide if args.magSide in ['pos', 'neg'] else ''
    )
    model = fastIsing.Ising(graph, **modelSettings)

    try:
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']

    except:
        # try to load data containing mixing and correlation time. If it doesn't exist
        # yet, use 'run_mixing.py' script to generate it
        subprocess.call(['python3', 'run_mixing.py', f'{args.T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']


    snapshotSettings = dict( \
        nSamples    = args.snapshots, \
        burninSteps = burninSteps, \
        maxDist     = maxDist
    )

    modelSettingsCond = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = ''
    )


    minDist = args.minDist
    nTrials = args.trials
    nSamples = args.numSamples

    # repetitive runs of the MI estimation procedure
    for i in range(args.runs):

        start = timer()

        all_MI  = {}
        all_HX  = {}

        threads = 10 # run 10 MC chains in parallel to collect snapshots
        snapshots, allNeighbours_G = simulation.getSnapshotsPerDistNodes(model, \
                                nodes, **snapshotSettings, \
                                threads=threads, initStateIdx=args.initState)

        for i, node in enumerate(nodes):
            if args.maxDist == -2:
                # find distance with max number of neighbours
                maxDist = np.argmax([len(allNeighbours_G[i][d]) \
                    for d in range(1, max(allNeighbours_G[i].keys())+1)]) + 1

            print(f'start conditional sampling for node {node}')
            all_MI[node], all_HX[node] = computeMI_cond(model, node, minDist, maxDist, \
                                allNeighbours_G[i], snapshots[i], nTrials, \
                                nSamples, modelSettingsCond, corrTimeSettings)
            print(f'MI = {all_MI[node]}')

        result = IO.SimulationResult('vector', \
                    networkSettings     = networkSettings, \
                    modelSettings       = modelSettings, \
                    snapshotSettings    = snapshotSettings, \
                    corrTimeSettings    = corrTimeSettings, \
                    mixingResults       = mixingResults, \
                    mi                  = all_MI, \
                    hx                  = all_HX, \
                    computeTime         = timer()-start, \
                    threshold           = args.threshold)
        result.saveToPickle(targetDirectory)

        print(f'run {r} finished')
        print(f'time elapsed: {timer()-start : .2f} seconds')

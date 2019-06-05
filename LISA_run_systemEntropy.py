#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infcy
from Utils import IO
import networkx as nx, itertools, scipy, time, subprocess, \
                os, pickle, sys, argparse, multiprocessing as mp
import itertools
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from scipy import stats



def compute_entropies(snapshots, nSamples):
    sum = np.sum([np.sum(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()])
    print(sum)
    condEntropies = [infcy.entropyEstimateH2(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()]
    print(condEntropies)
    condH = np.sum([condEntropies[i] * np.sum(np.fromiter(s.values(), dtype=int))/(nSamples) for i, s in enumerate(snapshots.values())])
    print(f'H2(S|s_i) = {condH}')

    allSnapshots = {}
    for _, s in snapshots.items():
        for k, v in s.items():
            if k in allSnapshots.keys():
                allSnapshots[k] += v
            else:
                allSnapshots[k] = v
    systemH = infcy.entropyEstimateH2(np.fromiter(allSnapshots.values(), dtype=int))
    print(f'H2(S) = {systemH}')

    return condH, systemH


nthreads = mp.cpu_count()
#nthreads = 1


parser = argparse.ArgumentParser(description='run MC chain, sample system snapshots and estimate the system entropy conditioned on the given node set')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--nodes', type=str, default='', help='path to numpy array containg nodes to be fixed')
parser.add_argument('--single', action="store_true", help='fix nodes individually')
#parser.add_argument('--centralNode', type=int, default=-1, help='node of interest, reference point for max dist')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')
parser.add_argument('--dist', type=int, default=-1, help='max dist up to which nodes are considered for system entropy estimate')
#parser.add_argument('--maxCorrTime', type=int, default=-1, help='max distance between two samples in the MC')
#parser.add_argument('--minCorrTime', type=int, default=1, help='min distance between two samples in the MC')
parser.add_argument('--snapshots', type=int, default=100, help='number of system snapshots')
parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate entropy')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=1, help='initial system state')



if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)


    networkSettings = dict( \
        path = args.graph, \
        nNodes = N, \
        fixedNodes = args.nodes
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
        #raise Exception('No mixing results found! Please run the mixing script first to determine the mixing time of the model.')
        subprocess.call(['python3', 'LISA_run_mixing.py', f'{args.T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000', \
                        '--corrthreshold', '0.5'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']


    systemSnapshotSettings = dict( \
        nSnapshots    = args.snapshots, \
        repeats       = args.repeats, \
        burninSamples = int(burninSteps), \
        distSamples     = int(distSamples)
    )
    IO.saveSettings(targetDirectory, systemSnapshotSettings, 'systemSnapshots')

    for i in range(args.runs):
        now = time.time()

        nodelist = np.load(args.nodes).astype(int)

        if args.single:
                if args.dist > 0:
                    allNodes = [np.array(list(nx.ego_graph(graph, node, args.dist))) for node in nodelist]
                    systemNodes = [list(allNodes[i][np.where(allNodes[i] != node)].astype(int)) for i, node in enumerate(nodelist)]
                else:
                    allNodes = np.array(list(graph))
                    systemNodes = [list(allNodes[np.where(allNodes != node)].astype(int)) for node in nodelist]
                    systemNodes = [list(allNodes) for node in nodelist]

                fixedNodes = [[node] for node in nodelist]


                snapshots = infcy.getSystemSnapshotsSets(model, systemNodes, fixedNodes, \
                              **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

                allCondH      = {}
                allSystemH    = {}
                for i, node in enumerate(nodelist):
                    print(f'--------------- node {node} ---------------')
                    allCondH[node], allSystemH[node] = compute_entropies(snapshots[i], args.snapshots*args.repeats)

                IO.savePickle(targetDirectory, f'condSystemEntropy_results_individual nodes_{now}', allCondH)
                IO.savePickle(targetDirectory, f'systemEntropy_results_individual nodes_{now}', allSystemH)



        elif args.nodes == '':
            snapshots = infcy.getSystemSnapshots(model, fixedNodesG=None, fixedStates=None, \
                          **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)
            print(np.fromiter(snapshots.values(), dtype=int))
            entropy = infcy.entropyEstimateH2(np.fromiter(snapshots.values(), dtype=int))
            print(f'system entropy = {entropy}')

        else:
            if args.nodes == 'test':
                fixedNodes = allNodes[:2]
            else:
                fixedNodes = np.load(args.nodes).astype(int)

            systemNodes = np.array([n for n in allNodes if n not in fixedNodes], dtype=int)


            snapshots = infcy.getSystemSnapshots(model, systemNodes, fixedNodes, \
                          **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)
            print([np.sum(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()])
            condEntropies = [infcy.entropyEstimateH2(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()]
            #print(f'H2 entropy estimates = {condEntropies}')
            condH = np.sum([condEntropies[i] * np.sum(np.fromiter(s.values(), dtype=int))/(args.snapshots*args.repeats) for i, s in enumerate(snapshots.values())])
            print(f'average H2(S|s_i) = {condH}')
            #condEntropies = [stats.entropy(np.fromiter(s.values(), dtype=int), base=2) for s in snapshots.values()]
            #print(f'naive plug-in entropyies = {condEntropies}')

            allSnapshots = {}
            for _, s in snapshots.items():
                for k, v in s.items():
                    if k in allSnapshots.keys():
                        allSnapshots[k] += v
                    else:
                        allSnapshots[k] = v
            systemEntropy = infcy.entropyEstimateH2(np.fromiter(allSnapshots.values(), dtype=int))
            print(f'system entropy H(S) = {systemEntropy}')

            tmp = dict( \
                    nodesOfInterest = list(fixedNodes), \
                    maxDist = args.dist, \
                    systemEntropy = systemEntropy, \
                    condEntropy = condH, \
                    T = args.T, \
                    samples = args.snapshots*args.repeats)
            IO.savePickle(targetDirectory, f'systemEntropy_results_{now}', tmp)
            print(targetDirectory)
            """

            snapshots = infcy.getSystemSnapshotsFixedNodes(model, systemNodes, fixedNodesG=np.empty(0, int), fixedStates=np.empty(0, int), \
                          **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)
            print(np.fromiter(snapshots.values(), dtype=int))
            print(np.sum(np.fromiter(snapshots.values(), dtype=int)))
            entropy = infcy.entropyEstimateH2(np.fromiter(snapshots.values(), dtype=int))
            print(entropy)

            allStates = np.array([state for state in itertools.product([-1,1], repeat=fixedNodes.size) if np.mean(state) >= 0]) # only take states on one side of the magnetization into account
            allEntropies = np.zeros(allStates.shape[0])
            for i, states in enumerate(allStates):
                snapshots = infcy.getSystemSnapshotsFixedNodes(model, systemNodes, fixedNodesG=fixedNodes, fixedStates=states, \
                              **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)
                print(np.fromiter(snapshots.values(), dtype=int))
                print(np.sum(np.fromiter(snapshots.values(), dtype=int)))
                allEntropies[i] = infcy.entropyEstimateH2(np.fromiter(snapshots.values(), dtype=int))

            print(allStates, allEntropies)
            """

        print(f'time elapsed: {timer()-start : .2f} seconds')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infoTheory, simulation
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
    counts = [np.fromiter(s.values(), dtype=int) for s in snapshots.values()]
    #print(f'counts = {counts}')
    condEntropies = [infoTheory.entropyEstimateH2(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()]
    condH = np.sum([condEntropies[i] * np.sum(np.fromiter(s.values(), dtype=int))/(nSamples) for i, s in enumerate(snapshots.values())])

    allSnapshots = {}
    for _, s in snapshots.items():
        for k, v in s.items():
            if k in allSnapshots.keys():
                allSnapshots[k] += v
            else:
                allSnapshots[k] = v
    systemH = infoTheory.entropyEstimateH2(np.fromiter(allSnapshots.values(), dtype=int))
    print(f'num system states = {len(allSnapshots)}, min count = {min(allSnapshots.values())}, max count = {max(allSnapshots.values())}')

    return condH, systemH, len(allSnapshots), max(allSnapshots.values())


nthreads = mp.cpu_count()


parser = argparse.ArgumentParser(description='run MCMC simulation, sample system snapshots and estimate the system entropy conditioned on the given node set')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--nodes', type=str, default='', help='path to numpy array containg nodes to be fixed')
parser.add_argument('--preselectedNodes', action="store_true", help='nodes should always be in the k-set, create sets containing these nodes plus one new node from the graph')
parser.add_argument('--k', type=int, default=1, help='set size')
parser.add_argument('--excludeNodes', action="store_true", help='exclude fixed nodes from system entropy')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')
parser.add_argument('--trials', type=int, default=1, help='number of trials. The median of all MI estimates is saved')
parser.add_argument('--dist', type=int, default=-1, help='max dist up to which nodes are considered for system entropy estimate')
parser.add_argument('--snapshots', type=int, default=10000, help='number of system snapshots')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=1, help='initial system state')



if __name__ == '__main__':

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)


    networkSettings = dict( \
        path = args.graph, \
        size = N, \
        fixedNodes = args.nodes
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
        subprocess.call(['python3', 'run_mixing.py', f'{args.T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']


    systemSnapshotSettings = dict( \
        nSnapshots    = args.snapshots, \
        burninSteps   = int(burninSteps), \
        distSamples   = int(distSamples)
    )
    IO.saveSettings(targetDirectory, systemSnapshotSettings, 'systemSnapshots')

    for i in range(args.runs):
        now = time.time()
        start = timer()

        if args.nodes == '':

            allEntropies = np.zeros(args.trials)
            numSysStates = np.zeros(args.trials)
            maxCounts    = np.zeros(args.trials)
            meanStateEntropies = np.zeros(args.trials)

            for trial in range(args.trials):
                snapshots = simulation.getSystemSnapshots(model, np.array(list(graph)), \
                              **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

                meanStateEntropies[trial] = np.mean([ stats.entropy(np.unique(np.frombuffer(s), return_counts=True)[1]/N, base=2) for s in snapshots.keys() ])
                print(f' mean state entropy = {meanStateEntropies[trial]}')

                #print(np.fromiter(snapshots.values(), dtype=int))
                allEntropies[trial] = infoTheory.entropyEstimateH2(np.fromiter(snapshots.values(), dtype=int))
                numSysStates[trial] = len(snapshots)
                maxCounts[trial] = max(snapshots.values())
                print(f'num system states = {len(snapshots)}, min count = {min(snapshots.values())}, max count = {max(snapshots.values())}')

            entropy = np.median(allEntropies)
            print(f'system entropy = {entropy}')

            result = IO.SimulationResult('systemMI', \
                        networkSettings     = networkSettings, \
                        modelSettings       = modelSettings, \
                        snapshotSettings    = systemSnapshotSettings, \
                        corrTimeSettings    = corrTimeSettings, \
                        mixingResults       = mixingResults, \
                        hx                  = allEntropies, \
                        numSystemStates     = numSysStates, \
                        maxCounts           = maxCounts, \
                        meanStateEntropies  = meanStateEntropies, \
                        computeTime         = timer()-start )
            result.saveToPickle(targetDirectory)

        elif args.nodes =='all':
            allNodes = np.array(list(graph))

            if args.nodes == 'test':
                fixedNodes = allNodes[:2]
            else:
                fixedNodes = np.load(args.nodes).astype(int)

            if args.excludeNodes:
                systemNodes = np.array([n for n in allNodes if n not in fixedNodes], dtype=int)
            else:
                systemNodes = allNodes

            systemH = np.zeros(args.trials)
            condH = np.zeros(args.trials)
            MI = np.zeros(args.trials)
            for trial in range(args.trials):
                snapshots = simulation.getSystemSnapshotsCond(model, [systemNodes], [fixedNodes], \
                              **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)[0]

                print([np.sum(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()])
                condEntropies = [infoTheory.entropyEstimateH2(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()]
                condH[trial] = np.sum([condEntropies[i] * np.sum(np.fromiter(s.values(), dtype=int))/args.snapshots for i, s in enumerate(snapshots.values())])
                allSnapshots = {}
                for _, s in snapshots.items():
                    for k, v in s.items():
                        if k in allSnapshots.keys():
                            allSnapshots[k] += v
                        else:
                            allSnapshots[k] = v
                systemH[trial] = infoTheory.entropyEstimateH2(np.fromiter(allSnapshots.values(), dtype=int))

                MI[trial] = systemH[trial] - condH[trial]
                print(f'Trial {trial}: MI = {MI[trial]}')

            mi = np.median(MI)
            idx = np.where(MI == mi)

            print(f'MI = {mi}')

            result = IO.SimulationResult('systemMI', \
                        networkSettings     = networkSettings, \
                        modelSettings       = modelSettings, \
                        snapshotSettings    = systemSnapshotSettings, \
                        corrTimeSettings    = corrTimeSettings, \
                        mixingResults       = mixingResults, \
                        fixedNodes          = fixedNodes, \
                        systemNodes         = systemNodes, \
                        mi                  = MI, \
                        hx                  = systemH, \
                        computeTime         = timer()-start )
            result.saveToPickle(targetDirectory)

        elif args.k > 0:

            allNodes = np.array(list(graph))
            if args.nodes == 'test':
                nodelist = np.array([allNodes[0]])
            else:
                nodelist = np.load(args.nodes).astype(int)
            if args.preselectedNodes:
                sets = [ list(nodelist).append(n) for n in allNodes if n not in nodelist ]
            else:
                sets = list(itertools.combinations(nodelist, args.k))
            #print(f'sets: {sets}')

            if args.excludeNodes:
                systemNodes = [list(allNodes[~np.isin(allNodes, s)].astype(int)) for s in sets]
            else:
                systemNodes = [list(allNodes) for s in sets]


            fixedNodes = [list(s) for s in sets]

            split_idx = np.arange(100, len(sets), 100)
            split_fixedNodes = np.split(fixedNodes, split_idx)
            split_systemNodes = np.split(systemNodes, split_idx)

            allMI         = {}
            allCondH      = {}
            allSystemH    = {}
            numSysStates  = {}
            maxCounts     = {}

            for f, s in zip(split_fixedNodes, split_systemNodes):

                for set in f:
                    if set.size == 1:
                        set = set[0]
                    else:
                        set = tuple(set)
                    allCondH[set] = np.zeros(args.trials)
                    allSystemH[set] = np.zeros(args.trials)
                    allMI[set] = np.zeros(args.trials)
                    numSysStates[set] = np.zeros(args.trials)
                    maxCounts[set] = np.zeros(args.trials)

                print(f'start {args.trials} trials')

                for trial in range(args.trials):

                    snapshots = simulation.getSystemSnapshotsSets(model, list(s), list(f), \
                                  **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

                    print('start computing entropies')

                    #print(f'all sets: {f}')

                    for i, set in enumerate(f):
                        if set.size == 1:
                            set = set[0]
                            print(f'--------------- node {set} ---------------')
                        else:
                            set = tuple(set)
                            print(f'--------------- set {set} ---------------')
                        allCondH[set][trial], allSystemH[set][trial], numSysStates[set][trial], maxCounts[set][trial] = compute_entropies(snapshots[i], args.snapshots)
                        allMI[set][trial] = allSystemH[set][trial] - allCondH[set][trial]
                        print(f'MI = {allMI[set][trial]}')


            result = IO.SimulationResult('systemMI', \
                        networkSettings     = networkSettings, \
                        modelSettings       = modelSettings, \
                        snapshotSettings    = systemSnapshotSettings, \
                        corrTimeSettings    = corrTimeSettings, \
                        mixingResults       = mixingResults, \
                        #mi                  = medianMI, \
                        mi                  = allMI, \
                        hx                  = allSystemH, \
                        k                   = args.k, \
                        computeTime         = timer()-start )
            result.saveToPickle(targetDirectory)

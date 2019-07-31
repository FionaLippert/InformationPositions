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
    #print(sum)
    condEntropies = [infoTheory.entropyEstimateH2(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()]
    #print(condEntropies)
    condH = np.sum([condEntropies[i] * np.sum(np.fromiter(s.values(), dtype=int))/(nSamples) for i, s in enumerate(snapshots.values())])
    #print(f'H2(S|s_i) = {condH}')

    allSnapshots = {}
    for _, s in snapshots.items():
        for k, v in s.items():
            if k in allSnapshots.keys():
                allSnapshots[k] += v
            else:
                allSnapshots[k] = v
    systemH = infoTheory.entropyEstimateH2(np.fromiter(allSnapshots.values(), dtype=int))
    #print(f'H2(S) = {systemH}')

    return condH, systemH



parser = argparse.ArgumentParser(description='')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--excludeNodes', action="store_true", help='exclude fixed nodes from system entropy')
parser.add_argument('--onlyRandom', action="store_true", help='do not run greedy algorithm, only rankom k-sets')
parser.add_argument('--trials', type=int, default=1, help='number of trials. The median of all MI estimates is saved')
parser.add_argument('--snapshots', type=int, default=10000, help='number of system snapshots')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=1, help='initial system state')
parser.add_argument('--k_max', type=int, default=3, help='max k-set size to be considered')

nthreads = mp.cpu_count()

if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load data
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    T = args.T

    networkSettings = dict( \
        path = args.graph, \
        nNodes = N
    )

    # setup Ising model with nNodes spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = args.magSide if args.magSide in ['pos', 'neg'] else ''
    )
    #IO.saveSettings(targetDirectory, modelSettings, 'model')
    model = fastIsing.Ising(graph, **modelSettings)

    try:
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']

    except:
        #raise Exception('No mixing results found! Please run the mixing script first to determine the mixing time of the model.')
        subprocess.call(['python3', 'run_mixing.py', f'{T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000', \
                        '--corrthreshold', '0.5'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']


    systemSnapshotSettings = dict( \
        nSnapshots    = args.snapshots, \
        burninSamples = int(burninSteps), \
        distSamples     = int(distSamples)
    )
    IO.saveSettings(targetDirectory, systemSnapshotSettings, 'systemSnapshots')



    selected_nodes = []
    remaining_nodes = list(graph)
    nodes_array = np.array(remaining_nodes)

    mi_greedy = {}
    h_greedy = {}

    mi_random = {}
    h_random = {}

    if args.onlyRandom:

        sets = [list(np.random.choice(nodes_array, k, replace=False)) for k in range(1, args.k_max + 1)]
        print(sets)
        if args.excludeNodes:
            systemNodes = [list(nodes_array[~np.isin(nodes_array, s)].astype(int)) for s in sets]
        else:
            systemNodes = [list(nodes_array) for s in sets]

        snapshots = simulation.getSystemSnapshotsSets(model, systemNodes, sets, \
                      **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

        for i, s in enumerate(sets):
            condRand, systemRand = infoTheory.compute_entropies(snapshots[i], args.snapshots)
            mi_random[tuple(s)] = systemRand - condRand
            h_random[tuple(s)] = condRand

    else:

        for k in range(1, args.k_max + 1):

            sets = [ selected_nodes + [n] for n in remaining_nodes ]
            print(sets)

            sets.append(list(np.random.choice(nodes_array, k, replace=False)))

            if args.excludeNodes:
                systemNodes = [list(nodes_array[~np.isin(nodes_array, s)].astype(int)) for s in sets]
            else:
                systemNodes = [list(nodes_array) for s in sets]

            MI      = np.zeros(len(remaining_nodes))
            condH   = np.zeros(len(remaining_nodes))
            systemH = np.zeros(len(remaining_nodes))


            snapshots = simulation.getSystemSnapshotsSets(model, systemNodes, sets, \
                          **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

            #IO.savePickle(targetDirectory, f'backup_snapshots_k={k}', snapshots)

            for i, n in enumerate(remaining_nodes):
                print(f'--------------- node set {selected_nodes + [n]} ---------------')

                condH[i], systemH[i] = infoTheory.compute_entropies(snapshots[i], args.snapshots)
                MI[i] = systemH[i] - condH[i]
                print(f'MI = {MI[i]}')


            ranking = np.argsort(MI)
            top_idx = ranking[-1]
            top_node = remaining_nodes[top_idx]

            print(f'best node choice: {top_node} with MI = {MI[top_idx]}')

            selected_nodes.append(top_node)
            remaining_nodes.remove(top_node)

            mi_greedy[tuple(selected_nodes)] = MI[top_idx]
            h_greedy[tuple(selected_nodes)] = condH[top_idx]

            condRand, systemRand = infoTheory.compute_entropies(snapshots[-1], args.snapshots)
            mi_random[tuple(sets[-1])] = systemRand - condRand
            h_random[tuple(sets[-1])] = condRand


    result = IO.SimulationResult('greedy', \
                networkSettings     = networkSettings, \
                modelSettings       = modelSettings, \
                snapshotSettings    = systemSnapshotSettings, \
                corrTimeSettings    = corrTimeSettings, \
                mixingResults       = mixingResults, \
                miGreedy            = mi_greedy, \
                hCondGreedy         = h_greedy, \
                miRandom            = mi_random, \
                hCondRandom         = h_random, \
                computeTime         = timer()-start )
    result.saveToPickle(targetDirectory)

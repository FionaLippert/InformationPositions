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
    print(sum)
    condEntropies = [infoTheory.entropyEstimateH2(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()]
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
    systemH = infoTheory.entropyEstimateH2(np.fromiter(allSnapshots.values(), dtype=int))
    print(f'H2(S) = {systemH}')

    return condH, systemH


nthreads = mp.cpu_count()
#nthreads = 1


parser = argparse.ArgumentParser(description='run MC chain, sample system snapshots and estimate the system entropy conditioned on the given node set')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
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
        nNodes = N
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
        subprocess.call(['python3', 'run_mixing.py', f'{args.T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000', \
                        '--corrthreshold', '0.5'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']


    numSnapshots = np.logspace(1,6,20)
    results = np.zeros(numSnapshots.size)

    for i, s in enumerate(numSnapshots):
        now = time.time()

        systemSnapshotSettings = dict( \
            nSnapshots    = s, \
            repeats       = args.repeats, \
            burninSamples = int(burninSteps), \
            distSamples     = int(distSamples)
        )

        snapshots = simulation.getSystemSnapshotsFixedNodes(model, np.array(list(graph)), fixedNodesG=np.empty(0, int), fixedStates=np.empty(0, int), \
                      **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

        #print(np.fromiter(snapshots.values(), dtype=int))
        entropy = infoTheory.entropyEstimateH2(np.fromiter(snapshots.values(), dtype=int))
        print(f'system entropy = {entropy}')

        results[i] = entropy

    print(results)


    tmp = dict( \
            systemEntropies = list(results), \
            T = args.T, \
            samples = [s * args.repeats for s in numSnapshots])
    IO.savePickle(targetDirectory, f'systemEntropy_varying_nSnapshots_{now}', tmp)
    print(targetDirectory)


    print(f'time elapsed: {timer()-start : .2f} seconds')

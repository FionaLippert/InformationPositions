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


nthreads = mp.cpu_count()
#nthreads = 1


parser = argparse.ArgumentParser(description='run MC chain, sample system snapshots and estimate the system entropy conditioned on the given node set')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--nodes', type=str, default='', help='path to numpy array containg nodes to be fixed')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')
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

        if args.nodes == '':
            snapshots = infcy.getSystemSnapshots(model, fixedNodesG=None, fixedStates=None, \
                          **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)
            print(np.fromiter(snapshots.values(), dtype=int))
            entropy = infcy.entropyEstimateH2(np.fromiter(snapshots.values(), dtype=int))
            print(f'system entropy = {entropy}')

        else:
            if args.nodes == 'test':
                fixedNodes = np.array(list(graph)[:2])
            else:
                fixedNodes = np.load(args.nodes)

            allStates = np.array([state for state in itertools.product([-1,1], repeat=fixedNodes.size) if np.mean(state) >= 0]) # only take states on one side of the magnetization into account
            allEntropies = np.zeros(allStates.shape[0])
            for i, states in enumerate(allStates):
                snapshots = infcy.getSystemSnapshots(model, fixedNodes, fixedStates=states, \
                              **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)
                print(np.fromiter(snapshots.values(), dtype=int))
                allEntropies[i] = infcy.entropyEstimateH2(np.fromiter(snapshots.values(), dtype=int))

            print(allEntropies)

        print(f'time elapsed: {timer()-start : .2f} seconds')

#!/usr/bin/env python3
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
            the magnetization-based MI between the given nodes and their \
            neighbourhood shells at distances 1 to maxDist')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--nodes', type=str, default='all',
            help='path to numpy array containing node IDs')
parser.add_argument('--neighboursDir', type=str, default='',
            help='path to directory containing pickled neighbours dictionary')
parser.add_argument('--maxDist', type=int, default=-1,
            help='max distance to central node. If -1, use diameter.')
parser.add_argument('--runs', type=int, default=1,
            help='number of repetititve runs')
parser.add_argument('--bins', type=int, default=100,
            help='number of bins for average magnetization of neighbours')
parser.add_argument('--numSamples', type=int, default=int(1e5),
            help='number of system samples')
parser.add_argument('--magSide', type=str, default='',
            help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=1,
            help='initial system state (given as index to model.agentStates). \
            If -1, repeated simulations start from random agent states')
parser.add_argument('--pairwise', action="store_true",
            help='compute pairwise correlation and MI')


if __name__ == '__main__':

    print("starting simulations for magnetization-based MI estimation")

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

    # try to load data containing mixing and correlation time. If it doesn't exist
    # yet, use 'run_mixing.py' script to generate it
    try:
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']
        print(f'mixing time      = {burninSteps}')
        print(f'correlation time = {distSamples}')
    except:
        subprocess.call(['python3', 'run_mixing.py', f'{args.T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']

    # try to load neighbourhood shell data. If it doesn't exist yet, generate it
    try:
        if len(args.neighboursDir) > 0:
            neighboursG = IO.loadPickle(args.neighboursDir, 'neighboursG')
        else:
            neighboursG = IO.loadPickle(targetDirectory, 'neighboursG')
    except:
        print(f'determining neighbours')
        neighboursG = model.neighboursAtDistAllNodes(nodes, maxDist)
        if len(args.neighboursDir) > 0:
            os.makedirs(args.neighboursDir, exist_ok=True)
            IO.savePickle(args.neighboursDir, 'neighboursG', neighboursG)
        else:
            IO.savePickle(targetDirectory, 'neighboursG', neighboursG)


    snapshotSettings = dict( \
        nSamples    = args.numSamples, \
        burninSteps = burninSteps, \
        distSamples = distSamples, \
        maxDist     = maxDist, \
        nBins       = args.bins
    )


    # repetitive runs of the MI estimation procedure
    for r in range(args.runs):

        startS = timer()
        avgSnapshots, avgSystemSnapshots, fullSnapshots = \
            simulation.getJointSnapshotsPerDistNodes(model, nodes, neighboursG, \
                                    **snapshotSettings, threads=nthreads, \
                                    initStateIdx=args.initState, getFullSnapshots=1)
        simulationTime = timer() - startS

        startC = timer()
        MI_avg, MI_system, HX = infoTheory.processJointSnapshots_allNodes(avgSnapshots, args.numSamples, nodes, maxDist, avgSystemSnapshots)

        result = IO.SimulationResult('magMI', \
                    networkSettings     = networkSettings, \
                    modelSettings       = modelSettings, \
                    snapshotSettings    = snapshotSettings, \
                    corrTimeSettings    = corrTimeSettings, \
                    mixingResults       = mixingResults, \
                    mi                  = MI_avg, \
                    miSystemMag         = MI_system, \
                    hx                  = HX, \
                    computeTime         = simulationTime + timer()-startC )
        result.saveToPickle(targetDirectory)

        if args.pairwise:

            startC = timer()
            MI, corr = infoTheory.pairwiseMI_allNodes(model, nodes, fullSnapshots)

            result = IO.SimulationResult('pairwiseMI', \
                        networkSettings     = networkSettings, \
                        modelSettings       = modelSettings, \
                        snapshotSettings    = snapshotSettings, \
                        corrTimeSettings    = corrTimeSettings, \
                        mixingResults       = mixingResults, \
                        mi                  = MI, \
                        corr                = corr, \
                        computeTime         = simulationTime + timer()-startC )
            result.saveToPickle(targetDirectory)

        print(f'run {r} finished')
        print(f'time elapsed: {timer()-startS : .2f} seconds')

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

nthreads = mp.cpu_count()
#nthreads = 1

parser = argparse.ArgumentParser(description='run MC chain and compute MI based on the joint PDF of the central node and its neighbours')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--nodes', type=str, default='all', help='path to numpy array of node IDs')
parser.add_argument('--neighboursDir', type=str, default='', help='path to directory containing pickled neighbours dictionary')
parser.add_argument('--maxDist', type=int, default=-1, help='max distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetititve runs')
parser.add_argument('--bins', type=int, default=100, help='number of bins for average magnetization of neighbours')
#parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate MI')
parser.add_argument('--numSamples', type=int, default=100000, help='number of system samples')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=1, help='initial system state (given as index to model.agentStates). If -1, repeated simulations start from different agent states')
parser.add_argument('--pairwise', action="store_true", help='compute pairwise correlation and MI')


if __name__ == '__main__':

    print("starting with average neighbour MI approach")

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    graph = nx.read_gpickle(args.graph)
    N = len(graph)

    # load network
    if args.maxDist > 0:
        maxDist = args.maxDist
    else:
        maxDist = nx.diameter(graph)

    if args.nodes == 'all':
        nodes = np.array(list(graph), dtype=int)
    else:
        nodes = np.load(args.nodes)
        centralNodeIdx = -1
    #deg = graph.degree[node]

    networkSettings = dict( \
        path = args.graph, \
        nNodes = N, \
        nodes = nodes
    )
    #IO.saveSettings(targetDirectory, networkSettings, 'network')


    # setup Ising model with nNodes spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = args.magSide if args.magSide in ['pos', 'neg'] else ''
    )
    #IO.saveSettings(targetDirectory, modelSettings, 'model')
    model = fastIsing.Ising(graph, **modelSettings)
    print(modelSettings['magSide'])

    try:
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']
        print(f'mixing time = {burninSteps}')
        print(f'correlation time = {distSamples}')

    except:
        #raise Exception('No mixing results found! Please run the mixing script first to determine the mixing and correlation time of the model.')
        subprocess.call(['python3', 'run_mixing.py', f'{args.T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000', \
                        '--corrthreshold', '0.5'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']

    try:
        if len(args.neighboursDir) > 0:
            neighboursG = IO.loadPickle(args.neighboursDir, 'neighboursG')
        else:
            neighboursG = IO.loadPickle(targetDirectory, 'neighboursG')
        #print(neighboursG)
    except:
        print(f'determining neighbours')
        neighboursG = model.neighboursAtDistAllNodes(nodes, maxDist)
        #print(neighboursG)
        if len(args.neighboursDir) > 0:
            os.makedirs(args.neighboursDir, exist_ok=True)
            IO.savePickle(args.neighboursDir, 'neighboursG', neighboursG)
        else:
            IO.savePickle(targetDirectory, 'neighboursG', neighboursG)



    snapshotSettings = dict( \
        nSamples    = args.numSamples, \
        burninSamples = burninSteps, \
        distSamples   = distSamples, \
        maxDist     = maxDist, \
        nBins       = args.bins
    )
    #IO.saveSettings(targetDirectory, snapshotSettings, 'jointSnapshots')


    for r in range(args.runs):

        avgSnapshots, avgSystemSnapshots, fullSnapshots = simulation.getJointSnapshotsPerDistNodes(model, nodes, \
                                                                            neighboursG, \
                                                                            **snapshotSettings, threads=nthreads, \
                                                                            initStateIdx=args.initState, getFullSnapshots=1)

        start_2 = timer()
        #print(fullSnapshots.shape)
        now = time.time()

        MI_avg, MI_system, HX = infoTheory.processJointSnapshots_allNodes(avgSnapshots, args.numSamples, nodes, maxDist, avgSystemSnapshots)

        #IO.savePickle(targetDirectory, f'MI_meanField_nodes_{now}', MI_avg)
        #IO.savePickle(targetDirectory, f'HX_meanField_nodes_{now}', HX)
        #IO.savePickle(targetDirectory, f'MI_systemMag_nodes_{now}', MI_system)

        result = IO.SimulationResult('avg', \
                    networkSettings     = networkSettings, \
                    modelSettings       = modelSettings, \
                    snapshotSettings    = snapshotSettings, \
                    corrTimeSettings    = corrTimeSettings, \
                    mixingResults       = mixingResults, \
                    mi                  = MI_avg, \
                    miSystemMag         = MI_system, \
                    hx                  = HX, \
                    computeTime         = timer()-start_2 )
        result.saveToPickle(targetDirectory)

        if args.pairwise:
            #np.save(os.path.join(targetDirectory, f'full_snapshots_{now}.npy'), fullSnapshots)
            MI, corr = infoTheory.pairwiseMI_allNodes(model, nodes, fullSnapshots)
            #np.save(os.path.join(targetDirectory, f'MI_pairwise_nodes_{now}.npy'), MI)
            #np.save(os.path.join(targetDirectory, f'corr_pairwise_nodes_{now}.npy'), corr)
            #MIs_pairwise = np.array([np.nanmean(MI[i,:,:], axis=1) for i in range(MI.shape[0])])
            #now = time.time()
            #np.save(os.path.join(targetDirectory, f'MI_pairwise_{now}.npy'), MI)
            #np.save(os.path.join(targetDirectory, f'corr_pairwise_{now}.npy'), corr)

            #print(f'time for pairwise MI: {timer()-start_2 : .2f} seconds')

            result = IO.SimulationResult('pairwise', \
                        networkSettings     = networkSettings, \
                        modelSettings       = modelSettings, \
                        snapshotSettings    = snapshotSettings, \
                        corrTimeSettings    = corrTimeSettings, \
                        mixingResults       = mixingResults, \
                        mi                  = MI, \
                        corr                = corr, \
                        computeTime         = timer()-start_2 )
            result.saveToPickle(targetDirectory)



    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)

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

parser = argparse.ArgumentParser(description='run MC chain and compute MI based on the joint PDF of the central node and its neighbours')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('node', type=int, help='central node ID')
parser.add_argument('--maxDist', type=int, default=-1, help='max distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetititve runs')
parser.add_argument('--bins', type=int, default=100, help='number of bins for average magnetization of neighbours')
parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate MI')
parser.add_argument('--numSamples', type=int, default=1000, help='number of system samples')
parser.add_argument('--initState', type=int, default=-1, help='')
parser.add_argument('--magSide', type=str, default='', help='')



if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir

    # load network
    maxDist = args.maxDist
    graph = nx.read_gpickle(args.graph)
    N = len(graph)

    if args.maxDist > 0:
        maxDist = args.maxDist
    else:
        maxDist = nx.diameter(graph)

    node = args.node

    networkSettings = dict( \
        path = args.graph, \
        nNodes = N, \
        node = node
    )
    IO.saveSettings(targetDirectory, networkSettings, 'network')


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
        raise Exception('No mixing results found! Please run the mixing script first to determine the mixing and correlation time of the model.')
    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)


    snapshotSettingsJoint = dict( \
        nSamples    = args.numSamples, \
        repeats     = args.repeats, \
        burninSamples = burninSteps, \
        distSamples   = distSamples, \
        maxDist     = maxDist, \
        nBins       = args.bins
    )
    IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')



    for r in range(args.runs):

        #avgSnapshots, Z = infcy.getJointSnapshotsPerDist2(model, node, allNeighbours_G, **snapshotSettingsJoint, threads=nthreads)
        now = time.time()
        avgSnapshots, avgSystemSnapshots, snapshots = infcy.getJointSnapshotsPerDist2(model, node, allNeighbours_G, **snapshotSettingsJoint, threads=nthreads, initStateIdx=args.initState, getFullSnapshots=1)
        #np.save(os.path.join(targetDirectory, f'full_snapshots_{now}.npy'), snapshots)
        with open(os.path.join(targetDirectory, f'node_mapping_{now}.pickle'), 'wb') as f:
            pickle.dump(model.mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        MI, corr = infcy.runMIoneNode(model, node, snapshots.reshape((args.repeats*args.numSamples, -1)), distMax=maxDist)
        #MIs_pairwise = np.array([np.nanmean(MI[i,:,:], axis=1) for i in range(MI.shape[0])])
        np.save(os.path.join(targetDirectory, f'MI_pairwise_{now}.npy'), MI)
        np.save(os.path.join(targetDirectory, f'corr_pairwise_{now}.npy'), corr)


        Z = args.numSamples*args.repeats

        MI_avg, MI_system, HX = infcy.processJointSnapshots(avgSnapshots, avgSystemSnapshots, Z, node, maxDist)
        now = time.time()

        IO.savePickle(targetDirectory, f'MI_meanField_nodes_{now}', MI_avg)
        IO.savePickle(targetDirectory, f'HX_meanField_nodes_{now}', HX)
        IO.savePickle(targetDirectory, f'MI_systemMag_nodes_{now}', MI_system)



    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)

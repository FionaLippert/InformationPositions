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
parser.add_argument('nodes', type=str, help='path to numpy array of node IDs')
parser.add_argument('--maxDist', type=int, default=-1, help='max distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetititve runs')
parser.add_argument('--bins', type=int, default=10, help='number of bins for average magnetization of neighbours')
parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate MI')
parser.add_argument('--numSamples', type=int, default=1000, help='number of system samples')


if __name__ == '__main__':

    print("starting with average neighbour MI approach")

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir

    graph = nx.read_gpickle(args.graph)
    N = len(graph)

    # load network
    if args.maxDist > 0:
        maxDist = args.maxDist
    else:
        maxDist = nx.diameter(graph)

    if args.nodes == 'test':
        nodes = np.array([528, 529, 527, 530, 526, 496, 495, 497, 560, 559])
    else:
        nodes = np.load(args.nodes)
    #deg = graph.degree[node]

    networkSettings = dict( \
        path = args.graph, \
        nNodes = N, \
        nodes = args.nodes
    )
    IO.saveSettings(targetDirectory, networkSettings, 'network')


    # setup Ising model with nNodes spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = ''
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

        neighboursG, avgSnapshots, avgSystemSnapshots, fullSnapshots = infcy.getJointSnapshotsPerDistNodes(model, nodes, \
                                                                            **snapshotSettingsJoint, threads=nthreads, \
                                                                            initStateIdx=1, getFullSnapshots=1)

        start_2 = timer()

        MI, corr = infcy.runMI(model, nodes, fullSnapshots.reshape((args.repeats*args.numSamples, nodes.size)), distMax=maxDist)
        #MIs_pairwise = np.array([np.nanmean(MI[i,:,:], axis=1) for i in range(MI.shape[0])])
        now = time.time()
        np.save(os.path.join(targetDirectory, f'MI_pairwise_{now}.npy'), MI)
        np.save(os.path.join(targetDirectory, f'corr_pairwise_{now}.npy'), corr)

        print(f'time for pairwise MI: {timer()-start_2 : .2f} seconds')

        Z = args.numSamples * args.repeats
        """
        avgSnapshots = np.sum(avgSnapshots, axis=0)
        avgSystemSnapshots = np.sum(avgSystemSnapshots, axis=0)

        MIs_avg = np.zeros((nodes.size, maxDist))
        MIs_system = np.zeros(nodes.size)
        Hs = np.zeros(nodes.size)

        for n in range(nodes.size):

            MIs_avg[n,:] = [computeMI_jointPDF(avgSnapshots[n][d], Z) for d in range(maxDist)]
            MIs_system[n] = computeMI_jointPDF(avgSystemSnapshots[n], Z)
            Hs[n] = compute_spin_entropy(avgSystemSnapshots[n], Z)
        """

        start_2 = timer()
        MIs_avg, MIs_system, Hs = infcy.processJointSnapshotsNodes(avgSnapshots, avgSystemSnapshots, Z, nodes.size, maxDist)
        now = time.time()
        np.save(os.path.join(targetDirectory, f'MI_avg_{now}.npy'), MIs_avg)
        np.save(os.path.join(targetDirectory, f'MI_system_{now}.npy'), MIs_system)
        np.save(os.path.join(targetDirectory, f'H_nodes_{now}.npy'), Hs)

        print(f'time for avg MI: {timer()-start_2 : .2f} seconds')


    with open(f'{targetDirectory}/neighbours.pickle', 'wb') as f:
        pickle.dump(neighboursG, f)


    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)

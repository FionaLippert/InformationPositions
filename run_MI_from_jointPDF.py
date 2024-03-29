#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infoTheory, simulation
from Utils import IO
import networkx as nx, itertools, scipy, time, \
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
parser.add_argument('node', type=int, help='central node ID')
parser.add_argument('--maxDist', type=int, default=-1, help='max distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetititve runs')
parser.add_argument('--numSamples', type=int, default=1000, help='number of system samples')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=-1, help='initial system state')



if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    args = parser.parse_args()
    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    maxDist = args.maxDist if args.maxDist > 0 else nx.diameter(graph)
    networkSettings = dict( \
        path = args.graph, \
        size = N, \
        node = node
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
        subprocess.call(['python3', 'run_mixing.py', f'{T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']
    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)


    snapshotSettingsJoint = dict( \
        nSamples    = args.numSamples, \
        burninSteps = burninSteps, \
        distSamples   = distSamples, \
        maxDist     = maxDist
    )


    for r in range(args.runs):

        now = time.time()
        snapshots, Z = simulation.getJointSnapshotsPerDist(model, node, allNeighbours_G, **snapshotSettingsJoint, threads=nthreads, initStateIdx=args.initState)

        with open(os.path.join(targetDirectory, f'node_mapping_{now}.pickle'), 'wb') as f:
            pickle.dump(model.mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

        MIs = []
        for d in range(maxDist):
            MI, HX, jointPDF, states = infoTheory.computeMI_jointPDF_fromDict(snapshots[d], Z)
            MIs.append(MI)
            P_Y = np.sum(jointPDF, axis=0)/Z
            H_XgivenY = [stats.entropy(jointPDF[:,i]/np.sum(jointPDF[:,i]), base=2) for i, s in enumerate(states)]
            for i, s in enumerate(states):
                print(np.frombuffer(s).astype(int), stats.entropy(jointPDF[:,i], base=2), P_Y[i])
            print(f'MI = {MI}')


        now = time.time()
        np.save(os.path.join(targetDirectory, f'MI_jointPDF_{now}.npy'), np.array(MIs))


        print(f'time elapsed: {timer()-start : .2f} seconds')

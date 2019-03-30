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
parser.add_argument('maxD', type=int, help='max distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetititve runs')
parser.add_argument('--bins', type=int, default=10, help='number of bins for average magnetization of neighbours')

def computeMI_joint(jointSnapshots, maxDist, Z):
    MIs = np.zeros(maxDist)
    for d in range(maxDist):
        P_XY = jointSnapshots[d].flatten()/Z
        P_X = np.sum(jointSnapshots[d], axis=1)/Z # sum over all bins
        P_Y = np.sum(jointSnapshots[d], axis=0)/Z # sum over all spin states

        #print(P_XY, P_Y, P_X)

        MI = stats.entropy(P_X, base=2) + stats.entropy(P_Y, base=2) - stats.entropy(P_XY, base=2)
        MIs[d] = MI
    return MIs




if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir

    # load network
    maxDist = args.maxD
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    
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

        """
        # determine mixing/correlation time
        mixingTimeSettings = dict( \
            nInitialConfigs = 10, \
            burninSteps  = 10, \
            nStepsRegress   = int(1e3), \
            nStepsCorr      = int(1e4), \
            thresholdReg    = 0.05, \
            thresholdCorr   = 0.01
        )
        IO.saveSettings(targetDirectory, mixingTimeSettings, 'mixingTime')
        mixingTime, meanMag, distSamples, mags = infcy.determineCorrTime(model, **mixingTimeSettings)
        print(f'correlation time = {distSamples}')
        print(f'mixing time      = {mixingTime}')
        print(f'mag level        = {meanMag}')
        
        mixingTime = min(mixingTime, 10000)

        mixingResults = dict(\
            mixingTime = mixingTime, \
            corrTime = distSamples, \
            magLevel = meanMag
        )
        IO.saveResults(targetDirectory, mixingResults, 'mixingResults')

        mixingTime = min(mixingTime, 5000)


        corrTimeSettings = dict( \
            nInitialConfigs = 10, \
            burninSteps  = mixingTime, \
            nStepsCorr      = int(1e4), \
            thresholdCorr   = 0.01, \
            checkMixing     = 0
        )
        IO.saveSettings(targetDirectory, corrTimeSettings, 'corrTime')
        """

    #allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(model.mapping[node], maxDist)
    
    
    snapshotSettingsJoint = dict( \
        nSamples    = int(1e3), \
        repeats     = int(1e2), \
        burninSamples = burninSteps, \
        distSamples   = distSamples, \
        maxDist     = maxDist, \
        nBins       = args.bins
    )
    IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')

    #joint_dir = os.path.join(targetDirectory, f'MI_joint_T={model.t}')
    #avg_dir = os.path.join(targetDirectory, f'MI_avg')
    #if not os.path.isdir(joint_dir): os.mkdir(joint_dir)
    #if not os.path.isdir(avg_dir): os.mkdir(avg_dir)

    for r in range(args.runs):

        avgSnapshots, Z = infcy.getJointSnapshotsPerDistNodes(model, nodes, **snapshotSettingsJoint, threads=nthreads)
        #print(f'Z={Z}')
        #Z = snapshotSettingsJoint['nSamples'] * snapshotSettingsJoint['repeats']

        #with open(f'{targetDirectory}/jointSnapshots_node={node}.pickle', 'wb') as f:
        #    pickle.dump(jointSnapshots, f)
        #with open(f'{targetDirectory}/avgSnapshots_node={node}.pickle', 'wb') as f:
        #    pickle.dump(avgSnapshots, f)
    
        now = time.time()

        MIs_avg = np.zeros((nodes.size, maxDist))
        for n in range(nodes.size):
            MIs_avg[n,:] = computeMI_joint(avgSnapshots[n], maxDist, Z)
        np.save(os.path.join(targetDirectory, f'MI_avg_{now}.npy'), MIs_avg)
        #print(MIs)
        #print(MIs_avg)
    

    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)

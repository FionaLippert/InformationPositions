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


parser = argparse.ArgumentParser(description='run MC chain and compute MI based on conditional PDF of the central node with neighbour states fixed')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('z', type=int, help='degree of star graph')
parser.add_argument('--minT', type=float, default=0)
parser.add_argument('--maxT', type=float, default=10)
parser.add_argument('--numT', type=int, default=10)
parser.add_argument('--depth', type=int, default=1, help='depth of star path graph')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')
parser.add_argument('--burninSteps', type=int, default=100, help='steps to reach equilibrium')
parser.add_argument('--distSamples', type=int, default=100, help='distance between two samples in the MC chain')
parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate MI')
parser.add_argument('--numSamplesCond', type=int, default=1000, help='number of samples per MC run with fixed neighbour states')
parser.add_argument('--numSamplesJoint', type=int, default=10000, help='number of samples per MC run')
parser.add_argument('--bins', type=int, default=100, help='number of bins')
parser.add_argument('--fixMag', action='store_true', help='initial system state')




def computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, nTrials, nSamples, modelSettings):
    MIs = []
    HXs = []
    subgraph_nodes = [node]
    for d in range(1, maxDist+1):
        # get subgraph and outer neighbourhood at distance d
        if len(neighbours_G[d]) > 0:

            subgraph = nx.ego_graph(model.graph, node, d)

            if d >= minDist:
                print(f'------------------- distance d={d}, num neighbours = {len(neighbours_G[d])}, subgraph size = {len(subgraph)}, num states = {len(snapshots[d-1])} -----------------------')

                model_subgraph = fastIsing.Ising(subgraph, **modelSettings)

                threads = nthreads if len(subgraph_nodes) > 20 else 1

                initState = 1 #if args.fixMag else -1


                _, _, MI, HX, HXgiveny, keys, probs = infcy.neighbourhoodMI(model_subgraph, node, \
                                d, neighbours_G, snapshots[d-1], nTrials, \
                                args.burninSteps, nSamples, \
                                args.distSamples, threads=threads, \
                                initStateIdx=initState, uniformPDF=1, out='MI')
                print(HXgiveny)

                MIs.append(MI)
                HXs.append(HX)
                print(MIs)

    return MIs, HXs



if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    targetDirectory = os.path.join(args.dir, f'z={args.z}')
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    #graph = nx.read_gpickle(f'networkData/undirected_star_z={args.z}.gpickle')
    graph = nx.Graph()
    graph.add_star(range(args.z+1))
    if args.depth > 1:
        for node in range(1, args.z+1):
            path_nodes = [node]
            path_nodes.extend(range(len(graph), len(graph)+args.depth))
            graph.add_path(path_nodes)

    N = len(graph)
    node = 0
    nodes = np.array(list(graph.nodes()))

    maxDist=args.depth
    minDist=1


    nTrials = args.repeats
    nSamples = args.numSamplesCond

    temps=np.linspace(args.minT, args.maxT, args.numT)

    #MIs = np.zeros((temps.size, args.depth))
    MIs_cond = {}
    MIs_avg = {}

    type = 'fixedMag' if args.fixMag else 'fairMag'

    for r in range(args.runs):

        for i, T in enumerate(temps):
            now = time.time()

            # setup Ising model with nNodes spin flip attempts per simulation step
            modelSettings = dict( \
                temperature     = T, \
                updateType      = 'async' ,\
                magSide         = ''
            )
            model = fastIsing.Ising(graph, **modelSettings)
            allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)

            # generate all possible snapshots
            snapshots = []
            for d in range(maxDist):
                num_neighbours = len(allNeighbours_G[d+1])
                prob = 1/np.power(2, num_neighbours)
                print(prob)
                s = itertools.product([-1,1], repeat=num_neighbours)
                print(list(s))
                if args.fixMag:
                    s = {np.array(state).astype(float).tobytes() : prob for state in itertools.product([-1,1], repeat=num_neighbours) if np.mean(state) > 0}
                else:
                    s = {np.array(state).astype(float).tobytes() : prob for state in itertools.product([-1,1], repeat=num_neighbours)}
                snapshots.append(s)

            MI, HX = computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, nTrials, nSamples, modelSettings)
            MIs_cond[T] = MI

            snapshotSettingsJoint = dict( \
                nSamples    = args.numSamplesJoint, \
                repeats     = args.repeats, \
                burninSamples = args.burninSteps, \
                distSamples   = args.distSamples, \
                maxDist     = maxDist, \
                nBins       = args.bins
            )
            IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')

            allNeighbours_G_allNodes = model.neighboursAtDistAllNodes(nodes, maxDist)

            avgSnapshots, avgSystemSnapshots, fullSnapshots = infcy.getJointSnapshotsPerDistNodes(model, nodes, \
                                                                                allNeighbours_G_allNodes, \
                                                                                **snapshotSettingsJoint, threads=nthreads, \
                                                                                initStateIdx=1, getFullSnapshots=1)


            MI, corr = infcy.runMI(model, nodes, fullSnapshots.reshape((args.repeats*args.numSamplesJoint, -1)), distMax=maxDist)
            np.save(os.path.join(targetDirectory, f'MI_pairwise_nodes_{now}.npy'), MI)
            np.save(os.path.join(targetDirectory, f'corr_pairwise_nodes_{now}.npy'), corr)

            Z = args.numSamplesJoint * args.repeats

            MI_avg, MI_system, HX = infcy.processJointSnapshotsNodes(avgSnapshots, Z, nodes, maxDist, avgSystemSnapshots)
            MIs_avg[T] = MI_avg


        with open(f'{targetDirectory}/MI_cond_Ts_{now}.pickle', 'wb') as f:
            pickle.dump(MIs_cond, f)
        with open(f'{targetDirectory}/MI_meanField_Ts_{now}.pickle', 'wb') as f:
            pickle.dump(MIs_avg, f)

        #np.save(f'{targetDirectory}/MI_cond_{now}_{type}.npy', MIs)
    #np.save(f'{targetDirectory}/temps_{now}.npy', temps)

    print(f'time elapsed: {timer()-start : .2f} seconds')

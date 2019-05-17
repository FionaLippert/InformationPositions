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


parser = argparse.ArgumentParser(description='run MC chain and compute MI based on conditional PDF of the central node with neighbour states fixed')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('node', type=int, help='central node ID')
parser.add_argument('maxDist', type=int, help='max distance to central node. If -1, use diameter, if -2 use distance where max neighbours are reached')
parser.add_argument('--minDist', type=int, default=1, help='min distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')
parser.add_argument('--maxCorrTime', type=int, default=-1, help='max distance between two samples in the MC')
parser.add_argument('--minCorrTime', type=int, default=1, help='min distance between two samples in the MC')
parser.add_argument('--snapshots', type=int, default=100, help='number of neighbourhood snapshots')
parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate MI')
parser.add_argument('--numSamples', type=int, default=1000, help='number of samples per MC run with fixed neighbour states')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=-1, help='initial system state')
parser.add_argument('--getStates', action="store_true", help='get system states instead of MI')
parser.add_argument('--getStateDistr', action="store_true", help='get distributions of neighbour vectors instead of MI')
parser.add_argument('--uniformPDF', action="store_true", help='assume uniform distribution over neighbourhood snapshots')
parser.add_argument('--generateSnapshots', action="store_true", help='generate all possible snapshots instead of sampling them from equilibrium')




def computeMI_cond(model, node, minDist, maxDist, neighbours_G, snapshots, nTrials, nSamples, modelSettings, corrTimeSettings):
    MIs             = []
    HXs             = []
    all_HXgiveny    = []
    all_keys        = []
    all_states      = {}
    all_stateDistr  = {}
    all_mappings    = {}

    subgraph_nodes = [node]
    for d in range(1, maxDist+1):
        # get subgraph and outer neighbourhood at distance d
        if d in neighbours_G.keys():
            subgraph_nodes.extend(neighbours_G[d])
            subgraph = nx.ego_graph(model.graph, node, d)


            if d >= minDist:
                print(f'------------------- distance d={d}, num neighbours = {len(neighbours_G[d])}, subgraph size = {len(subgraph_nodes)}, num states = {len(snapshots[d-1])} -----------------------')
                #print(subgraph_nodes)
                model_subgraph = fastIsing.Ising(subgraph, **modelSettings)
                all_mappings[d] = model_subgraph.mapping
                #print(model_subgraph.mapping)

                #print(f'neighbours G: {neighbours_G[d]}')
                #print(f'states to fix: {[np.frombuffer(s).astype(int) for s in snapshots[d-1]]}')


                # determine correlation time for subgraph Ising model
                if args.maxCorrTime == args.minCorrTime:
                    distSamples_subgraph = args.maxCorrTime
                    mixingTime_subgraph = corrTimeSettings['burninSteps']
                else:
                    mixingTime_subgraph, meanMag, distSamples_subgraph, _ = infcy.determineCorrTime(model_subgraph, nodeG=node, **corrTimeSettings)
                    if args.maxCorrTime > 0: distSamples_subgraph = min(distSamples_subgraph, args.maxCorrTime)
                    distSamples_subgraph = max(distSamples_subgraph, args.minCorrTime)
                print(f'correlation time = {distSamples_subgraph}')
                print(f'mixing time      = {mixingTime_subgraph}')

                #mixingTime_subgraph = 500

                threads = nthreads #if len(subgraph_nodes) > 20 else 1

                if args.getStates:
                    _, states = infcy.neighbourhoodMI(model_subgraph, node, d, neighbours_G, snapshots[d-1], \
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, \
                          threads=threads, initStateIdx=args.initState, out='states')
                    all_states[d] = states

                elif args.getStateDistr:
                    _, stateDistr = infcy.neighbourhoodMI(model_subgraph, node, d, neighbours_G, snapshots[d-1], \
                          nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, \
                          threads=threads, initStateIdx=args.initState, out='stateDistr')
                    print(stateDistr)
                    all_stateDistr[d] = stateDistr

                else:
                    _, _, MI, HX, HXgiveny, keys, probs = infcy.neighbourhoodMI(model_subgraph, node, d, neighbours_G, snapshots[d-1], \
                              nTrials=nTrials, burninSamples=mixingTime_subgraph, nSamples=nSamples, distSamples=distSamples_subgraph, \
                              threads=threads, initStateIdx=args.initState, uniformPDF=args.uniformPDF, out='MI')

                    MIs.append(MI)
                    HXs.append(HX)
                    all_keys.append(keys)
                    all_HXgiveny.append(HXgiveny)
                    print(MIs)
                    for i in range(len(snapshots[d-1])):
                        print(np.frombuffer(keys[i]).astype(int), HXgiveny[i], probs[i])


    if args.getStates:
        return all_states, all_mappings
    elif args.getStateDistr:
        return all_stateDistr, all_mappings
    else:
        return MIs, HXs, all_HXgiveny, all_keys



if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    node = args.node
    deg = graph.degree[node]

    if args.maxDist > 0:
        maxDist = args.maxDist
    else:
        maxDist = nx.diameter(graph)

    networkSettings = dict( \
        path = args.graph, \
        nNodes = N, \
        node = node, \
        degree = deg
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

    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)

    if args.maxDist == -2:
        # find distance with max number of neighbours
        maxDist = np.argmax([len(allNeighbours_G[d]) for d in range(1, max(allNeighbours_G.keys())+1)]) + 1

    snapshotSettingsCond = dict( \
        nSnapshots    = args.snapshots, \
        burninSamples = burninSteps, \
        maxDist     = maxDist
    )
    IO.saveSettings(targetDirectory, snapshotSettingsCond, 'snapshots')

    modelSettingsCond = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = ''
    )


    #with open(f'{targetDirectory}/neighboursG_node={node}.pickle', 'wb') as f:
    #    pickle.dump(allNeighbours_G, f)

    minDist = args.minDist
    nTrials = args.repeats
    nSamples = args.numSamples


    for i in range(args.runs):
        now = time.time()

        if args.generateSnapshots:
            snapshots = []
            for d in range(maxDist):
                num_neighbours = len(allNeighbours_G[d+1])
                prob = 1/np.power(2, num_neighbours)
                print(prob)
                if args.magSide == 'pos':
                    s = {np.array(state).astype(float).tobytes() : prob for state in itertools.product([-1,1], repeat=num_neighbours) if np.mean(state) >= 0}
                elif args.magSide == 'neg':
                    s = {np.array(state).astype(float).tobytes() : prob for state in itertools.product([-1,1], repeat=num_neighbours) if np.mean(state) <= 0}
                else:
                    s = {np.array(state).astype(float).tobytes() : prob for state in itertools.product([-1,1], repeat=num_neighbours)}
                snapshots.append(s)
        else:
            threads = nthreads if len(model.graph) > 20 else 1
            print(f'snapshots = {snapshotSettingsCond["nSnapshots"]}')
            print(f'samples = {snapshotSettingsCond["nSamples"]}')
            snapshots, _ , HX_eq  = infcy.getSnapshotsPerDist(model, node, allNeighbours_G, **snapshotSettingsCond, threads=threads, initStateIdx=args.initState)
            with open(f'{targetDirectory}/snapshots_{now}.pickle', 'wb') as f:
                pickle.dump(snapshots, f)
            print(stats.entropy(HX_eq, base=2))

            #np.save(f'{targetDirectory}/system_states_{now}.npy', system_states)

        if args.getStates:
            states, mappings = computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, nTrials, nSamples, modelSettingsCond, corrTimeSettings)
            with open(f'{targetDirectory}/subsystem_states_{now}.pickle', 'wb') as f:
                pickle.dump(states, f)
            with open(f'{targetDirectory}/subsystem_mappings_{now}.pickle', 'wb') as f:
                pickle.dump(mappings, f)

        elif args.getStateDistr:
            stateDistr, mappings = computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, nTrials, nSamples, modelSettingsCond, corrTimeSettings)
            with open(f'{targetDirectory}/neighbour_vector_distributions_{now}.pickle', 'wb') as f:
                pickle.dump(stateDistr, f)
            with open(f'{targetDirectory}/subsystem_mappings_{now}.pickle', 'wb') as f:
                pickle.dump(mappings, f)
        else:
            MI, HX, H_XgivenY, keys = computeMI_cond(model, node, minDist, maxDist, allNeighbours_G, snapshots, nTrials, nSamples, modelSettingsCond, corrTimeSettings)
            np.save(f'{targetDirectory}/MI_cond_{now}.npy', np.array(MI))
            np.save(f'{targetDirectory}/HX_{now}.npy', np.array(HX))

            #for d in range(minDist, maxDist+1):
            #    np.save(os.path.join(targetDirectory, f'HXgivenY_d={d}_{now}.npy'), H_XgivenY[d-minDist])
            #    np.save(os.path.join(targetDirectory, f'states_d={d}_{now}.npy'), keys[d-minDist])

    print(f'time elapsed: {timer()-start : .2f} seconds')

    #print(targetDirectory)

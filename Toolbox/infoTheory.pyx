# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Fiona Lippert'

from Models.models cimport Model

import time
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport parallel, prange, threadid
import multiprocessing as mp
from scipy import stats, special
import itertools

# cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.math cimport exp, log2, sqrt
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf
import ctypes
from timeit import default_timer as timer


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double entropyEstimateH2(long[::1] counts):
    """
    implementation of the H_2 entropy estimate from (Schürmann, 2015)

    Input:
        :counts: frequencies per state (number of observations, not probabilites!)
    Output:
        :H: entropy estimate (using log base 2)
    """
    cdef:
        long M = counts.shape[0]
        long N = np.sum(counts)
        long i, j
        double osc, H = 0

    H = 0
    for i in range(M):
        osc = np.sum([((-1)**j)/float(j) for j in range(1,counts[i])])
        H += counts[i]/float(N) * (special.digamma(N) - special.digamma(counts[i]) + np.log(2) + osc)

    H = H * np.log2(np.exp(1)) # transform from natural log to log base 2
    return H

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double entropyFromProbs(double[::1] probs) nogil:
    """
    naive plug-in entropy estimate

    Input:
        :probs: probabilities per state
    Output:
        :H: entropy estimate
    """
    cdef:
        double p, H = 0
        long i, n = probs.shape[0]

    for i in range(n):
        p = probs[i]
        if p > 0:
            H = H - p * log2(p)
    return H


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef double[::1] binaryEntropies(long[:,::1] snapshots):
    """
    estimate binary spin entropies from system snapshots

    Input:
        :snapshots: 2D array of shape [nSamples, nNodes] containing sampled Ising states (-1 or +1) of all nodes in the system
    Output:
        :H: array of entropy estimates for all nodes
    """
    cdef:
        np.ndarray tmp, H = np.sum(snapshots.base, axis=0, dtype=float)
        long length = snapshots.shape[0]
        double[::1] cview_H

    H = (length - np.abs(H))/2. + np.abs(H)
    H = H/length

    tmp = 1-H
    H = - H * np.log2(H) - tmp * np.log2(np.where(tmp==0, 1, tmp)) # compute entropy for each node (use 0*log(0) = 0)

    cview_H = H

    return cview_H


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef double pairwiseMI(long[:,::1] snapshots, double[::1] binEntropies, long nodeIdx1, long nodeIdx2) nogil:
    """
    compute mutual information for the given pair of nodes

    Input:
        :snapshots: 2D array of shape [nSamples, nNodes] containing sampled Ising states (-1 or +1) of all nodes in the system
        :binEntropies: spin entropies of all node in the system
        :nodeIdx1: index of first node (in snapshots and binEntropies array)
        :nodeIdx2: index of second node (in snapshots and binEntropies array)

    Output:
        :mi: mutual information between the two nodes
    """
    cdef:
        long idx, nSamples = snapshots.shape[0]
        vector[long] states
        vector[long] jointDistr = vector[long](nSamples, 0)
        double mi, jointEntropy


    for idx in range(nSamples):
        jointDistr[idx] = snapshots[idx][nodeIdx1] + snapshots[idx][nodeIdx2]*2 # -3,-1,1,3 represent the 4 possible combinations of spins

    with gil: jointEntropy = stats.entropy(np.unique(jointDistr, return_counts=True)[1]/nSamples, base=2)

    # mutual information
    mi = binEntropies[nodeIdx1] + binEntropies[nodeIdx2] - jointEntropy

    return mi


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef double spinCorrelation(long[:,::1] snapshots, long nodeIdx1, long nodeIdx2) nogil:
    """
    compute correlation <s_1 * s_2> - <s_1> * <s_2> for the given pair of nodes

    Input:
        :snapshots: 2D array of shape [nSamples, nNodes] containing sampled Ising states (-1 or +1) of all nodes in the system
        :nodeIdx1: index of first node (in snapshots and binEntropies array)
        :nodeIdx2: index of second node (in snapshots and binEntropies array)

    Output:
        :corr: correlation between the two nodes
    """
    cdef:
        long idx, nSamples = snapshots.shape[0]
        double avgNode1 = 0
        double avgNode2 = 0
        double avgProduct = 0
        double corr


    for idx in range(nSamples):
        avgNode1 += snapshots[idx][nodeIdx1]
        avgNode2 += snapshots[idx][nodeIdx2]
        avgProduct += snapshots[idx][nodeIdx1]*snapshots[idx][nodeIdx2]
        #with gil: print(snapshots[idx][node1], snapshots[idx][node2])

    # spin-spin correlation
    corr = avgProduct/nSamples - (avgNode1 * avgNode2)/(nSamples**2)

    return corr



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple computeMI_jointPDF(np.ndarray jointDistr, long Z):
    """
    compute mutual information from joint probability distribution

    Input:
        :jointDistr: 2D array containing the joint distribution over two
                     random variables (using frequencies, no probabilities)
        :Z: normalization constant to obtain probabilities

    Output:
        :MI: mutual information between the two RVs
        :H_X: entropy of the first RV
    """

    cdef:
        np.ndarray P_XY, P_X, P_Y
        double MI, H_X

    P_XY = jointDistr.flatten()/Z
    P_X = np.sum(jointDistr, axis=1)/Z # sum over all bins
    P_Y = np.sum(jointDistr, axis=0)/Z # sum over all spin states
    H_X = stats.entropy(P_X, base=2)
    MI = stats.entropy(P_X, base=2) + stats.entropy(P_Y, base=2) - stats.entropy(P_XY, base=2)
    return MI, H_X


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple computeMI_jointPDF_fromDict(unordered_map[int, unordered_map[string, long]] jointDistrDict, long Z):
    """
    compute mutual information from joint probability distribution given in dictionary form

    Input:
        :jointDistrDict: dictionary containing the joint distribution over two
                         random variables (using frequencies, no probabilities)
        :Z: normalizing constant to obtain probabilities

    Output:
        :MI: mutual information between the two RVs
        :H_X: entropy of the first RV
        :jointProbs: jointPDF extracted from dictionary of observations
        :states: array of all observed states of the second RV
    """
    cdef:
        np.ndarray P_XY, P_X, P_Y, jointProbs, states
        double MI, H_X, p
        dict d

    states = np.unique([s for d in dict(jointDistrDict).values() for s in d.keys()])
    jointProbs = np.array([[d[s] if s in d else 0 for s in states] for d in dict(jointDistrDict).values()])

    P_XY = jointProbs.flatten()/Z
    P_X = np.sum(jointProbs, axis=1)/Z
    P_Y = np.sum(jointProbs, axis=0)/Z # sum over all spin states
    H_X = stats.entropy(P_X, base=2)

    MI = stats.entropy(P_X, base=2) + stats.entropy(P_Y, base=2) - stats.entropy(P_XY, base=2)
    if MI == 0: MI = np.nan

    return MI, H_X, jointProbs, states


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple pairwiseMI_allNodes(Model model, np.ndarray nodesG, long[:,::1] snapshots, \
                int threads = -1):
    """
    compute mutual information for all pairs of nodes

    Input:
        :model: model used to extract snapshots
        :nodesG: array containing all nodes to be considered (using graph indexing)
        :snapshots: 2D array of shape [nSamples, nNodes] containing sampled
                    Ising states (-1 or +1) of all nodes in the system
        :threads: number of threads to use. If -1, use all threads available

    Output:
        :MI: 2D array of pairwise mutual information estimates
        :corr: 2D array of pairwise correlations
    """
    cdef:
        double[::1] entropies
        long i, n1, n2, d, nNodes = nodesG.shape[0]
        long[::1] nodesIdx = np.array([model.mapping[n] for n in nodesG])
        double[:,::1] MI = np.zeros((nNodes, nNodes))
        double[:,::1] corr = np.zeros((nNodes, nNodes))
        int nThreads = mp.cpu_count() if threads == -1 else threads

    entropies = binaryEntropies(snapshots)

    for n1 in prange(nNodes, nogil = True, schedule = 'dynamic', num_threads = nThreads):
        with gil: print(f'processing node {n1}')
        for n2 in range(n1, nNodes):
            MI[n1][n2] = pairwiseMI(snapshots, entropies, nodesIdx[n1], nodesIdx[n2])
            MI[n2][n1] = MI[n1][n2] # symmetric
            corr[n1][n2] = spinCorrelation(snapshots, nodesIdx[n1], nodesIdx[n2])
            corr[n2][n1] = corr[n1][n2] # symmetric

    return MI.base, corr.base

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple pairwiseMI_oneNode(Model model, long nodeG, long[:,::1] snapshots, \
                int threads = -1):
    """
    compute mutual information between the given node and all other nodes in the system

    Input:
        :model: model used to extract snapshots
        :nodeG: node of interest (using graph indexing)
        :snapshots: 2D array of shape [nSamples, nNodes] containing sampled
                    Ising states (-1 or +1) of all nodes in the system
        :threads: number of threads to use. If -1, use all threads available

    Output:
        :MI: 1D array of pairwise mutual information estimates
        :corr: 1D array of pairwise correlations
    """
    cdef:
        double[::1] entropies
        long i, n, d
        long nodeIdx = model.mapping[nodeG]
        double[::1] MI = np.zeros(model._nNodes)
        double[::1] corr = np.zeros(model._nNodes)
        int nThreads = mp.cpu_count() if threads == -1 else threads

    entropies = binaryEntropies(snapshots)

    for n in prange(model._nNodes, nogil = True, schedule = 'dynamic', num_threads = nThreads):
        MI[n] = pairwiseMI(snapshots, entropies, nodeIdx, n)
        corr[n] = spinCorrelation(snapshots, nodeIdx, n)

    return MI.base, corr.base


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple processJointSnapshots_allNodes(np.ndarray avgSnapshots, long Z, np.ndarray nodesG, long maxDist, np.ndarray avgSystemSnapshots=None):
    """
    extract mutual information quantities from joint distributions over spin states and avg magnetization of neighbourhoods / the entire system

    Input:
        :snapshots: array containing the joint distribution over two
                    random variables (e.g. individual spin state, and average
                    neighbourhood magnetization)
                    shape: [#Nodes, #Distances, #spinStates, #bins]
        :Z: normalization constant to obtain probabilities
        :nodesG: all nodes in the snapshot array (using graph indexing)
        :avgSystemSnapshots: optional. array containing the joint distribution over two
                             random variables (individual spin state, and average
                             system magnetization)
                             shape: [#Nodes, #spinStates, #bins]

    Output:
        :MI_avg: dict mapping nodes to array containing MI estimates per distance
        :MI_system: dict mapping nodes to MI with the avg system magnetization
    """
    cdef:
        long nNodes = nodesG.size
        MI_avg = {} #np.zeros((nNodes, maxDist))
        MI_system = {} #np.zeros(nNodes)
        HX = {} #np.zeros(nNodes)
        long n, d

    avgSnapshots = np.sum(avgSnapshots, axis=0)
    if avgSystemSnapshots is not None: avgSystemSnapshots = np.sum(avgSystemSnapshots, axis=0)

    for n in range(nNodes):
        MI_avg[nodesG[n]] = np.zeros(maxDist)
        for d in range(maxDist):
            MI_avg[nodesG[n]][d] = computeMI_jointPDF(avgSnapshots[n][d], Z)[0]
        if avgSystemSnapshots is not None: MI_system[nodesG[n]], HX[nodesG[n]] = computeMI_jointPDF(avgSystemSnapshots[n], Z)

    return MI_avg, MI_system, HX


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple processJointSnapshots_oneNode(np.ndarray avgSnapshots, np.ndarray avgSystemSnapshots, long Z, long maxDist):
    """
    extract mutual information quantities from joint distributions over spin states and avg magnetization of neighbourhoods / the entire system

    Input:
        :snapshots: array containing the joint distribution over two
                    random variables (e.g. individual spin state, and average
                    neighbourhood magnetization)
                    shape: [#Distances, #spinStates, #bins]
        :Z: normalization constant to obtain probabilities
        :avgSystemSnapshots: array containing the joint distribution over two
                             random variables (individual spin state, and average
                             system magnetization)
                             shape: [#spinStates, #bins]

    Output:
        :MI_avg: array containing MI estimates per distance
        :MI_system: MI with the avg system magnetization
    """
    cdef:
        long d
        np.ndarray MI_avg = np.zeros(maxDist)
        double MI_system, HX

    avgSnapshots = np.sum(avgSnapshots, axis=0)
    avgSystemSnapshots = np.sum(avgSystemSnapshots, axis=0)

    for d in range(maxDist):
        MI_avg[d] = computeMI_jointPDF(avgSnapshots[d], Z)[0]
    MI_system, HX = computeMI_jointPDF(avgSystemSnapshots, Z)

    return MI_avg, MI_system, HX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double pCond_theory(int dist, double beta, int num_children, int node_state, np.ndarray neighbour_states):
    cdef:
        double p, sum, prod
        long idx, n

    if dist == 0:
        return 0
    if dist == 1:
        p = np.exp(beta * node_state * np.sum(neighbour_states))/(np.exp(beta * node_state * np.sum(neighbour_states)) + np.exp(-beta * node_state * np.sum(neighbour_states)))
        return p
    else:
        sum = 0
        for states in itertools.product([1,-1], repeat=num_children):
            P_node_given_children = np.exp(beta * node_state * np.sum(states))/(np.exp(beta * node_state * np.sum(states)) + np.exp(-beta * node_state * np.sum(states)))
            prod = 1
            for idx, n in enumerate(states):
                prod *= pCond_theory(dist-1, beta, num_children, n, neighbour_states[num_children*idx:num_children*(idx+1)])
            sum += P_node_given_children * prod
        return sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double MI_tree_theory(int depth, double beta, int num_children):
    """
    compute mutual information for increasing distances for the root of a regular tree
    assuming neighbour states on which I condition are uniformly distributed

    Input:
        :depth: max tree depth to consider
        :beta: inverse temperature of the Ising model
        :num_children: number of children per node in the tree

    Output:
        :MI: array containing MI per distance
    """
    cdef:
        double pX1, MI, HX = 1 # uniformly distributed
        double HXgivenY = 0
        long s, num_neighbour_states = 2**(num_children**depth)

    for states in itertools.product([1,-1], repeat=num_children**depth):

        s = np.sum(states)
        states = np.array(states)/s if (s != 0) else np.array(states)
        pX1 = pCond_theory(depth, beta, num_children, 1, states)
        HXgivenY += stats.entropy([pX1, 1-pX1], base =2)

    MI = HX - HXgivenY/num_neighbour_states # states are also uniformly distributed
    return MI

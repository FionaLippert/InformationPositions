# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Casper van Elteren'

# MODELS
# from Models.models cimport Model
from Models.models cimport Model
from Models.fastIsing cimport Ising

import time
import numpy as np
cimport numpy as np
cimport cython
import time
from cython.parallel cimport parallel, prange, threadid
import multiprocessing as mp
import copy
from cpython.ref cimport PyObject

from scipy.stats import linregress
from scipy.signal import correlate
import scipy
from scipy import stats, special
import itertools

# progressbar
from tqdm import tqdm   #progress bar

# cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport srand, malloc, free
from libc.math cimport exp, log2, sqrt
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf
import ctypes
from cython.view cimport array as cvarray
from timeit import default_timer as timer
from cython.operator cimport dereference as deref, preincrement as prec
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cdef extern from *:
    """
    #include <Python.h>
    #include <mutex>

    std::mutex ref_mutex;

    class PyObjectHolder{
    public:
        PyObject *ptr;
        PyObjectHolder():ptr(nullptr){}
        PyObjectHolder(PyObject *o):ptr(o){
            std::lock_guard<std::mutex> guard(ref_mutex);
            Py_XINCREF(ptr);
        }
        //rule of 3
        ~PyObjectHolder(){
            std::lock_guard<std::mutex> guard(ref_mutex);
            Py_XDECREF(ptr);
        }
        PyObjectHolder(const PyObjectHolder &h):
            PyObjectHolder(h.ptr){}
        PyObjectHolder& operator=(const PyObjectHolder &other){
            {
                std::lock_guard<std::mutex> guard(ref_mutex);
                Py_XDECREF(ptr);
                ptr=other.ptr;
                Py_XINCREF(ptr);
            }
            return *this;

        }
    };
    """
    cdef cppclass PyObjectHolder:
        PyObject *ptr
        PyObjectHolder(PyObject *o) nogil




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef int encodeStateToAvg(long[::1] states, vector[long] nodes, double[::1] bins) nogil:
    """Maps states of given nodes to binned avg magnetization"""
    cdef:
        long N = nodes.size(), nBins = bins.shape[0]
        double avg = 0
        long i, n

    for i in range(N):
        avg += states[nodes[i]]

    avg /= N

    for i in range(nBins):
        if avg <= bins[i]:
            avg = i
            break

    return <int>avg


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

cpdef double MI_tree_theory(int depth, double beta, int num_children, int avg=0):
    cdef:
        double pX1, MI, HX = 1 # uniformly distributed
        double HXgivenY = 0
        long s, num_neighbour_states = 2**(num_children**depth)

    for states in itertools.product([1,-1], repeat=num_children**depth):

        s = np.sum(states)
        states = np.array(states)/s if (s != 0 and avg) else np.array(states)
        pX1 = pCond_theory(depth, beta, num_children, 1, states)
        HXgivenY += stats.entropy([pX1, 1-pX1], base =2)
        #print(f'p = {pX1}, H = {HXgivenY}')
        #print(f' ricks theory p = {(1+(2*np.exp(beta)/(np.exp(beta)+np.exp(-beta))-1)**depth)/2}')

    #print(f'H = {HXgivenY/num_neighbour_states}')

    MI = HX - HXgivenY/num_neighbour_states # states are also uniformly distributed
    return MI


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double entropyEstimateH2(long[::1] counts):
    cdef:
        long M = counts.shape[0]
        long N = np.sum(counts)
        long i, j
        double osc, H = 0

    H = 0
    for i in range(M):
        osc = np.sum([((-1)**j)/float(j) for j in range(1,counts[i])])
        #print(osc)
        H += counts[i]/float(N) * (special.digamma(N) - special.digamma(counts[i]) + np.log(2) + osc)
    #print(H)
    H = H * np.log2(np.exp(1)) # transform from natural log to log base 2
    return H


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef unordered_map[string, double] getSystemSnapshots(Model model, long[::1] fixedNodesG = None, long[::1] fixedStates = None, \
              long nSnapshots = int(1e3), long repeats = 10, long burninSamples = int(1e3), \
              long distSamples = int(1e3), int threads = -1, int initStateIdx = -1):
    """
    Extract full system snapshots from MC for large network, for which the decimal encoding causes overflows

    """
    cdef:
        unordered_map[string, double] snapshots
        long i, rep, sample, numNodes = fixedNodesG.shape[0]
        long[::1] initialState, fixedNodesIdx = np.array([model.mapping[fixedNodesG[i]] for i in range(numNodes)], int)
        vector[long] allNodes = list(model._nodeids)
        string state
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads


    # initialize thread-safe models. nThread MC chains will be run in parallel.
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.resetAllToAgentState(initStateIdx, rep)

        initialState = tmp.states
        for i in range(numNodes):
            initialState[fixedNodesIdx[i]] = fixedStates[i]
        tmp.setStates(initialState)
        tmp.fixedNodes = fixedNodesIdx
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    i = repeats
    pbar = tqdm(total = i * nSnapshots)
    for rep in prange(i, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        (<Model>modelptr).simulateNSteps(burninSamples)

        for sample in range(nSnapshots):
            (<Model>modelptr).simulateNSteps(distSamples)
            with gil: state = (<Model> modelptr).encodeStateToString(allNodes)
            snapshots[state] += 1 # each index corresponds to one system state, the array contains the count of each state
            with gil: pbar.update(1)

    return snapshots


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getSystemStates(Model model, long[::1] nodesG, \
              unordered_map[long, unordered_map[long, vector[long]]] neighboursG, \
              long nSteps      = 1000, \
              int maxDist = 1,\
              long burninSamples  = 1000, \
              double threshold    = 0.05, \
              long repeats = 10, \
              long nBins = 100, \
              int threads = -1):
    """
    run MC for large network, encode system states into strings
    """
    cdef:
        long nNodes = model._nNodes
        long nNodesG = nodesG.shape[0]
        long[:,:,::1] states = np.zeros((repeats, nSteps, nNodes), int)
        double[:,::1] mags = np.zeros((repeats, nSteps))

        long node
        long[::1] nodesIdx = np.zeros(nNodesG, 'int')

        long[:,:,:,:,::1] avgSnapshotsPos = np.zeros((repeats, nNodesG, maxDist, model.agentStates.shape[0], nBins), int)
        long[:,:,:,:,::1] avgSnapshotsNeg = np.zeros((repeats, nNodesG, maxDist, model.agentStates.shape[0], nBins), int)
        long[:,:,:,:,::1] avgSnapshotsSwitch = np.zeros((repeats, nNodesG, maxDist, model.agentStates.shape[0], nBins), int)
        unordered_map[int, int] idxer
        vector[unordered_map[long, vector[long]]] neighboursIdx = vector[unordered_map[long, vector[long]]](nNodesG)

        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins + 1)[1:] # values represent upper bounds for bins

        long[::1] Z = np.zeros(3, int)

        long rep, step, n, nodeSpin, avg, d
        double m, mAbs
        int pos = 0
        string state
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads


    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx


    for n in range(nNodesG):
        node = nodesG[n]
        nodesIdx[n] = model.mapping[node]
        for d in range(maxDist):
            neighboursIdx[n][d+1] = [model.mapping[neighbour] for neighbour in neighboursG[node][d+1]]


    # initialize thread-safe models. nThread MC chains will be run in parallel.
    for rep in range(repeats):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.resetAllToAgentState(-1, rep)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    for rep in prange(repeats, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        (<Model>modelptr).simulateNSteps(burninSamples)

        states[rep] = (<Model>modelptr)._simulate(nSteps)

        for step in range(nSteps):
            m = mean(states[rep][step], nNodes, abs=0)
            mAbs = m if m > 0 else -m
            mags[rep][step] = m

            for n in range(nNodesG):
                nodeSpin = idxer[states[rep][step][nodesIdx[n]]]
                for d in range(maxDist):
                    avg = encodeStateToAvg(states[rep][step], neighboursIdx[n][d+1], bins)
                    if mAbs > threshold:
                        if m > 0:
                            avgSnapshotsPos[rep][n][d][nodeSpin][avg] +=1
                            Z[0] += 1
                        else:
                            avgSnapshotsNeg[rep][n][d][nodeSpin][avg] +=1
                            Z[1] += 1
                    else:
                        avgSnapshotsSwitch[rep][n][d][nodeSpin][avg] +=1
                        Z[2] += 1

    return avgSnapshotsPos.base, avgSnapshotsNeg.base, avgSnapshotsSwitch.base, Z.base, mags.base




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getSnapshotsPerDist(Model model, long nodeG, \
              unordered_map[long, vector[long]] allNeighboursG, \
              long nSnapshots = 100, long burninSamples = int(1e3), \
              int maxDist = 1, int threads = -1, int initStateIdx = -1):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take neighbourhood snapshots for the given node, ignore all others
    """
    cdef:
        vector[unordered_map[string, double]] snapshots = vector[unordered_map[string, double]](maxDist)
        unordered_map[int, int] idxer
        long nodeIdx, idx, d, rep, sample, n
        #double partSpin = 1/(<double> nSamples)
        double part = 1/(<double> nSnapshots)
        string state
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin
        int nThreads = mp.cpu_count() if threads == -1 else threads
        #long[:, ::1] spins = np.zeros((nSamples, model._nNodes), int)
        double[::1] spinProbs = np.zeros(model.agentStates.shape[0])

        unordered_map[long, vector[long]] allNeighboursIdx

    nodeIdx = model.mapping[nodeG] # map to internal index

    for d in range(maxDist):
        allNeighboursIdx[d+1] = [model.mapping[n] for n in allNeighboursG[d+1]]

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    # initialize thread-safe models. nThread MC chains will be run in parallel.
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.resetAllToAgentState(initStateIdx, rep)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    for sample in prange(nSnapshots, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        # when a thread has finished its first loop iteration,
        # it will continue running and sampling from the MC chain of the
        # same model, without resetting

        (<Model>modelptr).simulateNSteps(burninSamples)

        nodeSpin = (<Model>modelptr)._states[nodeIdx]
        spinProbs[idxer[nodeSpin]] += part

        #if sample < nSnapshots:
        for d in range(maxDist):
            with gil: state = (<Model> modelptr).encodeStateToString(allNeighboursIdx[d+1])
            snapshots[d][state] += part # each index corresponds to one system state, the array contains the probability of each state

        #spins[sample] = (<Model>modelptr)._states


    return snapshots, allNeighboursIdx, spinProbs.base

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getSnapshotsPerDistNodes(Model model, long[::1] nodesG, \
              long nSamples = int(1e3), \
              long burninSamples = int(1e3), int maxDist = 1, int threads = -1, \
              int initStateIdx = -1):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take neighbourhoods snapshots for the given nodes, ignore all others
    """
    cdef:
        long nNodes = nodesG.shape[0]
        long[::1] nodesIdx = np.zeros(nNodes, 'int')

        vector[vector[unordered_map[string, double]]] snapshots = vector[vector[unordered_map[string, double]]](nNodes)
        vector[unordered_map[long, vector[long]]] neighboursIdx = vector[unordered_map[long, vector[long]]](nNodes)
        vector[unordered_map[long, vector[long]]] neighboursG = vector[unordered_map[long, vector[long]]](nNodes)

        long d, i, b, sample, rep, n
        double part = 1 / (<double> nSamples)
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, avg
        int nThreads = mp.cpu_count() if threads == -1 else threads


    for n in range(nNodes):
        snapshots[n] = vector[unordered_map[string, double]](maxDist)
        nodesIdx[n] = model.mapping[nodesG[n]]
        neighboursG[n], neighboursIdx[n] = model.neighboursAtDist(nodesG[n], maxDist)


    # initialize thread-safe models. nThread MC chains will be run in parallel.
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.resetAllToAgentState(initStateIdx, rep)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    for sample in prange(nSamples, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        (<Model>modelptr).simulateNSteps(burninSamples)

        for n in range(nNodes):
            nodeSpin = (<Model> modelptr)._states[nodesIdx[n]]
            for d in range(maxDist):
                with gil: state = (<Model> modelptr).encodeStateToString(neighboursIdx[n][d+1])
                snapshots[n][d][state] += part # each index corresponds to one system state, the array contains the probability of each state


    return snapshots, neighboursG



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getSnapshotsPerDist2(Model model, long nodeG, \
              unordered_map[long, vector[long]] allNeighboursG, \
              long repeats=int(1e2), long nSamples = int(1e3), \
              long burninSamples = int(1e3), long distSamples=100, \
              int maxDist = 1, long nBins=10, int threads = -1, \
              int initStateIdx = -1, int getFullSnapshots = 0):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take snapshots of the specified node subset, ignore all others
    """
    cdef:
        #vector[unordered_map[int, unordered_map[string, double]]] snapshots = vector[unordered_map[int, unordered_map[string, double]]](maxDist)
        #vector[unordered_map[int, unordered_map[string, double]]] oldSnapshots = vector[unordered_map[int, unordered_map[string, double]]](maxDist)
        #vector[unordered_map[int, unordered_map[int, double]]] avgSnapshots = vector[unordered_map[int, unordered_map[int, double]]](maxDist)
        #vector[unordered_map[int, unordered_map[int, double]]] oldAvgSnapshots = vector[unordered_map[int, unordered_map[int, double]]](maxDist)

        #unordered_map[int, vector[unordered_map[int, unordered_map[int, double]]]] avgSnapshots

        long[:,:,:,::1] avgSnapshots = np.zeros((repeats, maxDist, model.agentStates.shape[0], nBins), int)
        long[:,:,::1] avgSystemSnapshots = np.zeros((repeats, model.agentStates.shape[0], nBins), int)
        unordered_map[int, int] idxer

        int idx
        long nodeIdx, d, i, b, sample, rep, start
        #long Z = repeats * nSamples
        #double part = 1/Z
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, avg, avgSystem
        int nThreads = mp.cpu_count() if threads == -1 else threads
        #np.ndarray KL = np.ones(maxDist)
        #double KL_d
        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins + 1)[1:] # values represent upper bounds for bins
        vector[long] allNodes = list(model.mapping.values())

        unordered_map[long, vector[long]] allNeighboursIdx

        long[:,:,::1] fullSnapshots


    if getFullSnapshots: fullSnapshots = np.zeros((repeats, nSamples, model._nNodes), int)

    #print(bins)

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    nodeIdx = model.mapping[nodeG]
    for d in range(maxDist):
        allNeighboursIdx[d+1] = [model.mapping[n] for n in allNeighboursG[d+1]]

    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    i = repeats # somehow it doesn't work when I directly use repeats

    pbar = tqdm(total = i * nSamples)
    for rep in prange(i, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        with gil:
            (<Model>modelptr).seed += rep # enforce different seeds
            #print(f'{tid} seed: {(<Model> models_[tid].ptr).seed}')
            (<Model>modelptr).resetAllToAgentState(initStateIdx, rep)
            #print(f'{tid} initial state: {(<Model> models_[tid].ptr)._states.base}')

        (<Model>modelptr).simulateNSteps(burninSamples)

        for sample in range(nSamples):
            (<Model>modelptr).simulateNSteps(distSamples)

            nodeSpin = idxer[(<Model> modelptr)._states[nodeIdx]]

            for d in range(maxDist):
                #with gil: state = (<Model> modelptr).encodeStateToString(allNeighbours_idx[d+1])
                #snapshots[d][nodeSpin][state] += 1
                avg = (<Model> modelptr).encodeStateToAvg(allNeighboursIdx[d+1], bins)
                #avgSnapshots[d][nodeSpin][avg] +=1
                avgSnapshots[rep][d][nodeSpin][avg] += 1
                #with gil: print(rep, avgSnapshots[d])

            avgSystem = (<Model> modelptr).encodeStateToAvg(allNodes, bins)
            avgSystemSnapshots[rep][nodeSpin][avgSystem] += 1

            if getFullSnapshots:
                #start = rep * nSamples
                # raw system states are stored, because for large systems encoding of snapshots does not work (overflow)
                fullSnapshots[rep][sample] = (<Model>modelptr)._states

            with gil: pbar.update(1)

    if getFullSnapshots:
        return avgSnapshots.base, avgSystemSnapshots.base, fullSnapshots.base
    else:
        return avgSnapshots.base, avgSystemSnapshots.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getJointSnapshotsPerDistNodes(Model model, long[::1] nodesG, \
              unordered_map[long, unordered_map[long, vector[long]]] neighboursG, \
              long repeats=10, long nSamples = int(1e3), \
              long burninSamples = int(1e3), long distSamples=100, \
              int maxDist = 1, long nBins=10, int threads = -1, \
              int initStateIdx = -1, int getFullSnapshots = 0):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take snapshots of the specified node subset, ignore all others
    """
    cdef:
        long node, nNodes = nodesG.shape[0]
        long[::1] nodesIdx = np.zeros(nNodes, 'int')

        #vector[vector[unordered_map[int, unordered_map[int, double]]]] avgSnapshots = vector[vector[unordered_map[int, unordered_map[int, double]]]](nNodes)
        long[:,:,:,:,::1] avgSnapshots = np.zeros((repeats, nNodes, maxDist, model.agentStates.shape[0], nBins), int)
        long[:,:,:,::1] avgSystemSnapshots = np.zeros((repeats, nNodes, model.agentStates.shape[0], nBins), int)
        unordered_map[int, int] idxer
        vector[unordered_map[long, vector[long]]] neighboursIdx = vector[unordered_map[long, vector[long]]](nNodes)
        #vector[unordered_map[long, vector[long]]] neighboursG = vector[unordered_map[long, vector[long]]](nNodes)
        vector[long] allNodes = list(model.mapping.values())

        long d, i, b, sample, rep, n
        #long Z = repeats * nSamples
        #double part = 1/Z
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, avg, avgSystem
        int nThreads = mp.cpu_count() if threads == -1 else threads
        #np.ndarray KL = np.ones(maxDist)
        #double KL_d
        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins + 1)[1:] # values represent upper bounds for bins

        long[:,:,::1] fullSnapshots

    if getFullSnapshots: fullSnapshots = np.zeros((repeats, nSamples, model._nNodes), int)

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx


    for n in range(nNodes):
        #avgSnapshots[n] = vector[unordered_map[int, unordered_map[int, double]]](maxDist)
        node = nodesG[n]
        nodesIdx[n] = model.mapping[node]
        for d in range(maxDist):
            neighboursIdx[n][d+1] = [model.mapping[neighbour] for neighbour in neighboursG[node][d+1]]


    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    i = repeats
    pbar = tqdm(total = i * nSamples)
    for rep in prange(i, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        with gil:
            (<Model>modelptr).seed += rep # enforce different seeds
            #print(f'{tid} seed: {(<Model> models_[tid].ptr).seed}')
            (<Model>modelptr).resetAllToAgentState(initStateIdx, rep)
            #print(f'{tid} initial state: {(<Model> models_[tid].ptr)._states.base}')

        (<Model>modelptr).simulateNSteps(burninSamples)

        for sample in range(nSamples):
            (<Model>modelptr).simulateNSteps(distSamples)
            avgSystem = (<Model> modelptr).encodeStateToAvg(allNodes, bins)

            for n in range(nNodes):
                nodeSpin = idxer[(<Model> modelptr)._states[nodesIdx[n]]]
                for d in range(maxDist):
                    avg = (<Model> modelptr).encodeStateToAvg(neighboursIdx[n][d+1], bins)
                    #avgSnapshots[n][d][nodeSpin][avg] +=1
                    avgSnapshots[rep][n][d][nodeSpin][avg] +=1

                avgSystemSnapshots[rep][n][nodeSpin][avgSystem] += 1

            if getFullSnapshots:
                # raw system states are stored, because for large systems encoding of snapshots does not work (overflow).
                # TODO: use string coding?
                fullSnapshots[rep][sample] = (<Model>modelptr)._states

            with gil: pbar.update(1)

    if getFullSnapshots:
        return avgSnapshots.base, avgSystemSnapshots.base, fullSnapshots.base
    else:
        return avgSnapshots.base, avgSystemSnapshots.base



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getJointSnapshotsPerDist(Model model, long nodeG, \
              unordered_map[long, vector[long]] allNeighboursG,
              long repeats = 10, \
              long nSamples = int(1e3), long burninSamples = int(1e3),
              long distSamples = 100, \
              int maxDist = 1, \
              int threads = -1, int initStateIdx = -1):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take snapshots of the specified node subset, ignore all others
    """
    cdef:
        vector[unordered_map[int, unordered_map[string, long]]] snapshots = vector[unordered_map[int, unordered_map[string, long]]](maxDist)


        long nodeIdx, d, i, sample, rep
        long Z       = repeats * nSamples
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, avg
        int nThreads = mp.cpu_count() if threads == -1 else threads

        unordered_map[long, vector[long]] allNeighboursIdx

    nodeIdx = model.mapping[nodeG]
    for d in range(maxDist):
        allNeighboursIdx[d+1] = [model.mapping[n] for n in allNeighboursG[d+1]]

    #allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)

    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    for rep in prange(repeats, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        with gil:
            (<Model>modelptr).seed += rep # enforce different seeds
            (<Model>modelptr).resetAllToAgentState(initStateIdx, rep)

        (<Model>modelptr).simulateNSteps(burninSamples)


        for sample in range(nSamples):

            (<Model>modelptr).simulateNSteps(distSamples)

            #print(f'snapshot: {(<Model> models_[tid].ptr)._states.base}')
            nodeSpin = (<Model> modelptr)._states[nodeIdx]
            #with gil: print((<Model> modelptr)._states.base)
            for d in range(maxDist):
                with gil: state = (<Model> modelptr).encodeStateToString(allNeighboursIdx[d+1])
                snapshots[d][nodeSpin][state] += 1

    return snapshots, Z


cpdef np.ndarray monteCarloFixedNeighbours(Model model, string snapshot, long nodeIdx, \
               vector[long] neighboursIdx, long nTrials, long burninSamples, \
               long nSamples = 10, long distSamples = 10):

      return _monteCarloFixedNeighbours(model, snapshot, nodeIdx, \
                     neighboursIdx, nTrials, burninSamples, \
                     nSamples, distSamples).base



@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[::1] _monteCarloFixedNeighbours(Model model, string snapshot, long nodeG, \
               vector[long] neighboursG, long nTrials, long burninSamples = int(1e3), \
               long nSamples = int(1e3), long distSamples = int(1e2), int initStateIdx = -1) nogil:


    #with gil: past = timer()

    cdef:
       double Z = <double> nSamples * nTrials
       double part = 1/Z
       long nodeIdx, idx, n, rep, sample, nNeighbours = neighboursG.size()
       long nodeState
       #unordered_map[long, double] probCond
       unordered_map[int, int] idxer #= {state : idx for idx, state in enumerate(model.agentStates)}
       double[::1] probCondArr #= np.zeros(idxer.size())
       #vector[double] probCondVec = vector[double](model.agentStates.shape[0], 0)
       long[::1] decodedStates
       long[::1] initialState
       #long[:,::1] sampledStates
       int i
       vector[long] neighboursIdx = vector[long] (nNeighbours)


    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    with gil: decodedStates = np.frombuffer(snapshot).astype(int)

    with gil: probCondArr = np.zeros(idxer.size())

    #with gil: sampledStates = np.zeros((nSamples, model._nNodes), int)

    #for idx in range(length):
    #    n = neighbours[idx]
    #    initialState[n] = decodedStates[idx]


    #with gil: print('start repetitions')
    #for rep in range(repeats):

        #with gil: model.seed += rep # enforce different seeds

    #model._loadStatesFromString(decodedStates, neighbours) # keeps all other node states as they are
    #model.simulateNSteps(burninSamples)

    #with gil: print(snapshot)

    for idx in range(nNeighbours):
        n = neighboursG[idx]
        neighboursIdx[idx] = model._mapping[n] # map from graph to model index

    nodeIdx = model._mapping[nodeG]

    #with gil: print(f'neighbours Idx: {neighboursIdx}')
    #with gil: print(f'mapping: {model.mapping}')

    with gil: model.fixedNodes = neighboursIdx

    for trial in range(nTrials):
        #with gil: print(trial, part, probCondArr.base)
        # set states without gil
        if initStateIdx < 0:
            #with gil: initialState = np.random.choice(model.agentStates, size = model._nNodes)
            with gil:
                i = np.mod(trial, model.agentStates.shape[0])
                initialState = np.ones(model._nNodes, int) * model.agentStates[i]
        else:
            with gil: initialState = np.ones(model._nNodes, int) * model.agentStates[initStateIdx]



        #with gil: print('internal recovery of state')
        for idx in range(nNeighbours):
            n = neighboursIdx[idx]
            initialState[n] = decodedStates[idx]
            #with gil: print(n, decodedStates[idx])

        #with gil: print(initialState.base)
        #with gil: print(model.mapping)


        model._setStates(initialState)
        #with gil: print({model.rmapping[idx]: initialState[idx] for idx in range(model._nNodes)})
        with gil: model.seed += 1
        #model._loadStatesFromString(decodedStates, neighbours) # keeps all other node states as they are
        model.simulateNSteps(burninSamples) # go to equilibrium

        for sample in range(nSamples):
            model.simulateNSteps(distSamples)
            nodeState = model._states[nodeIdx]
            #with gil: print([model._states[n] for n in neighboursIdx])

            #with gil: print(nodeState, model._states[neighbours[0]])
            #probCond[nodeState] += part
            #with gil: print(f'before: {probCondArr.base}, {part}')
            probCondArr[idxer[nodeState]] += part
            #with gil: print(f'after: {probCondArr.base}, {part}')
            #probCondVec[idxer[nodeState]] += part

            #sampledStates[sample] = model._states


    #print(f"time for sampling = {timer() - past: .2f} sec")
    with gil: model.releaseFixedNodes()

    #with gil: print(f'{timer() - past} sec')

    return probCondArr
    #return sampledStates


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef unordered_map[long, unordered_map[string, double]] _monteCarloFixedNeighboursStateDistr(Model model, string snapshot, long nodeIdx, long dist, \
               unordered_map[long, vector[long]] allNeighboursIdx, long nTrials, long burninSamples = int(1e3), \
               long nSamples = int(1e3), long distSamples = int(1e2), int initStateIdx = -1) nogil:


    cdef:
       long n, d, idx, rep, sample, length = allNeighboursIdx[dist].size()
       long minDist
       unordered_map[int, int] idxer
       long[::1] decodedStates
       long[::1] initialState
       int i
       string state
       #unordered_map[long, vector[long]] allNeighboursIdx

       unordered_map[long, unordered_map[string, double]] snapshots

    with gil: minDist = np.min(list(dict(allNeighboursIdx).keys()))
    #snapshots = vector[unordered_map[string, double]](dist-minDist)


    #for d in range(minDist, dist):
    #    allNeighboursIdx[d+1] = [model.mapping[n] for n in allNeighboursG[d+1]]

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    with gil: decodedStates = np.frombuffer(snapshot).astype(int)


    for trial in range(nTrials):
        # set states without gil
        if initStateIdx < 0:
            with gil:
                i = np.mod(trial, model.agentStates.shape[0])
                initialState = np.ones(model._nNodes, int) * model.agentStates[i]
        else:
            with gil: initialState = np.ones(model._nNodes, int) * model.agentStates[initStateIdx]

        for idx in range(length):
            n = allNeighboursIdx[dist][idx]
            initialState[n] = decodedStates[idx]


        model._setStates(initialState)
        with gil: model.seed += 1
        model.simulateNSteps(burninSamples) # go to equilibrium

        for sample in range(nSamples):
            model.simulateNSteps(distSamples)

            for d in range(minDist, dist):
                with gil: state = model.encodeStateToString(allNeighboursIdx[d])
                snapshots[d][state] += 1

    return snapshots



@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef long[:,::1] _monteCarloFixedNeighboursStates(Model model, string snapshot, long nodeIdx, \
               vector[long] neighboursIdx, long nTrials, long burninSamples = int(1e3), \
               long nSamples = int(1e3), long distSamples = int(1e2), int initStateIdx = -1, int saveDistrAtDist = -1) nogil:


    cdef:
       double Z = <double> nSamples * nTrials
       double part = 1/Z
       long idx, rep, sample, length = neighboursIdx.size()

       long[::1] decodedStates
       long[::1] initialState
       long[:,::1] sampledStates
       int i



    with gil: decodedStates = np.frombuffer(snapshot).astype(int)

    with gil: probCondArr = np.zeros(model.agentStates.shape[0])

    with gil: sampledStates = np.zeros((nSamples, model._nNodes), int)

    for trial in range(nTrials):

        if initStateIdx < 0:
            with gil:
                i = np.mod(trial, model.agentStates.shape[0])
                initialState = np.ones(model._nNodes, int) * model.agentStates[i]
        else:
            with gil: initialState = np.ones(model._nNodes, int) * model.agentStates[initStateIdx]

        for idx in range(length):
            n = neighboursIdx[idx]
            initialState[n] = decodedStates[idx]


        model._setStates(initialState)
        with gil: model.seed += 1

        model.simulateNSteps(burninSamples) # go to equilibrium

        for sample in range(nSamples):
            model.simulateNSteps(distSamples)
            sampledStates[sample] = model._states

    return sampledStates



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double entropyFromProbs(double[::1] probs) nogil:
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
cpdef tuple neighbourhoodMI(Model model, long nodeG, long dist, unordered_map[long, vector[long]] allNeighboursG, unordered_map[string, double] snapshots, \
              long nTrials, long burninSamples, long nSamples, long distSamples, int threads = -1, int initStateIdx = -1, \
              int uniformPDF = 0, str out = 'MI', int progbar = 0):

    assert out in ['MI', 'states', 'stateDistr']

    cdef:
        Model tmp
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads
        #vector[long] neighbours = model.neighboursAtDist(node, dist)[dist]
        long minDist, nodeIdx, n, sample, idx, nNeighbours = allNeighboursG[dist].size()
        vector[long] neighboursIdx = vector[long](nNeighbours)
        #long totalSnapshots = nSnapshots * nNeighbours
        #double part = 1 / (<double> totalSnapshots)
        #unordered_map[string, double] snapshots
        string state
        double[::1] pY, pX, allHXgiveny
        double HX, HXgiveny, HXgivenY = 0, MI = 0

    for idx in range(nNeighbours):
        n = allNeighboursG[dist][idx]
        #print(n)
        neighboursIdx[idx] = model.mapping[n] # map from graph to model index

    #print(f'neighbours Idx: {neighboursIdx}')
    #print(f'mapping: {model.mapping}')

    for tid in range(nThreads):
       tmp = copy.deepcopy(model)
       #tmp.fixedNodes = neighboursIdx
       models_.push_back(PyObjectHolder(<PyObject *> tmp))

    nodeIdx = model.mapping[nodeG]


    cdef:
        dict snapshotsDict = snapshots
        long numStates = len(snapshotsDict)
        vector[string] keys = list(snapshotsDict.keys())
        double[::1] probsStates = np.ones(numStates)/numStates if uniformPDF else np.array([snapshots[k] for k in keys])
        double[::1] weights = np.array([snapshots[k] for k in keys])
        double[:,::1] container = np.zeros((keys.size(), model.agentStates.shape[0]))

        long[:,:,::1] states = np.zeros((keys.size(), nSamples, model._nNodes), int)
        vector[unordered_map[long, unordered_map[string, double]]] stateDistr
        unordered_map[long, vector[long]] allNeighboursIdx

    allHXgiveny = np.zeros(numStates)


    if out == 'MI':

        # get conditional probabilities
        if progbar: pbar = tqdm(total = keys.size())
        for idx in prange(keys.size(), nogil = True, schedule = 'dynamic', num_threads = nThreads):
            tid = threadid()
            modelptr = models_[tid].ptr

            container[idx] = _monteCarloFixedNeighbours((<Model>modelptr), \
                          keys[idx], nodeG, allNeighboursG[dist], nTrials, burninSamples, nSamples, distSamples, initStateIdx)

            HXgiveny = entropyFromProbs(container[idx])
            HXgivenY -= probsStates[idx] * HXgiveny

            allHXgiveny[idx] = HXgiveny

            if progbar:
                with gil: pbar.update(1)# compute MI based on conditional probabilities

        pX = np.sum(np.multiply(weights.base, container.base.transpose()), axis=1)
        pX = pX / np.sum(pX)
        #print(pX.base)
        HX = entropyFromProbs(pX)
        #HX = entropyFromProbs(spinProbs)
        MI = HXgivenY + HX

        print(f'MI= {MI}, H(X|Y) = {HXgivenY}, H(X) = {HX}')

        return snapshotsDict, container.base, MI, HX, allHXgiveny.base, keys, weights.base


    elif out == 'states':

        pbar = tqdm(total = keys.size())
        for idx in prange(keys.size(), nogil = True, schedule = 'dynamic', num_threads = nThreads):
            tid = threadid()
            modelptr = models_[tid].ptr

            states[idx] = _monteCarloFixedNeighboursStates((<Model>modelptr), \
                        keys[idx], nodeIdx, neighboursIdx, nTrials, burninSamples, nSamples, distSamples, initStateIdx)

            with gil: pbar.update(1)

        return snapshotsDict, states.base

    elif out == 'stateDistr':

        stateDistr = vector[unordered_map[long, unordered_map[string, double]]] (keys.size())

        minDist = np.min(list(dict(allNeighboursG).keys()))
        print(f'min dist = {minDist}')
        for d in range(minDist, dist):
            allNeighboursIdx[d+1] = [model.mapping[n] for n in allNeighboursG[d+1]]

        pbar = tqdm(total = keys.size())
        for idx in prange(keys.size(), nogil = True, schedule = 'dynamic', num_threads = nThreads):
            tid = threadid()
            modelptr = models_[tid].ptr

            stateDistr[idx] = _monteCarloFixedNeighboursStateDistr((<Model>modelptr), \
                        keys[idx], nodeIdx, dist, allNeighboursIdx, nTrials, burninSamples, nSamples, distSamples, initStateIdx)

            with gil: pbar.update(1)

        return snapshotsDict, stateDistr


"""
cpdef double[::1] runNeighbourhoodMI(Model model, long node, long nSamples, long distSample, int maxDist, \
            vector[unordered_map[string, long]] snapshots, long nTopSnapshots, unordered_map[long, vector[long]] allNeighbours, \
            double[::1] centralNodeProbs, dict modelSettings, dict mixingTimeSettings, long nCondSamples):

    cdef:
        double[::1] MIperDist = np.zeros(maxDist)
        long d

    nodeSubset = [node]
    for d in range(maxDist):
        nodeSubset.extend(allNeighbours[d+1])
        modelSubgraph = Ising(model.graph.subgraph(nodeSubset), **modelSettings)

        mixingTime, corrTime = determineCorrTime(modelSubgraph, **mixingTimeSettings)

        selectedSnapshots = sorted(dict(snapshots[d]).items(), key=lambda x: x[1], reverse=True)[:nTopSnapshots]
        sum = np.sum([v for k,v in selectedSnapshots])
        selectedSnapshotDict = {k: v/sum for k,v in selectedSnapshots}

        _, _, MI = neighbourhoodMI(modelSubgraph, node, allNeighbours[d+1], selectedSnapshotDict, centralNodeProbs, \
                      nSamples=nCondSamples, distSamples=corrTime)
        MIperDist[d] = MI

    return MIperDist
"""
"""
cpdef tuple test2(Model model, long node, int dist, long nSnapshots, long nStepsToSnapshot, \
              long nSamples, long distSamples, long nRunsSampling):
    cdef:
        #vector[long] neighbours = list(model.mapping.values())[1:]
        vector[long] neighbours = model.neighboursAtDist(node, dist)[dist]
        long idx, nNeighbours = neighbours.size()
        #unordered_map[string, double] snapshots
        dict snapshots
        bytes state
        double[::1] probConds, pXgivenY, pX = np.zeros(model.agentStates.shape[0])
        #dict allProbConds = {}
        unordered_map[string, double[::1]] allProbConds
        dict allProbCondsDict
        #double[:,::1] allProbConds
        double pY, HXgiveny, HXgivenY = 0, MI


    print(f'neighbours at dist={dist}: {neighbours}')
    snapshots = getSnapShotsLargeNetwork(model, nSnapshots*nNeighbours, neighbours, nStepsToSnapshot)

    #allProbConds = np.zeros((len(snapshots), model.agentStates.shape[0]))

    for state in tqdm(snapshots):
        pXgivenY = monteCarloFixedNeighbours(model, \
                      state, node, neighbours, nSamples, distSamples, nRunsSampling)
        # compute MI based on conditional probabilities
        pY = snapshots[state]
        HXgiveny = entropyFromProbs(pXgivenY)
        HXgivenY -= pY * HXgiveny
        pX += pY * pXgivenY.base

    MI = HXgivenY + entropyFromProbs(pX)

    #allProbCondsDict = allProbConds

    print(f'MI= {MI}')

    return snapshots, MI
"""


"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef long[::1] simulate(Model model, int nSamples = int(1e2)) nogil:
    cdef:
        long[:, ::1] r = model.sampleNodes(nSamples)
        int step

    for step in range(nSamples):
        model._updateState(r[step])

    return model._states
"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef long[:,::1] equilibriumSampling(Model model, long repeats, \
            long burninSamples, long nSamples, long distSamples, \
            int threads = -1, int initStateIdx = -1):
    cdef:
        PyObject *modelptr
        vector[PyObjectHolder] models_
        Model tmp
        long start, sample, idx, step, rep, s
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads
        long[:,::1] snapshots = np.zeros((repeats * nSamples, model._nNodes), int) #np.intc)


    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    pbar = tqdm(total = repeats) # init  progbar
    for rep in prange(repeats, nogil = True, schedule = 'dynamic', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        with gil:
            (<Model>modelptr).resetAllToAgentState(initStateIdx, rep)
            (<Model>modelptr).seed += rep # enforce different seeds
        (<Model>modelptr).simulateNSteps(burninSamples)
        start = rep * nSamples

        for sample in range(nSamples):
            # raw system states are stored, because for large systems encoding of snapshots does not work (overflow)
            snapshots[start + sample] = (<Model>modelptr).simulateNSteps(distSamples)

        with gil:
            pbar.update(1)

    return snapshots



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef long[:,::1] equilibriumSamplingMagThreshold(Model model, long repeats, \
                      long burninSamples, long nSamples, long distSamples, \
                      int switch=1, double threshold=0.05, int threads = -1, \
                      int initStateIdx = -1):
    cdef:
        PyObject *modelptr
        vector[PyObjectHolder] models_
        Model tmp
        long start, sample = 0, idx, step, rep, s, nNodes = model._nNodes
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads
        double mu
        long[:,::1] snapshots = np.zeros((repeats * nSamples, nNodes), int) #np.intc)

    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    pbar = tqdm(total = repeats) # init  progbar
    for rep in prange(repeats, nogil = True, schedule = 'dynamic', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        with gil:
            (<Model>modelptr).resetAllToAgentState(initStateIdx, rep)
            (<Model>modelptr).seed += rep
        (<Model>modelptr).simulateNSteps(burninSamples)
        start = rep * nSamples

        for sample in range(nSamples):
            mu = mean((<Model>modelptr).simulateNSteps(distSamples), nNodes, abs=1)
            while ((switch and mu >= threshold) and (not switch and mu < threshold)):
                # continue simulating until system state reached where intended avg mag level is reached
                mu = mean((<Model>modelptr).simulateNSteps(1), nNodes, abs=1)
            snapshots[start + sample] = (<Model>modelptr)._states

        with gil:
            pbar.update(1)

    return snapshots

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef double[::1] binaryEntropies(long[:,::1] snapshots):
    cdef:
        #np.ndarray H = np.zeros(nNodes)
        #long[:, ::1] s = np.array([decodeState(i, nNodes) for i in tqdm(snapshots)])
        #np.ndarray H = np.sum(s.base, axis=0)
        np.ndarray tmp, H = np.sum(snapshots.base, axis=0, dtype=float)
        long length = snapshots.shape[0]
        double[::1] cview_H

    #print(snapshots.base.shape)
    #print(H)
    H = (length - np.abs(H))/2. + np.abs(H)
    H = H/length

    tmp = 1-H
    H = - H * np.log2(H) - tmp * np.log2(np.where(tmp==0, 1, tmp)) # compute entropy for each node (use 0*log(0) = 0)

    cview_H = H

    #print("bin entropies: {}".format(H))

    return cview_H

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef double pairwiseMI(long[:,::1] snapshots, double[::1] binEntropies, long nodeIdx1, long nodeIdx2) nogil:
    cdef:
        long idx, nSamples = snapshots.shape[0]
        vector[long] states
        vector[long] jointDistr = vector[long](nSamples, 0)
        double mi, jointEntropy


    for idx in range(nSamples):
        #states = decodeState(snapshots[idx], nNodes)
        jointDistr[idx] = snapshots[idx][nodeIdx1] + snapshots[idx][nodeIdx2]*2 # -3,-1,1,3 represent the 4 possible combinations of spins

    with gil:
        #print(binEntropies[node1], binEntropies[node2], entropy(jointDistr))
        jointEntropy = entropy(jointDistr)

    # mutual information
    mi = binEntropies[nodeIdx1] + binEntropies[nodeIdx2] - jointEntropy

    return mi


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef double spinCorrelation(long[:,::1] snapshots, long nodeIdx1, long nodeIdx2) nogil:
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



cpdef double entropy(vector[long] samples):
    cdef:
        long[::1] counts = np.unique(samples, return_counts=True)[1]
        double entropy, n = <double> samples.size()

    entropy = -np.sum([c/n * np.log2(c/n) for c in counts])
    return entropy


cpdef tuple computeMI_jointPDF(np.ndarray snapshots, long Z):
    cdef:
        np.ndarray P_XY, P_X, P_Y
        double MI, H_X

    P_XY = snapshots.flatten()/Z
    P_X = np.sum(snapshots, axis=1)/Z # sum over all bins
    P_Y = np.sum(snapshots, axis=0)/Z # sum over all spin states
    H_X = stats.entropy(P_X, base=2)
    MI = stats.entropy(P_X, base=2) + stats.entropy(P_Y, base=2) - stats.entropy(P_XY, base=2)
    return MI, H_X

cpdef tuple computeMI_jointPDF_exact(unordered_map[int, unordered_map[string, long]] snapshots, long Z):
    cdef:
        np.ndarray P_XY, P_X, P_Y, jointPDF, states
        double MI, H_X, p
        dict d

    states = np.unique([s for d in dict(snapshots).values() for s in d.keys()])
    jointPDF = np.array([[d[s] if s in d else 0 for s in states] for d in dict(snapshots).values()])
    #print(jointPDF)
    #jointPDF = np.array([[s for s in states] for d in snapshots[d].values()]) #.reshape((len(snapshots), -1))
    #P_XY = np.array([p for d in dict(snapshots).values() for p in dict(d).values()])/Z
    P_XY = jointPDF.flatten()/Z
    #P_X = np.array([np.sum([p for p in dict(d).values()]) for d in dict(snapshots).values()]) # sum over all neighbourhood states
    P_X = np.sum(jointPDF, axis=1)/Z
    P_Y = np.sum(jointPDF, axis=0)/Z # sum over all spin states
    H_X = stats.entropy(P_X, base=2)

    MI = stats.entropy(P_X, base=2) + stats.entropy(P_Y, base=2) - stats.entropy(P_XY, base=2)
    if MI == 0: MI = np.nan
    return MI, H_X, jointPDF, states


cpdef tuple processJointSnapshotsNodes(np.ndarray avgSnapshots, long Z, np.ndarray nodesG, long maxDist, np.ndarray avgSystemSnapshots=None):

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

cpdef tuple processJointSnapshots(np.ndarray avgSnapshots, np.ndarray avgSystemSnapshots, long Z, long node, long maxDist):

    cdef:
        long n, d
        np.ndarray MI_avg = np.zeros(maxDist)
        double MI_system, HX

    avgSnapshots = np.sum(avgSnapshots, axis=0)
    avgSystemSnapshots = np.sum(avgSystemSnapshots, axis=0)

    for d in range(maxDist):
        MI_avg[d] = computeMI_jointPDF(avgSnapshots[d], Z)[0]
    MI_system, HX = computeMI_jointPDF(avgSystemSnapshots, Z)

    return MI_avg, MI_system, HX




@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef np.ndarray magTimeSeries(Model model, long burninSamples, \
                                long nSamples, int abs=0):

    return _magTimeSeries(model, burninSamples, nSamples, abs).base

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef double[::1] _magTimeSeries(Model model, long burninSamples, \
                                long nSamples, int abs=0):

    cdef:
        double[::1] mags = np.zeros(nSamples)
        long sample


    #string = encodeStateToString(model.states.base)

    #print(string)
    #print(model.states)

    model.simulateNSteps(burninSamples)

    for sample in range(nSamples):
        mags[sample] = mean(model.simulateNSteps(1), model._nNodes, abs)

    return mags

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple runMI(Model model, np.ndarray nodesG, long[:,::1] snapshots, \
                int distMax=1, int threads = -1, int initStateIdx = -1, long centralNodeIdx = -1):

    cdef:
        #long[::1] cv_nodes = nodes
        #long[:,::1] snapshots
        double[::1] entropies
        long i, n1, n2, d, nNodes = nodesG.shape[0]
        long[::1] nodesIdx = np.array([model.mapping[n] for n in nodesG])
        double[:,::1] MI = np.zeros((nNodes, nNodes))
        double[:,::1] corr = np.zeros((nNodes, nNodes))
        int nThreads = mp.cpu_count() if threads == -1 else threads

    # run multiple MC chains in parallel and sample snapshots
    #if magThreshold == 0:
    #    snapshots = equilibriumSampling(model, repeats, burninSamples, nSamples, distSamples, threads=threads, initStateIdx=initStateIdx)
    #elif magThreshold > 0:
    #    # only sample snapshots with abs avg mag larger than magThreshold
    #    snapshots = equilibriumSamplingMagThreshold(model, repeats, burninSamples, nSamples, distSamples, switch=0, threshold=magThreshold, threads=threads, initStateIdx=initStateIdx)
    #else:
    #  # only sample snapshots with abs avg mag smaller than magThreshold
    #    snapshots = equilibriumSamplingMagThreshold(model, repeats, burninSamples, nSamples, distSamples, switch=1, threshold=np.abs(magThreshold), threads=threads, initStateIdx=initStateIdx)

    #print(snapshots.base.shape)
    #entropies =np.array([binaryEntropies(snapshots.base[i,:,:]) for i in range(snapshots.shape[0])])
    entropies = binaryEntropies(snapshots)

    for n1 in prange(nNodes, nogil = True, schedule = 'dynamic', num_threads = nThreads):
        with gil: print(f'processing node {n1}')
        if centralNodeIdx < 0:
            for n2 in range(n1, nNodes):
                MI[n1][n2] = pairwiseMI(snapshots, entropies, nodesIdx[n1], nodesIdx[n2])
                MI[n2][n1] = MI[n1][n2] # symmetric
                corr[n1][n2] = spinCorrelation(snapshots, nodesIdx[n1], nodesIdx[n2])
                corr[n2][n1] = corr[n1][n2] # symmetric
        else:
            MI[n1][centralNodeIdx] = pairwiseMI(snapshots, entropies, nodesIdx[n1], nodesIdx[centralNodeIdx])
            MI[centralNodeIdx][n1] = MI[n1][centralNodeIdx] # symmetric
            corr[n1][centralNodeIdx] = spinCorrelation(snapshots, nodesIdx[n1], nodesIdx[centralNodeIdx])
            corr[centralNodeIdx][n1] = corr[n1][centralNodeIdx] # symmetric

    return MI.base, corr.base

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple runMIoneNode(Model model, long nodeG, long[:,::1] snapshots, \
                int distMax=1, int threads = -1, int initStateIdx = -1):

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
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef double mean(long[::1] arr, long len, int abs=0) nogil:
    cdef:
        double mu = 0
        long i
    for i in range(len):
        mu = mu + arr[i]
    mu = mu / len

    if abs and mu < 0:
        return -mu
    else:
        return mu


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef np.ndarray magnetizationParallel(Model model,\
                          np.ndarray temps  = np.logspace(-3, 2, 20),\
                      long n             = int(1e3),\
                      long burninSamples = 100, int threads = -1):
    """
    Computes the magnetization as a function of temperatures
    Input:
          :model: the model to use for simulations
          :temps: a range of temperatures
          :n:     number of samples to simulate for
          :burninSamples: number of samples to throw away before sampling
    Returns:
          :temps: the temperature range as input
          :mag:  the magnetization for t in temps
          :sus:  the magnetic susceptibility
    """

    # if workload is small, use serial implementation
    #if n < 1e4:
    #    return model.matchMagnetization(temps, n, burninSamples)

    # otherwise parallel is faster
    cdef:
        PyObject *modelptr
        vector[PyObjectHolder] models_
        Model tmp
        long idx, nNodes = model._nNodes, t, step, start
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads
        long nTemps = temps.shape[0]
        long totalSteps = n + burninSamples
        double m
        vector[double] mag_sum
        #double[::1] betas = np.array([1.0 / float(t) if t != 0 else np.inf for t in temps])
        double[::1] temps_cview = temps
        double[:,::1] results = np.zeros((4, nTemps))



    # threadsafe model access
    for tid in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    pbar = tqdm(total = nTemps)
    for t in prange(nTemps, nogil = True, \
                         schedule = 'static', num_threads = nThreads): # simulate with different temps in parallel
        tid = threadid()
        modelptr = models_[tid].ptr
        with gil:
            (<Model>modelptr).reset()
            (<Model>modelptr).seed += t
            (<Model>modelptr).t = temps[t]
            (<Model>modelptr).resetAllToAgentState(1, 0)

        # simulate until equilibrium reached
        (<Model>modelptr).simulateNSteps(burninSamples)

        mag_sum = simulateGetMeanMag((<Model>modelptr), n)

        m = mag_sum[0] / n
        results[0][t] = m #if m > 0 else -m
        results[1][t] = ((mag_sum[1] / n) - (m * m)) / temps_cview[t] # susceptibility
        results[2][t] = 1 - (mag_sum[3]/n) / (3 * (mag_sum[1]/n)**2) # Binder's cumulant
        results[3][t] = mag_sum[4] / n

        with gil:
            pbar.update(1)

    return results.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
#@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef np.ndarray autocorrelation(np.ndarray timeSeries):
    cdef:
        int padding = 1
        np.ndarray autocorr
        double corrTime

    # get next power of two for padding
    while padding < timeSeries.size:
        padding = padding << 1

    # use discrete fourier transform for faster computation of autocorrelation
    autocorr = np.fft.fft(timeSeries - np.mean(timeSeries), n=2*padding)
    autocorr = np.fft.ifft(autocorr * np.conjugate(autocorr)).real
    if autocorr[0] == 0:
        print(timeSeries)
        autocorr = np.ones(autocorr.size)
    else:
        autocorr /= autocorr[0] # normalize
    autocorr = autocorr[:timeSeries.size]

    return autocorr


@cython.boundscheck(False)
#@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple determineMixingTime(Model model,\
                      long burninSteps = 100,\
                      long nStepsRegress = int(1e3),\
                      double threshold = 0.05,\
                      long nStepsCorr = int(1e4), \
                      int checkMixing = 1,
                      long nodeG = -1):

    cdef:
        long s, nNodes = model._nNodes
        double[::1] mags = np.zeros(burninSteps)
        np.ndarray allMags  # for regression
        long lag, h, counter, mixingTime, sample # tmp var and counter
        double beta        # slope value
        np.ndarray magSeries = np.zeros(nStepsCorr), autocorr, x = np.arange(nStepsRegress)# for regression
        double slope, intercept, r_value, p_value, std_err, corrTime

    counter = 0
    allMags = np.array(mean(model.states, nNodes, abs=1))
    #initial_conf = model.states

    # simulate until mag has stabilized
    # remember mixing time needed

    if checkMixing:
        while True: #counter < maxStepsBurnin:
            #for sample in range(stepSizeBurnin):
            #    mags[sample] = mean(model.simulateNSteps(1), model._nNodes, abs=1)

            mags = _magTimeSeries(model, 0, burninSteps, abs=1)

            allMags = np.hstack((allMags, mags.base))
            if counter >= nStepsRegress :
                # do linear regression
                slope, intercept, r_value, p_value, std_err = linregress(x, allMags[-nStepsRegress:])
                if 1 - p_value < threshold: # p-value of test for slope=0
                    #print(slope, intercept, p_value, std_err)
                    break
            counter = counter + burninSteps
        #else:
        mixingTime = counter - nStepsRegress
        #print('Number of bunin samples needed {0}\n\n'.format(mixingTime))
        #print(f'absolute mean magnetization last sample {y[-1]}')

    else: # run given number of burnin steps
        mixingTime = burninSteps
        model.simulateNSteps(mixingTime)
        intercept = simulateGetMeanMag(model, 10)[0]/10.0 # get mean magnetization of 10 steps in equilibrium

    # measure correlation time (autocorrelation for varying lags)
    if nodeG in model.mapping:
        nodeIdx = model.mapping[nodeG]
        for sample in range(nStepsCorr):
            model.simulateNSteps(1)
            magSeries[sample] = model._states[nodeIdx]
    else:
        magSeries = magTimeSeries(model, 0, nStepsCorr)
    autocorr = autocorrelation(magSeries)

    return allMags, mixingTime, intercept, autocorr


cpdef tuple determineCorrTime(Model model, \
              int nInitialConfigs = 10, \
              long burninSteps = 10, \
              long nStepsRegress = int(1e3), \
              double thresholdReg = 0.05, \
              long nStepsCorr = int(1e3), \
              double thresholdCorr = 0.05, \
              int checkMixing = 1,
              long nodeG = -1):
    cdef:
        #vector[PyObjectHolder] models_
        #Model tmp
        #int tid, nThreads = mp.cpu_count()
        long nNodes = model._nNodes
        long mixingTime
        double corrTime, intercept, meanMag = 0
        long idx
        #double[::1] temps_cview = temps
        np.ndarray tmp, mags, autocorr, initialConfigs = np.linspace(0.5, 1, nInitialConfigs)
        double prob, t
        long mixingTimeMax = 0
        double corrTimeMax = 0 # double because it might be infinity
        dict allMagSeries = {}


    # threadsafe model access
    #for tid in range(nThreads):
    #    tmp = copy.deepcopy(model)
    #    models_.push_back(PyObjectHolder(<PyObject *> tmp))

    #pbar = tqdm(total = nTemps) # init  progbar
    for prob in tqdm(initialConfigs):
        model.states = np.random.choice([-1,1], size = nNodes, p=[prob, 1-prob])
        mags, mixingTime, intercept, autocorr = determineMixingTime(model,\
                              burninSteps,\
                              nStepsRegress,\
                              thresholdReg,\
                              nStepsCorr, \
                              checkMixing,
                              nodeG)

        allMagSeries[prob] = mags

        meanMag += intercept
        if mixingTime > mixingTimeMax: mixingTimeMax = mixingTime
        tmp = np.where(np.abs(autocorr) < thresholdCorr)[0]
        corrTime = tmp[0] if tmp.size > 0 else np.inf
        if corrTime > corrTimeMax: corrTimeMax = corrTime

    meanMag /= nInitialConfigs

    return mixingTimeMax, meanMag, corrTimeMax, allMagSeries


cpdef long mixing2(Model model, int nInitialConfigs, long nSteps, double threshold, int threads = -1):

    cdef:
        double[::1] initialConfigs = np.linspace(0.5, 1, nInitialConfigs)
        vector[PyObjectHolder] models_
        Model tmp
        long simTime = 0, idx, nNodes = model._nNodes
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads
        double[:,::1] estimators = np.zeros((nInitialConfigs, 2)) # mean and std of last nSteps
        double var

    for idx in range(nInitialConfigs):
       tmp = copy.deepcopy(model)
       tmp.states = np.random.choice([-1,1], size = nNodes, p=[initialConfigs[idx], 1-initialConfigs[idx]])
       tmp.seed += idx
       models_.push_back(PyObjectHolder(<PyObject *> tmp))

    while True:
        for idx in prange(nInitialConfigs, nogil = True, schedule = 'dynamic', num_threads = nThreads):
            #tid = threadid()
            estimators[idx] = simulateGetStdMag(<Model>models_[idx].ptr, nSteps)

        simTime += nSteps
        var = np.std(estimators.base[:,0]) # variance among parallel MC chains
        std = 0
        avg = np.mean(estimators.base[:,0])
        #print(f'mean = {avg}')
        #for step in range(nInitialConfigs):
        #    std = std + (estimators[step, 0] - avg)*(estimators[step, 0] - avg)
        print(var)
        #print(f'std within chains = {estimators.base[:,1]}')
        #print(f'mean within chains = {estimators.base[:,0]}')
        if var < threshold: # check if mag level has stabilized within run and among parallel runs
            break

    return simTime



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef vector[double] simulateGetMeanMag(Model model, long nSamples = int(1e2)) nogil:
    cdef:
        long[:, ::1] r = model.sampleNodes(nSamples)
        long step
        double m, m_abs, sum, sum_abs, sum_2, sum_3, sum_4
        vector[double] out = vector[double](5,0)

    sum = 0
    sum_2 = 0
    sum_3 = 0
    sum_4 = 0
    # collect magnetizations
    for step in range(nSamples):
        model._updateState(r[step])

        m = mean(model._states, model._nNodes)
        m_abs = mean(model._states, model._nNodes, abs=1)
        sum = sum + m
        sum_abs = sum_abs + m_abs
        sum_2 = sum_2 + (m*m)
        sum_3 = sum_3 + (m**3)
        sum_4 = sum_4 + (m**4)

    out[0] = sum
    out[1] = sum_2
    out[2] = sum_3
    out[3] = sum_4
    out[4] = sum_abs

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef long[:,::1] simulateGetStates(Model model, long burninSteps = int(1e2), long nSamples = int(1e2)) nogil:
    cdef:
        long[:, ::1] r = model.sampleNodes(nSamples)
        long step
        long[:,::1] states

    with gil: states = np.zeros((nSamples, model._nNodes), dtype=int)

    model.simulateNSteps(burninSteps)

    for step in range(nSamples):
        model._updateState(r[step])
        states[step] = model._states

    return states


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef double[::1] simulateGetStdMag(Model model, long nSamples = int(1e2), int abs=1) nogil:
    cdef:
        long[:, ::1] r = model.sampleNodes(nSamples)
        long step
        double avg = 0, m, std = 0
        vector[double] mags = vector[double] (nSamples, 0)
        double[::1] out

    with gil: out = np.zeros(2)

    # collect magnetizations
    for step in range(nSamples):
        m = mean(model._updateState(r[step]), model._nNodes, abs)
        mags[step] = m
        avg = avg + m

    avg = avg/(<double> nSamples)
    for step in range(nSamples):
        std = std + (mags[step] - avg)*(mags[step] - avg)

    std = sqrt(std/<double> nSamples)
    with gil: std = np.std(mags)
    out[0] = avg
    out[1] = std

    return out



cpdef np.ndarray f(Worker x):
    # print('id', id(x))
    return x.parallWrap()


@cython.auto_pickle(True)
cdef class Worker:
    """
    This class was used to wrap the c classes for use with the multiprocessing toolbox.
    However, the performance decreased a lot. I think this may be a combination of the nogil
    sections. Regardless the 'single' threaded stuff above is plenty fast for me.
    Future me should look into dealing with the gil  and wrapping everything in  c arrays
    """
    cdef int deltas
    cdef int idx
    cdef int repeats
    cdef np.ndarray startState
    cdef Model model
    # cdef dict __dict__
    def __init__(self, *args, **kwargs):
        # for k, v in kwargs.items():
        #     setattr(self, k, v)

        self.deltas     = kwargs['deltas']
        self.model      = kwargs['model']
        self.repeats    = kwargs['repeats']
        self.startState = kwargs['startState']
        self.idx        = kwargs['idx']

    cdef np.ndarray parallWrap(self):
        cdef long[::1] startState = self.startState
        # start unpacking
        cdef int deltas           = self.deltas
        cdef int repeats          = self.repeats
        # cdef long[::1] startState = startState
        cdef Model model          = self.model
        # pre-declaration
        cdef double[::1] out = np.zeros((deltas + 1) * model._nNodes * model._nStates)
        cdef double Z              = <double> repeats
        cdef double[:] copyNudge   = model._nudges.copy()
        cdef bint reset            = True
        # loop stuff
        cdef long[:, ::1] r
        cdef int k, delta, node, statei, counter, half = deltas // 2
        # pbar = tqdm(total = repeats)
        for k in range(repeats):
            for node in range(model._nNodes):
                model._states[node] = startState[node]
                model._nudges[node] = copyNudge[node]
            # reset simulation
            reset   = True
            counter = 0
            r       = model.sampleNodes(repeats * (deltas + 1))
            for delta in range(deltas + 1):
                # bin data
                for node in range(model._nNodes):
                    for statei in range(model._nStates):
                        if model._states[node] == model.agentStates[statei]:
                            out[counter] += 1 / Z
                        counter += 1
                # update
                model._updateState(r[counter])

                # turn-off
                if reset:
                    if model._nudgeType == 'pulse' or \
                    model._nudgeType    == 'constant' and delta >= half:
                        model._nudges[:] = 0
                        reset            = False
            # pbar.update(1)
        # pbar.close()
        return out.base.reshape((deltas + 1, model._nNodes, model._nStates))

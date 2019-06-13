# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Fiona Lippert'

from Models.models cimport Model
from Models.fastIsing cimport Ising
from Toolbox.infoTheory import *
from Toolbox.infoTheory cimport *

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef unordered_map[string, double] getSystemSnapshotsFixedNodes(Model model, long[::1] nodes, long[::1] fixedNodesG, long[::1] fixedStates, \
              long nSnapshots = int(1e3), long repeats = 10, long burninSamples = int(1e3), \
              long distSamples = int(1e3), int threads = -1, int initStateIdx = -1):
    """
    Freeze given nodes to the given states, simulate the system in equilibrium, and extract snapshots of the remaining nodes
    """
    cdef:
        unordered_map[string, double] snapshots
        long i, rep, sample, numNodes = fixedNodesG.shape[0]
        long[::1] initialState, fixedNodesIdx = np.array([model.mapping[fixedNodesG[i]] for i in range(numNodes)], int)
        vector[long] allNodesIdx = [model.mapping[n] for n in nodes]
        string state
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads


    initialState = np.ones(model._nNodes, int)#tmp.states
    for i in range(numNodes):
        initialState[fixedNodesIdx[i]] = fixedStates[i]


    # initialize thread-safe models. nThread MC chains will be run in parallel.
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep

        tmp.fixedNodes = fixedNodesIdx
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    i = repeats
    pbar = tqdm(total = i * nSnapshots)
    for rep in prange(i, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        with gil: (<Model>modelptr).setStates(initialState)
        #with gil: print(f'init state {initialState.base}')
        #with gil: print(f'fixed states {fixedStates.base}: {[s for s in (<Model>modelptr)._states]}')

        (<Model>modelptr).simulateNSteps(burninSamples)

        for sample in range(nSnapshots):
            (<Model>modelptr).simulateNSteps(distSamples)
            #with gil: print([(<Model>modelptr)._states[i] for i in fixedNodesIdx])
            with gil:
                state = (<Model> modelptr).encodeStateToString(allNodesIdx)
                snapshots[state] += 1 # each index corresponds to one system state, the array contains the count of each state
            with gil: pbar.update(1)

    return snapshots


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef vector[unordered_map[string, unordered_map[string, double]]] getSystemSnapshotsSets(Model model, vector[vector[long]] systemNodesG, vector[vector[long]] condNodesG, \
              long nSnapshots = int(1e3), long repeats = 10, long burninSamples = int(1e3), \
              long distSamples = int(1e3), int threads = -1, int initStateIdx = -1):
    """
    simulate the system in equilibrium, for all sets of nodes in 'systemNodesG'
    extract the distribution of the joint states of these nodes, conditioned on
    the states of the respective set of nodes in 'condNodesG'
    """
    cdef:
        long n, i, rep, sample, set, numSets = condNodesG.size()
        vector[long] arr
        vector[unordered_map[string, unordered_map[string, double]]]  snapshots = vector[unordered_map[string, unordered_map[string, double]]](numSets)
        vector[vector[long]] condNodesIdx = [[model.mapping[n] for n in arr] for arr in condNodesG]
        vector[vector[long]] systemNodesIdx = [[model.mapping[n] for n in arr] for arr in systemNodesG]
        string systemState, condState
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads


    # initialize thread-safe models. nThread MC chains will be run in parallel.
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.resetAllToAgentState(initStateIdx, rep)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    i = repeats
    pbar = tqdm(total = i * nSnapshots)
    for rep in prange(i, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        (<Model>modelptr).simulateNSteps(burninSamples)

        for sample in range(nSnapshots):
            (<Model>modelptr).simulateNSteps(distSamples)
            #with gil: print([(<Model>modelptr)._states[i] for i in fixedNodesIdx])
            for set in range(numSets):
                with gil:
                    condState = (<Model> modelptr).encodeStateToString(condNodesIdx[set])
                    systemState = (<Model> modelptr).encodeStateToString(systemNodesIdx[set])
                    snapshots[set][condState][systemState] += 1 # each index corresponds to one system state, the array contains the count of each state
            with gil: pbar.update(1)

    return snapshots



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef unordered_map[string, unordered_map[string, double]] getSystemSnapshots(Model model, long[::1] systemNodesG, long[::1] condNodesG, \
              long nSnapshots = int(1e3), long repeats = 10, long burninSamples = int(1e3), \
              long distSamples = int(1e3), int threads = -1, int initStateIdx = -1):
    """
    simulate the system in equilibrium, extract the distribution of the joint
    states of all nodes in 'systemNodesG', conditioned on the states of the
    nodes in 'condNodesG'
    """
    cdef:
        unordered_map[string, unordered_map[string, double]]  snapshots
        long i, rep, sample, numNodes = condNodesG.shape[0]
        long[::1] initialState, condNodesIdx = np.array([model.mapping[condNodesG[i]] for i in range(numNodes)], int)
        vector[long] systemNodesIdx = [model.mapping[n] for n in systemNodesG]
        string systemState, condState
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads


    # initialize thread-safe models. nThread MC chains will be run in parallel.
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.resetAllToAgentState(initStateIdx, rep)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    i = repeats
    pbar = tqdm(total = i * nSnapshots)
    for rep in prange(i, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        (<Model>modelptr).simulateNSteps(burninSamples)

        for sample in range(nSnapshots):
            (<Model>modelptr).simulateNSteps(distSamples)
            #with gil: print([(<Model>modelptr)._states[i] for i in fixedNodesIdx])
            with gil:
                condState = (<Model> modelptr).encodeStateToString(list(condNodesIdx))
                systemState = (<Model> modelptr).encodeStateToString(systemNodesIdx)
                snapshots[condState][systemState] += 1 # each index corresponds to one system state, the array contains the count of each state
            with gil: pbar.update(1)

    return snapshots


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getAvgSnapshots_switch(Model model, long[::1] nodesG, \
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
    simulate the system in equilibrium, take snapshots of the neighbourhood at
    distances up to maxDist centered around the given node, and count the number
    of occurrences per neighbourhood state
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
            with gil:
                state = (<Model> modelptr).encodeStateToString(allNeighboursIdx[d+1])
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
    simulate the system in equilibrium, take snapshots of the neighbourhood at
    distances up to maxDist centered around a node, and count the number
    of occurrences per neighbourhood state. Do this for all given nodes.
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
                with gil:
                    state = (<Model> modelptr).encodeStateToString(neighboursIdx[n][d+1])
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
    simulate the system in equilibrium, take samples of the neighbourhood at
    distances up to maxDist centered around the given node, and determine the
    distribution over neighbourhood magnetization levels (binned).
    If getFullSnapshots = 1: also take snapshots of the entire system to
                             determine pairwise MI and correlation
    """
    cdef:
        long[:,:,:,::1] avgSnapshots = np.zeros((repeats, maxDist, model.agentStates.shape[0], nBins), int)
        long[:,:,::1] avgSystemSnapshots = np.zeros((repeats, model.agentStates.shape[0], nBins), int)
        unordered_map[int, int] idxer

        int idx
        long nodeIdx, d, i, b, sample, rep, start
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, avg, avgSystem
        int nThreads = mp.cpu_count() if threads == -1 else threads
        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins + 1)[1:] # values represent upper bounds for bins
        vector[long] allNodes = list(model.mapping.values())
        unordered_map[long, vector[long]] allNeighboursIdx
        long[:,:,::1] fullSnapshots

    if getFullSnapshots: fullSnapshots = np.zeros((repeats, nSamples, model._nNodes), int)

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    nodeIdx = model.mapping[nodeG]
    for d in range(maxDist):
        allNeighboursIdx[d+1] = [model.mapping[n] for n in allNeighboursG[d+1]]

    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    i = repeats # somehow it doesn't compile when I directly use repeats

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
                # TODO: use string encoding?
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
    simulate the system in equilibrium, take samples of the neighbourhood at
    distances up to maxDist centered around a node, and determine the
    distribution over neighbourhood magnetization levels (binned).
    Do this for all given nodes.
    If getFullSnapshots = 1: also take snapshots of the entire system to
                             determine pairwise MI and correlation
    """
    cdef:
        long node, nNodes = nodesG.shape[0]
        long[::1] nodesIdx = np.zeros(nNodes, 'int')

        long[:,:,:,:,::1] avgSnapshots = np.zeros((repeats, nNodes, maxDist, model.agentStates.shape[0], nBins), int)
        long[:,:,:,::1] avgSystemSnapshots = np.zeros((repeats, nNodes, model.agentStates.shape[0], nBins), int)
        unordered_map[int, int] idxer
        vector[unordered_map[long, vector[long]]] neighboursIdx = vector[unordered_map[long, vector[long]]](nNodes)
        vector[long] allNodes = list(model.mapping.values())

        long d, i, b, sample, rep, n
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, avg, avgSystem
        int nThreads = mp.cpu_count() if threads == -1 else threads

        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins + 1)[1:] # values represent upper bounds for bins
        long[:,:,::1] fullSnapshots

    if getFullSnapshots: fullSnapshots = np.zeros((repeats, nSamples, model._nNodes), int)

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx


    for n in range(nNodes):
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
                # TODO: use string encoding?
                fullSnapshots[rep][sample] = (<Model>modelptr)._states

            with gil: pbar.update(1)

    if getFullSnapshots:
        return avgSnapshots.base, avgSystemSnapshots.base, fullSnapshots.base
    else:
        return avgSnapshots.base, avgSystemSnapshots.base




cpdef np.ndarray monteCarloFixedNeighbours(Model model, string snapshot, long nodeIdx, \
               vector[long] neighboursIdx, long nTrials, long burninSamples, \
               long nSamples = 10, long distSamples = 10):
      """
      python wrapper
      """
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
    """
    Fix the given neighbours to the states in the snapshot, then simulate the system in equilibrium
    and extract the conditional state distribution of the node of interest
    """
    cdef:
       double Z = <double> nSamples * nTrials
       double part = 1/Z
       long nodeIdx, nodeState, idx, n, rep, sample, nNeighbours = neighboursG.size()
       unordered_map[int, int] idxer #= {state : idx for idx, state in enumerate(model.agentStates)}
       double[::1] probCondArr #= np.zeros(idxer.size())
       long[::1] decodedStates, initialState
       int i
       vector[long] neighboursIdx = vector[long] (nNeighbours)

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    with gil: decodedStates = np.frombuffer(snapshot).astype(int)
    with gil: probCondArr = np.zeros(idxer.size())

    for idx in range(nNeighbours):
        n = neighboursG[idx]
        neighboursIdx[idx] = model._mapping[n] # map from graph to model index

    nodeIdx = model._mapping[nodeG]

    with gil: model.fixedNodes = neighboursIdx

    for trial in range(nTrials):
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


        with gil: model.setStates(initialState)
        #with gil: print({model.rmapping[idx]: initialState[idx] for idx in range(model._nNodes)})
        with gil: model.seed += 1
        #model._loadStatesFromString(decodedStates, neighbours) # keeps all other node states as they are
        model.simulateNSteps(burninSamples) # go to equilibrium

        for sample in range(nSamples):
            model.simulateNSteps(distSamples)
            nodeState = model._states[nodeIdx]
            probCondArr[idxer[nodeState]] += part

    with gil: model.releaseFixedNodes()

    #with gil: print(f'CHECK sum {np.sum(probCondArr.base)}')

    return probCondArr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef unordered_map[long, unordered_map[string, double]] _monteCarloFixedNeighboursStateDistr(Model model, string snapshot, long nodeIdx, long dist, \
               unordered_map[long, vector[long]] allNeighboursIdx, long nTrials, long burninSamples = int(1e3), \
               long nSamples = int(1e3), long distSamples = int(1e2), int initStateIdx = -1) nogil:


    cdef:
       long n, d, idx, rep, sample, minDist, length = allNeighboursIdx[dist].size()
       unordered_map[int, int] idxer
       long[::1] decodedStates, initialState
       int i
       string state
       unordered_map[long, unordered_map[string, double]] snapshots

    with gil: minDist = np.min(list(dict(allNeighboursIdx).keys()))

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


        with gil: model.setStates(initialState)
        with gil: model.seed += 1
        model.simulateNSteps(burninSamples) # go to equilibrium

        for sample in range(nSamples):
            model.simulateNSteps(distSamples)

            for d in range(minDist, dist):
                with gil: state = model.encodeStateToString(allNeighboursIdx[d])
                snapshots[d][state] += 1

    return snapshots



@cython.boundscheck(False)
@cython.wraparound(False)
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
       long[::1] decodedStates, initialState
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


        with gil: model.setStates(initialState)
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
        long minDist, nodeIdx, n, sample, idx, nNeighbours = allNeighboursG[dist].size()
        vector[long] neighboursIdx = vector[long](nNeighbours)
        string state
        double[::1] pY, pX, allHXgiveny
        double HX, HXgiveny, HXgivenY = 0, MI = 0

    for idx in range(nNeighbours):
        n = allNeighboursG[dist][idx]
        neighboursIdx[idx] = model.mapping[n] # map from graph to model index


    for tid in range(nThreads):
       tmp = copy.deepcopy(model)
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
        HX = entropyFromProbs(pX)
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




@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef np.ndarray magTimeSeries(Model model, long burninSamples, \
                                long nSamples, int abs=0):
    """
    python wrapper
    """
    return _magTimeSeries(model, burninSamples, nSamples, abs).base

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef double[::1] _magTimeSeries(Model model, long burninSamples, \
                                long nSamples, int abs=0):
    """
    simulate the system in equilibrium,
    determine the system magnetization at each time step
    """
    cdef:
        double[::1] mags = np.zeros(nSamples)
        long sample

    model.simulateNSteps(burninSamples)

    for sample in range(nSamples):
        mags[sample] = mean(model.simulateNSteps(1), model._nNodes, abs)

    return mags




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef double mean(long[::1] arr, long len, int abs=0) nogil:
    """"
    nogil implementation of mean
    if abs = 1: return abs(mean)
    """
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

    # simulate until mag has stabilized
    # remember mixing time needed
    if checkMixing:
        while True:
            mags = _magTimeSeries(model, 0, burninSteps, abs=1)
            allMags = np.hstack((allMags, mags.base))
            if counter >= nStepsRegress :
                # do linear regression
                slope, intercept, r_value, p_value, std_err = linregress(x, allMags[-nStepsRegress:])
                if 1 - p_value < threshold: # p-value of test for slope=0
                    break
            counter = counter + burninSteps
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

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
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
        long nNodes = model._nNodes
        long mixingTime
        double corrTime, intercept, meanMag = 0
        long idx
        np.ndarray tmp, mags, autocorr, initialConfigs = np.linspace(0.5, 1, nInitialConfigs)
        double prob, t
        long mixingTimeMax = 0
        double corrTimeMax = 0 # double because it might be infinity
        dict allMagSeries = {}


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

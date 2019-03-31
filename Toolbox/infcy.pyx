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
from scipy import stats
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

# print gil stuff; no mp used currently so ....useless
def checkDistribution():
    '''Warning statement'''
    from platform import platform
    if 'windows' in platform().lower():
        print('Warning: Windows detected. Please remember to respect the GIL'\
              ' when using multi-core functions')
checkDistribution() # print it only once

# TODO rework this to include agentStates
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef long encodeState(long[::1] state) nogil:
    """Maps state to decimal number"""
    cdef:
        long binNum = 1
        long N = state.shape[0]
        long i
        long dec = 0
    for i in range(N):
        if state[i] == 1:
            dec += binNum
        binNum *= 2
    return dec


# TODO rework this to include agentStates
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef vector[long] decodeState(long dec, long N) nogil:
    """Decodes decimal number to state"""
    cdef:
        long i = 0
        # long[::1] buffer = np.zeros(N, dtype = int) - 1
        vector [long] buffer = vector[long](N, -1) # init with -1
    while dec > 0:
        if dec % 2:
            buffer[i] = 1
        i += 1
        dec = dec // 2
    return buffer


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
cpdef dict getSnapShots(Model model, int nSamples, int step = 1,\
                   int burninSamples = int(1e3)):
    """
    Use single Markov chain to extract snapshots from the model
    """
    cdef:
        unordered_map[int, double] snapshots
        int i, sample
        int N          = nSamples * step
        long[:, ::1] r = model.sampleNodes(N) # returns N shuffled versions of node IDs
        double Z       = <double> nSamples
        int idx
        double past    = timer()
        list modelsPy  = []
        vector[PyObjectHolder] models_
        cdef int tid, nThreads = mp.cpu_count()
    # threadsafe model access; can be reduces to n_threads
    for sample in range(nSamples):
        tmp = copy.deepcopy(model)
        tmp.reset()
        tmp.seed += sample # enforce different seeds
        modelsPy.append(tmp)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    tid = threadid()
    pbar = tqdm(total = nSamples)
    for sample in prange(nSamples, nogil = True, \
                         schedule = 'static', num_threads = nThreads): # nSample independent models in parallel
        # perform n steps
        for i in range(step):
            (<Model> models_[sample].ptr)._updateState(r[(i + 1) * (sample + 1)])
        with gil:
            idx = encodeState((<Model> models_[sample].ptr)._states)
            snapshots[idx] += 1/Z # each index corresponds to one system state, the array contains the probability of each state
            pbar.update(1)

    # pbar = tqdm(total = nSamples)
    # model.reset() # start from random
    # for i in range(N):
    #     if i % step == 0:
    #         idx             = encodeState(model._states)
    #         snapshots[idx] += 1 / Z
    #         pbar.update(1)
    #     # model._updateState(r[i])
    #     model._updateState(r[i])
    # pbar.close()
    print(f'Found {len(snapshots)} states')
    print(f"Delta = {timer() - past: .2f} sec")
    return snapshots



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getSnapshotsPerDist(Model model, long nodeG, \
          long nSamples = int(1e3), long burninSamples = int(1e3), int maxDist = 1, double threshold = 0.05, int threads=-1):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take snapshots of the specified node subset, ignore all others
    """
    cdef:
        vector[unordered_map[string, long]] snapshots = vector[unordered_map[string, long]](maxDist)
        vector[unordered_map[string, long]] oldSnapshots = vector[unordered_map[string, long]](maxDist)
        unordered_map[int, int] idxer
        long[::1] centralNodeSamples = np.zeros(model.agentStates.shape[0], int)
        long nodeIdx, idx, d, i, sample
        double Z       = 0#= <double> nSamples
        #double part = 1/Z
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads
        int nodeSpin
        np.ndarray mse = np.ones(maxDist)

        unordered_map[long, vector[long]] allNeighbours_G
        unordered_map[long, vector[long]] allNeighbours_idx

    nodeIdx = model.mapping[nodeG] # map to internal index
    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(nodeG, maxDist)

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    # for testing
    #allNeighbours = model.neighboursAtDist(0, 3)
    #print(allNeighbours[2])

    while True:
        #pbar = tqdm(total = nSamples) # init  progbar
        for sample in prange(nSamples, nogil = True, schedule = 'static', num_threads = nThreads):
            tid = threadid()
            modelptr = models_[tid].ptr

            if Z == 0:
                with gil:
                    (<Model>modelptr).seed += sample # enforce different seeds
                    #print(f'{tid} seed: {(<Model> models_[tid].ptr).seed}')
                    (<Model>modelptr).reset()
                    #print(f'{tid} initial state: {(<Model> models_[tid].ptr)._states.base}')


            (<Model>modelptr).simulateNSteps(burninSamples)

            nodeSpin = (<Model>modelptr)._states[nodeIdx]
            centralNodeSamples[idxer[nodeSpin]] += 1
            #print(f'snapshot: {(<Model> models_[tid].ptr)._states.base}')
            for d in range(maxDist):
                #print(d, allNeighbours[d+1])
                with gil: state = (<Model> modelptr).encodeStateToString(allNeighbours_idx[d+1])
                #if(np.frombuffer(idx).size > allNeighbours[d+1].size()):
                #    print(f'error!!!! {d} {np.frombuffer(idx)} {allNeighbours[d+1]}')
                #    for i in range(allNeighbours[d+1].size()):
                #        print((<Model> models_[tid].ptr)._states[allNeighbours[d+1][i]])
                #print(d, np.fromstring(idx))
                snapshots[d][state] += 1 #part # each index corresponds to one system state, the array contains the probability of each state
            #print(f'{tid}final state: {(<Model> models_[tid].ptr)._states.base}')
            #with gil: pbar.update(1)
        # check mean squared error between previous distr and current distr of snapshots
        #mse = 0

        if Z > 0: # not in first iteration
            for d in range(maxDist):
                pNew = np.array([snapshots[d][k]/(Z+nSamples) for k in dict(snapshots[d])])
                pOld = np.array([oldSnapshots[d][k]/Z if k in dict(oldSnapshots[d]) else 0 for k in dict(snapshots[d])])
                KL = scipy.stats.entropy(pOld, pNew, base=2) # computes the Kullback-Leibler divergence: information gain if pNew is used instead of pOld
                #differences = np.array([(oldSnapshots[d][k]/Z - snapshots[d][k]/(Z+nSamples)) if k in dict(oldSnapshots[d]) else snapshots[d][k]/(Z+nSamples) for k in dict(snapshots[d])])

                #mse[d] = np.sum(np.power(differences, 2))
                mse[d] = KL
        oldSnapshots = snapshots
        Z += nSamples
        print(f'MSE = {mse}')
        if np.all(mse < threshold):
            break

    #cdef dict s = snapshots
    #print(f'Found {len(snapshots)} states with probs {list(s.values())}')
    #print(f"Time to get snapshots = {timer() - past: .2f} sec")
    #for d in range(maxDist):
    #    print(dict(snapshots[d]).values())
    #    snapshots[d] = {k: v / Z for k, v in dict(snapshots[d]).items()}
    return snapshots, centralNodeSamples.base, Z, allNeighbours_G, allNeighbours_idx



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getSnapshotsPerDist2(Model model, long nodeG, unordered_map[long, vector[long]] allNeighboursG, \
          long nSamples = int(1e2), long burninSamples = int(1e3), int maxDist = 1, int threads = -1):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take snapshots of the specified node subset, ignore all others
    """
    cdef:
        vector[unordered_map[string, double]] snapshots = vector[unordered_map[string, double]](maxDist)
        unordered_map[int, int] idxer
        #double[::1] centralNodeSamples = np.zeros(model.agentStates.shape[0])
        #double[::1] centralNodeSamplesOld = np.zeros(model.agentStates.shape[0])
        long nodeIdx, idx, d, i, sample
        double Z       = 0
        double part = 1/(<double> nSamples)
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin
        int nThreads = mp.cpu_count() if threads == -1 else threads
        double mse = 1, KL = 1

        #unordered_map[long, vector[long]] allNeighbours_G
        unordered_map[long, vector[long]] allNeighboursIdx

    nodeIdx = model.mapping[nodeG] # map to internal index
    for d in range(maxDist):
        allNeighboursIdx[d+1] = [model.mapping[n] for n in allNeighboursG[d+1]]

    # initialize thread-safe models. nThread MC chains will be run in parallel.
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.reset()
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    for sample in prange(nSamples, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        # when a thread has finished its first loop iteration, it will continue running and sampling from the MC chain of the same model, without resetting

        #with gil:
        #    (<Model>models_[tid].ptr).seed += sample # enforce different seeds
            #print(f'{tid} seed: {(<Model> models_[tid].ptr).seed}')
        #    (<Model>models_[tid].ptr).reset()
            #print(f'{tid} initial state: {(<Model> models_[tid].ptr)._states.base}')


        (<Model>modelptr).simulateNSteps(burninSamples)

        nodeSpin = (<Model>modelptr)._states[nodeIdx]

        for d in range(maxDist):
            with gil: state = (<Model> modelptr).encodeStateToString(allNeighboursIdx[d+1])
            snapshots[d][state] += part #part # each index corresponds to one system state, the array contains the probability of each state


    return snapshots, allNeighboursIdx #, allNeighbours_G, allNeighbours_idx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getSnapshotsPerDistNodes(Model model, long[::1] nodesG, long repeats=int(1e2), \
          long nSamples = int(1e3), long burninSamples = int(1e3), int maxDist = 1, int threads = -1):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take snapshots of neighbourhoods, ignore all others
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


    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.reset()
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    for sample in prange(nSamples, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        # when a thread has finished its first loop iteration, it will continue running and sampling from the MC chain of the same model, without resetting

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
cpdef tuple getJointSnapshotsPerDist2(Model model, long nodeG, unordered_map[long, vector[long]] allNeighboursG, long repeats=int(1e2), \
          long nSamples = int(1e3), long burninSamples = int(1e3), long distSamples=100, int maxDist = 1, long nBins=10, int threads = -1):
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

        long[:,:,::1] avgSnapshots = np.zeros((maxDist, model.agentStates.shape[0], nBins), int)
        unordered_map[int, int] idxer

        int idx
        long nodeIdx, d, i, b, sample, rep
        long Z = repeats * nSamples
        #double part = 1/Z
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, avg
        int nThreads = mp.cpu_count() if threads == -1 else threads
        #np.ndarray KL = np.ones(maxDist)
        #double KL_d
        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins)

        unordered_map[long, vector[long]] allNeighboursIdx

    #print(bins)

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    nodeIdx = model.mapping[nodeG]
    for d in range(maxDist):
        allNeighboursIdx[d+1] = [model.mapping[n] for n in allNeighboursG[d+1]]

    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    i = repeats

    for rep in prange(i, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        with gil:
            (<Model>modelptr).seed += rep # enforce different seeds
            #print(f'{tid} seed: {(<Model> models_[tid].ptr).seed}')
            (<Model>modelptr).reset()
            #print(f'{tid} initial state: {(<Model> models_[tid].ptr)._states.base}')

        (<Model>modelptr).simulateNSteps(burninSamples)

        for sample in range(nSamples):
            (<Model>modelptr).simulateNSteps(distSamples)

            nodeSpin = (<Model> modelptr)._states[nodeIdx]
            for d in range(maxDist):
                #with gil: state = (<Model> modelptr).encodeStateToString(allNeighbours_idx[d+1])
                #snapshots[d][nodeSpin][state] += 1
                avg = (<Model> modelptr).encodeStateToAvg(allNeighboursIdx[d+1], bins)
                #avgSnapshots[d][nodeSpin][avg] +=1
                avgSnapshots[d][idxer[nodeSpin]][avg] += 1
                #with gil: print(rep, avgSnapshots[d])

    return avgSnapshots.base, Z

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getJointSnapshotsPerDistNodes(Model model, long[::1] nodesG, long repeats=int(1e2), \
          long nSamples = int(1e3), long burninSamples = int(1e3), long distSamples=100, int maxDist = 1, long nBins=10, int threads = -1):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take snapshots of the specified node subset, ignore all others
    """
    cdef:
        long nNodes = nodesG.shape[0]
        long[::1] nodesIdx = np.zeros(nNodes, 'int')

        #vector[vector[unordered_map[int, unordered_map[int, double]]]] avgSnapshots = vector[vector[unordered_map[int, unordered_map[int, double]]]](nNodes)
        long[:,:,:,::1] avgSnapshots = np.zeros((nNodes, maxDist, model.agentStates.shape[0], nBins), int)
        unordered_map[int, int] idxer
        vector[unordered_map[long, vector[long]]] neighboursIdx = vector[unordered_map[long, vector[long]]](nNodes)
        vector[unordered_map[long, vector[long]]] neighboursG = vector[unordered_map[long, vector[long]]](nNodes)

        long d, i, b, sample, rep, n
        long Z = repeats * nSamples
        #double part = 1/Z
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, avg
        int nThreads = mp.cpu_count() if threads == -1 else threads
        #np.ndarray KL = np.ones(maxDist)
        #double KL_d
        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins)

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx


    for n in range(nNodes):
        #avgSnapshots[n] = vector[unordered_map[int, unordered_map[int, double]]](maxDist)
        nodesIdx[n] = model.mapping[nodesG[n]]
        neighboursG[n], neighboursIdx[n] = model.neighboursAtDist(nodesG[n], maxDist)


    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    i = repeats

    for rep in prange(i, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        with gil:
            (<Model>modelptr).seed += rep # enforce different seeds
            #print(f'{tid} seed: {(<Model> models_[tid].ptr).seed}')
            (<Model>modelptr).reset()
            #print(f'{tid} initial state: {(<Model> models_[tid].ptr)._states.base}')

        (<Model>modelptr).simulateNSteps(burninSamples)

        for sample in range(nSamples):
            (<Model>modelptr).simulateNSteps(distSamples)

            for n in range(nNodes):
                nodeSpin = (<Model> modelptr)._states[nodesIdx[n]]
                for d in range(maxDist):
                    avg = (<Model> modelptr).encodeStateToAvg(neighboursIdx[n][d+1], bins)
                    #avgSnapshots[n][d][nodeSpin][avg] +=1
                    avgSnapshots[n][d][idxer[nodeSpin]][avg] +=1

    return avgSnapshots.base, Z, neighboursG



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getJointSnapshotsPerDist(Model model, long nodeG, unordered_map[long, vector[long]] allNeighboursG, \
          long nSamples = int(1e3), long burninSamples = int(1e3), int maxDist = 1, int nBins=10, double threshold = 0.05, int threads = -1):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Only take snapshots of the specified node subset, ignore all others
    """
    cdef:
        vector[unordered_map[int, unordered_map[string, double]]] snapshots = vector[unordered_map[int, unordered_map[string, double]]](maxDist)
        vector[unordered_map[int, unordered_map[string, double]]] oldSnapshots = vector[unordered_map[int, unordered_map[string, double]]](maxDist)
        vector[unordered_map[int, unordered_map[int, double]]] avgSnapshots = vector[unordered_map[int, unordered_map[int, double]]](maxDist)
        vector[unordered_map[int, unordered_map[int, double]]] oldAvgSnapshots = vector[unordered_map[int, unordered_map[int, double]]](maxDist)


        long nodeIdx, d, i, sample
        double Z       = 0#= <double> nSamples
        #double part = 1/Z
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, avg
        int nThreads = mp.cpu_count() if threads == -1 else threads
        np.ndarray KL = np.ones(maxDist)
        double KL_d
        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins)

        unordered_map[long, vector[long]] allNeighboursIdx

    nodeIdx = model.mapping[nodeG]
    for d in range(maxDist):
        allNeighboursIdx[d+1] = [model.mapping[n] for n in allNeighboursG[d+1]]

    #allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)

    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    # for testing
    #allNeighbours = model.neighboursAtDist(0, 3)
    #print(allNeighbours[2])

    while True:
        #pbar = tqdm(total = nSamples) # init  progbar
        for sample in prange(nSamples, nogil = True, schedule = 'static', num_threads = nThreads):
            tid = threadid()
            modelptr = models_[tid].ptr
            if Z == 0:
                with gil:
                    (<Model>modelptr).seed += sample # enforce different seeds
                    #print(f'{tid} seed: {(<Model> models_[tid].ptr).seed}')
                    (<Model>modelptr).reset()
                    #print(f'{tid} initial state: {(<Model> models_[tid].ptr)._states.base}')


            (<Model>modelptr).simulateNSteps(burninSamples)

            #print(f'snapshot: {(<Model> models_[tid].ptr)._states.base}')
            nodeSpin = (<Model> modelptr)._states[nodeIdx]
            #with gil: print((<Model> modelptr)._states.base)
            for d in range(maxDist):
                with gil: state = (<Model> modelptr).encodeStateToString(allNeighboursIdx[d+1])
                avg = (<Model> modelptr).encodeStateToAvg(allNeighboursIdx[d+1], bins)
                #if(np.frombuffer(idx).size > allNeighbours[d+1].size()):
                #    print(f'error!!!! {d} {np.frombuffer(idx)} {allNeighbours[d+1]}')
                #    for i in range(allNeighbours[d+1].size()):
                #        print((<Model> models_[tid].ptr)._states[allNeighbours[d+1][i]])
                #snapshots[d][idx] += 1 #part # each index corresponds to one system state, the array contains the probability of each state
                snapshots[d][nodeSpin][state] += 1
                avgSnapshots[d][nodeSpin][avg] +=1
            #print(f'{tid}final state: {(<Model> models_[tid].ptr)._states.base}')
            #with gil: pbar.update(1)
        # check mean squared error between previous distr and current distr of snapshots
        #mse = 0
        if Z > 0:
            for d in range(maxDist):
                pNew = np.array([avgSnapshots[d][s][k]/(Z+nSamples) for s in model.agentStates for k in dict(avgSnapshots[d][s])])
                pOld = np.array([oldAvgSnapshots[d][s][k]/Z if k in dict(oldAvgSnapshots[d][s]) else 0 for s in model.agentStates for k in dict(avgSnapshots[d][s])])
                KL_d = scipy.stats.entropy(pOld, pNew, base=2) # computes the Kullback-Leibler divergence: information gain if pNew is used instead of pOld
                #differences = np.array([(oldSnapshots[d][s][k]/Z - snapshots[d][s][k]/(Z+nSamples)) if k in dict(oldSnapshots[d][s]) \
                #                        else snapshots[d][s][k]/(Z+nSamples) for s in model.agentStates for k in dict(snapshots[d][s])])

                #mse[d] = np.sum(np.power(differences, 2))
                KL[d] = KL_d
        oldAvgSnapshots = avgSnapshots
        Z += nSamples
        print(f'KL = {KL}')
        if np.all(KL < threshold):
            break

    #cdef dict s = snapshots
    #print(f'Found {len(snapshots)} states with probs {list(s.values())}')
    #print(f"Time to get snapshots = {timer() - past: .2f} sec")
    #for d in range(maxDist):
    #    for s in model.agentStates:
    #        snapshots[d][s] = {k: v / Z for k, v in dict(snapshots[d][s]).items()}
    return snapshots, avgSnapshots, Z


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
cdef double[::1] _monteCarloFixedNeighbours(Model model, string snapshot, long nodeIdx, \
               vector[long] neighboursIdx, long nTrials, long burninSamples = int(1e3), \
               long nSamples = int(1e3), long distSamples = int(1e2)) nogil:


    #with gil: past = timer()

    cdef:
       double Z = <double> nSamples * nTrials
       double part = 1/Z
       long idx, rep, sample, length = neighboursIdx.size()
       long nodeState
       #unordered_map[long, double] probCond
       unordered_map[int, int] idxer #= {state : idx for idx, state in enumerate(model.agentStates)}
       double[::1] probCondArr #= np.zeros(idxer.size())
       #vector[double] probCondVec = vector[double](model.agentStates.shape[0], 0)
       long[::1] decodedStates
       long[::1] initialState

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    with gil: decodedStates = np.frombuffer(snapshot).astype(int)

    with gil: probCondArr = np.zeros(idxer.size())

    #for idx in range(length):
    #    n = neighbours[idx]
    #    initialState[n] = decodedStates[idx]


    #with gil: print('start repetitions')
    #for rep in range(repeats):

        #with gil: model.seed += rep # enforce different seeds

    #model._loadStatesFromString(decodedStates, neighbours) # keeps all other node states as they are
    #model.simulateNSteps(burninSamples)

    #with gil: print(snapshot)

    for trial in range(nTrials):
        #with gil: print(trial, part, probCondArr.base)
        # set states without gil
        with gil: initialState = np.random.choice(model.agentStates, size = model._nNodes)

        for idx in range(length):
            n = neighboursIdx[idx]
            initialState[n] = decodedStates[idx]


        model._setStates(initialState)
        with gil: model.seed += 1
        #model._loadStatesFromString(decodedStates, neighbours) # keeps all other node states as they are
        model.simulateNSteps(burninSamples) # go to equilibrium

        for sample in range(nSamples):
            model.simulateNSteps(distSamples)
            nodeState = model._states[nodeIdx]

            #with gil: print(nodeState, model._states[neighbours[0]])
            #probCond[nodeState] += part
            #with gil: print(f'before: {probCondArr.base}, {part}')
            probCondArr[idxer[nodeState]] += part
            #with gil: print(f'after: {probCondArr.base}, {part}')
            #probCondVec[idxer[nodeState]] += part

    #print(f"time for sampling = {timer() - past: .2f} sec")
    #model.releaseFixedNodes()

    #with gil: print(f'{timer() - past} sec')

    return probCondArr

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



cpdef tuple neighbourhoodMI(Model model, long nodeG, vector[long] neighboursG, unordered_map[string, double] snapshots, \
              long nTrials, long burninSamples, long nSamples, long distSamples, int threads = -1):
    cdef:
        Model tmp
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads
        #vector[long] neighbours = model.neighboursAtDist(node, dist)[dist]
        long nodeIdx, n, sample, idx, nNeighbours = neighboursG.size()
        vector[long] neighboursIdx = vector[long](nNeighbours)
        #long totalSnapshots = nSnapshots * nNeighbours
        #double part = 1 / (<double> totalSnapshots)
        #unordered_map[string, double] snapshots
        string state
        double[::1] pY, pX
        double HXgiveny, HXgivenY = 0, MI = 0

    for idx in range(nNeighbours):
        n = neighboursG[idx]
        neighboursIdx[idx] = model.mapping[n] # map from graph to model index
    #print(neighbours)

    for tid in range(nThreads):
       tmp = copy.deepcopy(model)
       tmp.fixedNodes = neighboursIdx
       models_.push_back(PyObjectHolder(<PyObject *> tmp))

    nodeIdx = model.mapping[nodeG]
    #print(node)

    #print(f'neighbours at dist={dist}: {neighbours}')

    # get snapshots and their probabilities
    #snapshots = getSnapShotsLargeNetwork(model, nSnapshots*nNeighbours, neighbours, nStepsToSnapshot)

    # TODO extract snapshots for all distances at once. Then pass snapshots to MI computation
    """
    pbar = tqdm(total = totalSnapshots) # init  progbar
    for sample in prange(totalSnapshots, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        with gil:
            (<Model>models_[tid].ptr).seed += sample # enforce different seeds
            (<Model>models_[tid].ptr).reset()

        (<Model>models_[tid].ptr).simulateNSteps(nStepsToSnapshot)

        with gil:
            state = (<Model> models_[tid].ptr).encodeStateToString(neighbours)
            snapshots[state] += part
            pbar.update(1)
    """

    cdef:
        dict snapshotsDict = snapshots
        vector[string] keys = list(snapshotsDict.keys())
        double[::1] probs = np.array([snapshots[k] for k in keys])
        double[:,::1] container = np.zeros((keys.size(), model.agentStates.shape[0]))

    #print(f'Found {len(snapshots)} states')

    # fix neighbour states
    #for tid in range(nThreads):
    #    (<Model> models_[tid].ptr).fixedNodes = neighbours # fix neighbour states

    # get conditional probabilities
    pbar = tqdm(total = keys.size())
    for idx in prange(keys.size(), nogil = True, schedule = 'dynamic', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        #with gil: past = timer()
        container[idx] = _monteCarloFixedNeighbours((<Model>modelptr), \
                        keys[idx], nodeIdx, neighboursIdx, nTrials, burninSamples, nSamples, distSamples)
        #with gil: print(f'{timer() - past} sec')
        #with gil: print(np.fromstring(keys[idx]), container.base[idx])
        HXgiveny = entropyFromProbs(container[idx])
        HXgivenY -= probs[idx] * HXgiveny

        with gil: pbar.update(1)

    # compute MI based on conditional probabilities
    pX = np.sum(np.multiply(probs.base, container.base.transpose()), axis=1)
    #print(pX.base)
    MI = HXgivenY + entropyFromProbs(pX)

    print(f'MI= {MI}')

    return snapshotsDict, container.base, MI

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

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef dict monteCarlo(\
               Model model, dict snapshots,
               int deltas = 10,  int repeats = 11,
               int nThreads = -1):
    """
    Monte carlo sampling of the snapshots
    ISSUES:
        currently have to enforce the gil in order to not overwrite
        the model states. Best would be to copy the extensions. However,
        I dunno how to properly reference them in arrays
    """
    # TODO: solve the memory issues;  currently way too much is ofloaded to the memory
    # one idea would be to have the buffers inside the for loop. However prange is not possible
    # if that is used.
    print('Pre-computing rngs')
    cdef:
        float past = timer()
    # pre-declaration
        double Z              = <double> repeats
        double[::1] copyNudge = model.nudges.copy()
        bint reset            = True
        # loop stuff
        # extract startstates
        # list comprehension is slower than true loops cython
        long[:, ::1] s = np.array([decodeState(i, model._nNodes) for i in tqdm(snapshots)]) # array of system states
        int states     = len(snapshots) # number of different states observed

        # CANT do this inline which sucks either assign it below with gill or move this to proper c/c++
        # loop parameters
        int repeat, delta, node, statei, half = deltas // 2, state
        vector[int] kdxs        = list(snapshots.keys()) # extra mapping idx (list of decimal representations of states)
        dict conditional = {}
        # unordered_map[int, double *] conditional
        long[::1] startState
        int jdx
        double[:, :, :, ::1] out     = np.zeros((states , (deltas), model._nNodes, model._nStates))
        long[  :,       ::1] r       = model.sampleNodes( states * (deltas) * repeats)
        # list m = []

        int nNodes = model._nNodes, nStates = model._nStates
        long[::1] agentStates = model.agentStates
        str nudgeType = model._nudgeType

        unordered_map[int, int] idxer = {state : idx for idx, state in enumerate(agentStates)}

        list modelsPy = []
        vector[PyObjectHolder] models_
        Model tmp

    # threadsafe model access; can be reduces to n_threads
    for state in range(states):
        tmp = copy.deepcopy(model)
        tmp.seed += state # enforce different seeds
        modelsPy.append(tmp)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    print('starting runs')
    if nThreads == -1:
        nThreads = mp.cpu_count()
    pbar = tqdm(total = states) # init  progbar
    for state in prange(states, nogil = True, schedule = 'static', num_threads = nThreads):
    # for state in range(states):
        # with gil:
        for repeat in range(repeats):
            # reset the buffers to the start state
            # model._states[:] = s[state]
            # model._nudges[:] = copyNudge
            # (<Model>models_[state].ptr)._states = s[state] # this overwrites s ; don't
            # (<Model>models_[state].ptr)._nudges = copyNudge
            # only copy values
            for node in range(nNodes):
                # kinda uggly syntax
                (<Model>models_[state].ptr)._states[node] = s[state][node]
                (<Model>models_[state].ptr)._nudges[node] = copyNudge[node]
                # (<Model>models[n])._nudges[node] = copyNudge[node]
            #     model._states[node] = s[state][node]
            #     model._nudges[node] = copyNudge[node]
            # reset simulation
            # sample for N times
            for delta in range(deltas ):
                # bin data
                for node in range(nNodes):
                    out[state, delta, node, idxer[(<Model>models_[state].ptr)._states[node]]] += 1 / Z
                    # for statei in range(nStates):
                        # if (<Model>models[n])._states[node] == agentStates[statei]:
                        # if model._states[node] == model.agentStates[statei]:
                            # out[state, delta, node, statei] += 1 / Z
                            # break
                # update
                jdx  = (delta + 1) * (repeat + 1)  * (state + 1)- 1
                # (<Model>models[n])._updateState(r[jdx])
                # model._updateState(model.sampleNodes(1)[0])
                (<Model>models_[state].ptr)._updateState(r[jdx])
                # with gil:
                #     print(np.all((<Model>models_[state].ptr)._states.base == s[state].base))
                # turn-off the nudges
                    # check for type of nudge
                if nudgeType == 'pulse' or \
                nudgeType    == 'constant' and delta >= half:
                    # (<Model>models[n])._nudges[:] =
                    (<Model>models_[state].ptr)._nudges[:] = 0
                    # for node in range(nNodes):
                        # model._nudges[node] = 0
                    # printf('%d %d\n', tid, deltas)
                    # with gil:

                    # model._nudges[:] = 0
        # TODO: replace this with a concurrent unordered_map
        with gil:
            pbar.update(1)
            conditional[kdxs[state]] = out.base[state]# [state, 0, 0, 0]
    # cdef unordered_map[int, double *].iterator start = conditional.begin()
    # cdef unordered_map[int, double *].iterator end   = conditional.end()
    # cdef int length = (deltas + 1) * nNodes * nStates
    # cdef np.ndarray buffer = np.zeros(length)
    # cdef _tmp = {}
    # while start != end:
    #     idx = deref(start).first
    #     for state in range(length):
    #         buffer[state] = deref(start).second[state]
    #     _tmp[idx] = buffer.reshape((deltas + 1, nNodes, nStates)).copy()
    #     prec(start)


    # # free memory
    # for nThread in range(n):
    #     Py_XDECREF(models.at(nThread))
    pbar.close()
    print(f"Delta = {timer() - past: .2f} sec")
    return conditional

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
cpdef long[:,::1] equilibriumSampling(Model model, long repeats, long burninSamples, long nSamples, long distSamples, int threads = -1):
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
            (<Model>modelptr).reset()
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
                      int switch=1, double threshold=0.05, int threads = -1):
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
            (<Model>modelptr).reset()
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
cpdef double mutualInformationIDL(long[:,::1] snapshots, double[::1] binEntropies, long nodeIdx1, long nodeIdx2) nogil:
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




@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef double[::1] MIAtDist(Model model, long[:,::1] snapshots, \
              double[::1] entropies, long nodeIdx, vector[long] neighboursIdx) nogil:

    cdef:
        #int[::1] neighbours
        #vector[int] neighbours
        long idx, n = neighboursIdx.size() #.shape[0]
        #vector[double] MI
        double[::1] out

    with gil: # TODO possible without GIL?
        #neighbors = model.neighborsAtDist(node, dist)
        #neighbours = allNeighbours[dist]
        #n = neighbors.shape[0]
        #out = np.full(3, np.nan)
        out = np.full(model._nNodes, np.nan)
        #print("num neighbors: {}".format(n))

    if n > 0:
        #MI = vector[double](n, -1)
        #MI = vector[double](model._nNodes, -1)

        for idx in range(n):
            #MI[idx] = mutualInformationIDL(snapshots, entropies, node, neighbors[idx])
            out[idx] = mutualInformationIDL(snapshots, entropies, nodeIdx, neighboursIdx[idx])

        #with gil:
        #    out[0] = np.mean(MI)
        #    out[1] = np.std(MI)
        #    out[2] = np.sum(MI)

    return out
    #return np.mean(MI_array), np.std(MI_array)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef double[::1] corrAtDist(Model model, long[:,::1] snapshots, \
              long nodeIdx, vector[long] neighboursIdx) nogil:

    cdef:
        long idx, n = neighboursIdx.size() #.shape[0]
        double[::1] out

    with gil: # TODO possible without GIL?
        out = np.full(model._nNodes, np.nan)

    if n > 0:
        for idx in range(n):
            out[idx] = spinCorrelation(snapshots, nodeIdx, neighboursIdx[idx])

    return out


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
cpdef tuple runMI(Model model, np.ndarray nodesG, long repeats=10, long burninSamples=100, long nSamples=100, \
                  long distSamples=100, int distMax=1, \
                  double magThreshold=0, int threads = -1):

    cdef:
        #long[::1] cv_nodes = nodes
        long[:,::1] snapshots
        double[::1] entropies
        long n, d, nNodes = nodesG.shape[0]
        long[::1] nodesIdx = np.array([model.mapping[n] for n in nodesG])
        double[:,:,::1] MI = np.zeros((nNodes, distMax, model._nNodes))
        double[:,:,::1] corr = np.zeros((nNodes, distMax, model._nNodes))
        int nThreads = mp.cpu_count() if threads == -1 else threads
        unordered_map[long, vector[long]] allNeighboursIdx
        int[::1] neighbours

    # run multiple MC chains in parallel and sample snapshots
    if magThreshold == 0:
        snapshots = equilibriumSampling(model, repeats, burninSamples, nSamples, distSamples, threads=threads)
    elif magThreshold > 0:
        # only sample snapshots with abs avg mag larger than magThreshold
        snapshots = equilibriumSamplingMagThreshold(model, repeats, burninSamples, nSamples, distSamples, switch=0, threshold=magThreshold, threads=threads)
    else:
      # only sample snapshots with abs avg mag smaller than magThreshold
        snapshots = equilibriumSamplingMagThreshold(model, repeats, burninSamples, nSamples, distSamples, switch=1, threshold=np.abs(magThreshold), threads=threads)

    entropies = binaryEntropies(snapshots)

    for n in prange(nNodes, nogil = True, \
                         schedule = 'dynamic', num_threads = nThreads):

        with gil: _, allNeighboursIdx = model.neighboursAtDist(nodesG[n], distMax)

        for d in range(distMax):
            #with gil: neighbours = allNeighbours[d+1]
            MI[n][d] = MIAtDist(model, snapshots, entropies, nodesIdx[n], allNeighboursIdx[d+1])
            corr[n][d] = corrAtDist(model, snapshots, nodesIdx[n], allNeighboursIdx[d+1])

    #degrees = [model.graph.degree(n) for n in nodes]

    return snapshots.base, MI.base, corr.base #, degrees


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
        double[:,::1] results = np.zeros((3, nTemps))



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

        # simulate until equilibrium reached
        (<Model>modelptr).simulateNSteps(burninSamples)

        mag_sum = simulateGetMeanMag((<Model>modelptr), n)

        m = mag_sum[0] / n
        results[0][t] = m if m > 0 else -m
        results[1][t] = ((mag_sum[1] / n) - (m * m)) / temps_cview[t] # susceptibility
        results[2][t] = 1 - (mag_sum[3]/n) / (3 * (mag_sum[1]/n)**2) # Binder's cumulant

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
        double sum, sum_2, sum_3, sum_4
        vector[double] out = vector[double](4,0)

    sum = 0
    sum_2 = 0
    sum_3 = 0
    sum_4 = 0
    # collect magnetizations
    for step in range(nSamples):
        m = mean(model._updateState(r[step]), model._nNodes)
        sum = sum + m
        sum_2 = sum_2 + (m*m)
        sum_3 = sum_3 + (m**3)
        sum_4 = sum_4 + (m**4)

    out[0] = sum
    out[1] = sum_2
    out[2] = sum_3
    out[3] = sum_4

    return out


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

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef mutualInformation(dict conditional, int deltas, \
                          dict snapshots, Model model):
    '''
    Returns the node distribution and the mutual information decay
    '''
    cdef  np.ndarray px = np.zeros((deltas, model._nNodes, model._nStates))
    cdef  np.ndarray H  = np.zeros((deltas, model._nNodes))
    for key, p in conditional.items():
        # p    = np.asarray(p)
        H   -= np.nansum(p * np.log2(p), -1) * snapshots[key]
        px  += p  * snapshots[key] # update node distribution
    H += np.nansum(px *  np.log2(px), -1)
    return px, -H

cpdef runMC(Model model, dict snapshots, int deltas, int repeats):
    """ wrapper to perform MC and MI"""
    cdef:
        dict conditional = monteCarlo(model = model, snapshots = snapshots,\
                        deltas = deltas, repeats = repeats,\
                        )
        np.ndarray px, mi
    px, mi = mutualInformation(conditional, deltas, snapshots, model)
    return conditional, px, mi
    # @cython.boundscheck(False) # compiler directive
    # @cython.wraparound(False) # compiler directive
    # @cython.nonecheck(False)
    # @cython.cdivision(True)
    # cpdef dict monteCarlo(\
    #                Model model, dict snapshots,
    #                int deltas = 10,  int repeats = 11,
    #                ):
    #
    #     cdef float past = time.process_time()
    #      # store nudges already there
    #     cdef list models = []
    #     cdef dict params
    #     import copy
    #     params = dict(\
    #                 model      = model,\
    #                 # graph      = model.graph,\
    #                 # nudges     = model.nudges.base.copy(),\
    #                 temp       = model.t,\
    #                 repeats    = repeats,\
    #                 deltas     = deltas,\
    #                 )
    #     from functools import partial
    #     f = partial(worker, **params)
    #     print(f)
    #     cdef np.ndarray s = np.array([q for q in snapshots])
    #     cdef int n = len(s) // (mp.cpu_count() - 1)
    #     if n == 0:
    #         n = 1
    #     cdef list states  = [s[i : i + n] for i in range(0, len(s), n)]
    #     cdef dict conditional = {}
    #     # with mp.Pool(2) as p:
    #     with mp.Pool(mp.cpu_count() - 1) as p:
    #         for res in p.imap(f, tqdm(states)):
    #             for k, v in res.items():
    #                 conditional[k] = v
    #         # conditional = {kdx : res for kdx, res in zip(snapshots, p.map(f, tqdm(models)))}
    #     # print(conditional)
    #     print(f"Delta = {time.process_time() - past}")
    #     return conditional
    #
    #
    #     # object graph,\
    #     # np.ndarray nudges,\
    # @cython.boundscheck(False) # compiler directive
    # @cython.wraparound(False) # compiler directive
    # @cython.nonecheck(False)
    # @cython.cdivision(True)
    # @cython.initializedcheck(False)
    # cpdef dict worker(\
    #                 np.ndarray idx,\
    #                 Model model,\
    #                 double temp,\
    #                 int repeats, \
    #                 int deltas,\
    #                   ):
    #     # setup the worker
    #     # cdef Ising model = copy.deepcopy(Ising(graph, temperature = temp, updateType = 'single'))
    #     # model.nudges     = nudges.copy()
    #     # cdef Model model = copy.deepcopy(m)
    #     # cdef Ising model = copy.deepcopy(model)
    #     # model.nudges = nudges.copy()
    #     print(id(model))
    #     cdef dict conditional = {}
    #     # print(model.seed)
    #     cdef int states            = idx.size
    #     # decode the states
    #     cdef int nNodes            = model.nNodes
    #     cdef int nStates           = model.nStates
    #     cdef str nudgeType         = model.nudgeType
    #     cdef double[::1] copyNudge = model.nudges.base.copy()
    #
    #     cdef long[:, ::1] s = np.asarray([decodeState(i, nNodes) for i in idx])
    #     cdef long[:, ::1] r = model.sampleNodes( states * (deltas + 1) * repeats)
    #     # mape state to index
    #     cdef unordered_map[int, int] idxer = {i : j for j, i in enumerate(model.agentStates)}
    #     cdef double[:, :, :, ::1] out = np.zeros((states, deltas + 1, nNodes, nStates))
    #     cdef int half = deltas // 2
    #     cdef state, repeat, node, jdx
    #     cdef double Z = <double> repeats
    #     print(id(model), id(model.states.base), mp.current_process().name, id(model.nudges.base))
    #     cdef bint reset
    #     for state in range(states):
    #         # with gil:
    #         for repeat in range(repeats):
    #             # reset the buffers to the start state
    #             # model._states[:] = s[state]
    #             # model._nudges[:] = copyNudge
    #             for node in range(nNodes):
    #                 model._states[node] = s[state][node]
    #                 model._nudges[node] = copyNudge[node]
    #             # reset simulation
    #             reset   = True
    #             # sample for N times
    #             for delta in range(deltas + 1):
    #                 # bin data
    #                 for node in range(nNodes):
    #                     out[state, delta, node, idxer[model._states[node]]] += 1 / Z
    #                 # update
    #                 jdx  = (delta + 1) * (repeat + 1)  * (state + 1) - 1
    #                 # (<Model>models[n])._updateState(r[jdx])
    #                 # model._updateState(model.sampleNodes(1)[0])
    #                 model._updateState(r[jdx])
    #                 # turn-off the nudges
    #                 if reset:
    #                     # check for type of nudge
    #                     if nudgeType == 'pulse' or \
    #                     nudgeType    == 'constant' and delta >= half:
    #                         for node in range(nNodes):
    #                             model._nudges[node] = 0
    #                         reset            = False
    #         # pbar.update(1)
    #         conditional[idx[state]] = out.base[state]
    #     return conditional
    #
    # #


# # belongs to worker
# @cython.boundscheck(False) # compiler directive
# @cython.wraparound(False) # compiler directive
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef dict monteCarlo(\
#                Model model, dict snapshots,
#                int deltas = 10,  int repeats = 11,
#                ):
#
#     cdef float past = time.process_time()
#      # store nudges already there
#     cdef list models = []
#     cdef dict params
#     import copy
#     for startidx, val in snapshots.items():
#         params = dict(\
#                     model      = model,\
#                     repeats    = repeats,\
#                     deltas     = deltas,\
#                     idx        = startidx,\
#                     startState = np.asarray(decodeState(startidx, model._nNodes)),\
#                     )
#
#         models.append(Worker(**params))
#     # cdef np.ndarray s = np.array([decodeState(q, model._nNodes) for q in snapshots], ndmin = 2)
#     cdef dict conditional = {}
#     # with mp.Pool(2) as p:
#     with mp.Pool(3) as p:
#         for kdx, res in zip(snapshots, p.apply_async(f, tqdm(models)):
#             conditional[kdx] = res
#         # conditional = {kdx : res for kdx, res in zip(snapshots, p.map(f, tqdm(models)))}
#     print(conditional)
#     print(f"Delta = {time.process_time() - past}")
#     return conditional


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

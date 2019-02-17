# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Casper van Elteren'

# MODELS
# from Models.models cimport Model
from Models.models cimport Model

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

# progressbar
from tqdm import tqdm   #progress bar

# cython
from libcpp.vector cimport vector
from libc.stdlib cimport srand, malloc, free
from libc.math cimport exp, log2
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
cpdef int encodeState(long[::1] state) nogil:
    """Maps state to decimal number"""
    cdef:
        int binNum = 1
        int N = state.shape[0]
        int i
        int dec = 0
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
cpdef vector[long] decodeState(int dec, int N) nogil:
    """Decodes decimal number to state"""
    cdef:
        int i = 0
        # long[::1] buffer = np.zeros(N, dtype = int) - 1
        vector [long] buffer = vector[long](N, -1) # init with -1
    while dec > 0:
        if dec % 2:
            buffer[i] = 1
        i += 1
        dec = dec // 2
    return buffer

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
cpdef long[:,::1] collectSnapshots(Model model, int repeats, int burninSamples, int nSamples, int distSamples):
    cdef:
        #list modelsPy  = []
        vector[PyObjectHolder] models_
        Model tmp
        int start, sample, idx, step, nThreads = mp.cpu_count()
        long rep, s, tid
        long[:,::1] snapshots = np.zeros((repeats * nSamples, model._nNodes), int) #np.intc)


    #for rep in range(repeats):
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        #tmp.seed += rep # enforce different seeds
        #modelsPy.append(tmp)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    pbar = tqdm(total = repeats) # init  progbar
    for rep in prange(repeats, nogil = True, schedule = 'dynamic', num_threads = nThreads):
        tid = threadid()
        with gil:
            (<Model>models_[tid].ptr).reset()
            (<Model>models_[tid].ptr).seed += rep
        (<Model>models_[tid].ptr).simulateNSteps(burninSamples)
        tid = threadid()
        #with gil: print("thread {}".format(tid))
        start = rep * nSamples

        for sample in range(nSamples):
            snapshots[start + sample] = (<Model>models_[tid].ptr).simulateNSteps(distSamples)
            #snapshots[start + sample] = encodeState((<Model>models_[rep].ptr)._states)
            #with gil:
                #code = encodeState((<Model>models_[rep].ptr)._states)
                #print("raw states: {}".format((<Model>models_[rep].ptr)._states.base))
                #print("decoded   : {}".format(decodeState(code, model._nNodes)))
                #print("difference: {}".format(np.array((<Model>models_[rep].ptr)._states.base) - np.array(decodeState(code, model._nNodes))))
        with gil:
            pbar.update(1)

    return snapshots


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef long[:,::1] collectSnapshotsMagThreshold(Model model, int repeats, int burninSamples, int nSamples, int distSamples, int switch=1, double threshold=0.05):
    cdef:
        #list modelsPy  = []
        vector[PyObjectHolder] models_
        Model tmp
        int start, sample = 0, idx, step, nThreads = mp.cpu_count()
        long rep, s, tid, nNodes = model._nNodes
        double mu
        long[:,::1] snapshots = np.zeros((repeats * nSamples, nNodes), int) #np.intc)


    #for rep in range(repeats):
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        #tmp.seed += rep # enforce different seeds
        #modelsPy.append(tmp)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    pbar = tqdm(total = repeats) # init  progbar
    for rep in prange(repeats, nogil = True, schedule = 'dynamic', num_threads = nThreads):
        tid = threadid()
        with gil:
            (<Model>models_[tid].ptr).reset()
            (<Model>models_[tid].ptr).seed += rep
        (<Model>models_[tid].ptr).simulateNSteps(burninSamples)
        tid = threadid()
        #with gil: print("thread {}".format(tid))
        start = rep * nSamples

        for sample in range(nSamples):
            mu = mean((<Model>models_[tid].ptr).simulateNSteps(distSamples), nNodes, abs=1)
            while ((switch and mu >= threshold) and (not switch and mu < threshold)):
                # continue simulating until system state reached where intended avg mag level is reached
                mu = mean((<Model>models_[tid].ptr).simulateNSteps(1), nNodes, abs=1)
            snapshots[start + sample] = (<Model>models_[tid].ptr)._states
            #snapshots[start + sample] = encodeState((<Model>models_[rep].ptr)._states)
            #with gil:
                #code = encodeState((<Model>models_[rep].ptr)._states)
                #print("raw states: {}".format((<Model>models_[rep].ptr)._states.base))
                #print("decoded   : {}".format(decodeState(code, model._nNodes)))
                #print("difference: {}".format(np.array((<Model>models_[rep].ptr)._states.base) - np.array(decodeState(code, model._nNodes))))
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
        double[::1] cview_H
        int idx
        double len = <double> snapshots.shape[0]

    #print(s.base)

    H = (len - np.abs(H))/2. + np.abs(H)
    H = H/len

    tmp = 1-H
    H = - H * np.log2(H) - tmp * np.log2(np.where(tmp==0, 1, tmp)) # compute entropy for each node (use 0*log(0) = 0)

    cview_H = H

    #print("bin entropies: {}".format(H))

    return cview_H

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef double mutualInformationIDL(long[:,::1] snapshots, double[::1] binEntropies, int node1, int node2) nogil:
    cdef:
        int idx, nSamples = snapshots.shape[0]
        vector[long] states
        vector[int] jointDistr = vector[int](nSamples, 0)
        #unordered_map[int, int] jointStates
        double mi, jointEntropy

    #with gil:
    #    jointDistr = np.zeros(len, dtype=np.intc)

    for idx in range(nSamples):
        #states = decodeState(snapshots[idx], nNodes)
        jointDistr[idx] = snapshots[idx][node1] + snapshots[idx][node2]*2 # -3,-1,1,3 represent the 4 possible combinations of spins

    #jointStates[-3] = 0
    #jointStates[-1] = 1
    #jointStates[1] = 2
    #jointStates[3] = 3

    with gil:
        #print(binEntropies[node1], binEntropies[node2], entropy(jointDistr))
        jointEntropy = entropy(jointDistr)

    mi = binEntropies[node1] + binEntropies[node2] - jointEntropy
    return mi



cpdef double entropy(vector[int] samples):
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
cpdef double[::1] MIAtDist(Model model, long[:,::1] snapshots, double[::1] entropies, int node, vector[int] neighbours) nogil:

    cdef:
        #int[::1] neighbours
        #vector[int] neighbours
        int idx, n = neighbours.size() #.shape[0]
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
            out[idx] = mutualInformationIDL(snapshots, entropies, node, neighbours[idx])

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
cpdef np.ndarray magTimeSeries(Model model, int burninSamples, int nSamples, int abs=0):

    return _magTimeSeries(model, burninSamples, nSamples, abs).base

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef double[::1] _magTimeSeries(Model model, int burninSamples, int nSamples, int abs=0):

    cdef:
        double[::1] mags = np.zeros(nSamples)
        int sample

    #model.reset()
    model.simulateNSteps(burninSamples)

    for sample in range(nSamples):
        mags[sample] = mean(model.simulateNSteps(1), model._nNodes, abs)

    return mags

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple runMI(Model model, int repeats, int burninSamples, int nSamples, int distSamples, np.ndarray nodes, int distMax, double magThreshold=0, str targetDirectory=None):
    #states = model._states
    #code = encodeState(states)
    #print(code)
    #dec_states = decodeState(code, model._nNodes)
    #print("original  : {}".format(states.base))
    #print("decoded   : {}".format(dec_states))
    #print("difference: {}".format(model._states.base - decodeState(code, model._nNodes)))

    cdef:
        int[::1] cv_nodes = nodes
        long[:,::1] snapshots #= collectSnapshots(model, repeats, burninSamples, nSamples, distSamples)
        double[::1] entropies #= binaryEntropies(snapshots, model._nNodes)
        int n, d, nNodes = nodes.shape[0]
        double[:,:,::1] MI = np.zeros((nNodes, distMax, model._nNodes))
        int nThreads = mp.cpu_count()
        unordered_map[int, vector[int]] allNeighbours
        int[::1] neighbours

    if magThreshold == 0:
        snapshots = collectSnapshots(model, repeats, burninSamples, nSamples, distSamples)
    elif magThreshold > 0:
        snapshots = collectSnapshotsMagThreshold(model, repeats, burninSamples, nSamples, distSamples, switch=0, threshold=magThreshold)
    else:
        snapshots = collectSnapshotsMagThreshold(model, repeats, burninSamples, nSamples, distSamples, switch=1, threshold=np.abs(magThreshold))

    if targetDirectory is not None: np.save(f'{targetDirectory}/snapshots_T={model.t}_{time.time()}.npy', snapshots)

    entropies = binaryEntropies(snapshots)

    for n in prange(nNodes, nogil = True, \
                         schedule = 'dynamic', num_threads = nThreads):

        with gil: allNeighbours = model.neighboursAtDist(n, distMax)

        for d in range(distMax):
            #with gil: neighbours = allNeighbours[d+1]
            MI[n][d] = MIAtDist(model, snapshots, entropies, cv_nodes[n], allNeighbours[d+1])

    degrees = [model.graph.degree(model.rmapping[n]) for n in nodes]

    return MI.base, degrees

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
                      int n             = int(1e3),\
                      int burninSamples = 100):
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
    if n < 1e4:
        return model.matchMagnetization(temps, n, burninSamples)

    # otherwise parallel is faster
    cdef:
        #list modelsPy  = []
        vector[PyObjectHolder] models_
        Model tmp
        long idx, nNodes = model._nNodes
        int t, step, start, tid
        int nThreads = mp.cpu_count()
        int nTemps = temps.shape[0]
        int totalSteps = n + burninSamples
        double sum, sum_square, m
        vector[double] mag_sum
        np.ndarray betas = np.array([1 / t if t != 0 else np.inf for t in temps])
        double[::1] cview_betas = betas
        np.ndarray results = np.zeros((2, nTemps))
        double[:,::1] cview_results = results
        #long[:, ::1] r = model.sampleNodes( n * nTemps) # TODO sample in smaller blocks to avoid memory error!



    # threadsafe model access
    for t in range(nThreads):
        tmp = copy.deepcopy(model)
        #tmp.reset()
        #tmp.seed += t # enforce different seeds
        #tmp.t = temps[t]
        #modelsPy.append(tmp)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    #tid = threadid()
    pbar = tqdm(total = nTemps)
    for t in prange(nTemps, nogil = True, \
                         schedule = 'static', num_threads = nThreads): # simulate with different temps in parallel
        tid = threadid()
        with gil:
            (<Model>models_[tid].ptr).reset()
            (<Model>models_[tid].ptr).seed += t
            (<Model>models_[tid].ptr).t = temps[t]
        # simulate until equilibrium reached
        #start = t * totalSteps
        #for step in range(burninSamples):
        #    idx = start + step
        #    (<Model>models_[t].ptr)._updateState(r[idx])
        (<Model>models_[tid].ptr).simulateNSteps(burninSamples)

        sum = 0
        sum_square = 0
        # collect magnetizations
        #for step in range(n):
        #    idx = start + burninSamples + step
        #    m = mean((<Model>models_[t].ptr)._updateState(r[idx]), nNodes)
        #    sum = sum + m
        #    sum_square = sum_square + (m*m)

        mag_sum = simulateGetMeanMag((<Model>models_[tid].ptr), n)

        m = mag_sum[0] / n
        cview_results[0][t] = m if m > 0 else -m
        cview_results[1][t] = ((mag_sum[1] / n) - (m * m)) * cview_betas[t] #* (<Model> models_[t].ptr).beta

        with gil:
            pbar.update(1)

    return results

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
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
                      int stepSizeBurnin = 100,\
                      int nStepsRegress = int(1e3),\
                      double threshold = 0.05,\
                      int nStepsCorr = int(1e4)):

    cdef:
        vector[PyObjectHolder] models_
        Model tmp
        int s, tid, nNodes = model._nNodes
        int nThreads = mp.cpu_count()
        #vector[double] mag_sum

        double[::1] mags = np.zeros(stepSizeBurnin)
        np.ndarray allMags  # for regression
        int lag, h, counter # tmp var and counter
        double beta        # slope value
        np.ndarray magSeries, autocorr, x = np.arange(nStepsRegress)# for regression
        double slope, intercept, r_value, p_value, std_err, mixingTime, corrTime

    counter = 0
    allMags = np.array(mean(model.states, nNodes, abs=1))
    #initial_conf = model.states

    # simulate until mag has stabilized
    # remember mixing time needed
    while True: #counter < maxStepsBurnin:
        #for sample in range(stepSizeBurnin):
        #    mags[sample] = mean(model.simulateNSteps(1), model._nNodes, abs=1)

        mags = _magTimeSeries(model, 0, stepSizeBurnin, abs=1)

        allMags = np.hstack((allMags, mags.base))
        if counter >= nStepsRegress :
            # do linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x, allMags[-nStepsRegress:])
            if 1 - p_value < threshold: # p-value of test for slope=0
                #print(slope, intercept, p_value, std_err)
                break
        counter = counter + stepSizeBurnin
    #else:
    mixingTime = counter - nStepsRegress
    #print('Number of bunin samples needed {0}\n\n'.format(mixingTime))
    #print(f'absolute mean magnetization last sample {y[-1]}')

    # measure correlation time (autocorrelation for varying lags)
    #model.states = initial_conf
    magSeries = magTimeSeries(model, 0, nStepsCorr)

    autocorr = autocorrelation(magSeries)

    return allMags, mixingTime, autocorr



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef vector[double] simulateGetMeanMag(Model model, int nSamples = int(1e2)) nogil:
    cdef:
        long[:, ::1] r = model.sampleNodes(nSamples)
        int step
        double sum, sum_square
        vector[double] out = vector[double](2, 0)

    sum = 0
    sum_square = 0
    # collect magnetizations
    for step in range(nSamples):
        m = mean(model._updateState(r[step]), model._nNodes)
        sum = sum + m
        sum_square = sum_square + (m*m)

    out[0] = sum
    out[1] = sum_square

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

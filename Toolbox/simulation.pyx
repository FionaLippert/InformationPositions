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
from cython.parallel cimport parallel, prange, threadid
import multiprocessing as mp
import copy
from cpython.ref cimport PyObject
from scipy.stats import linregress
from scipy.signal import correlate
import scipy
from scipy import stats, special
import itertools
from tqdm import tqdm
import warnings

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
cdef long encodeStateToAvg(long[::1] states, vector[long] nodes, double[::1] bins) nogil:
    """
    Maps states of given nodes to their binned average magnetization

    Input:
      :states: vector of system states
      :nodes: vector of node indices
      :bins: vector of upper bin bounderies

    Output:
      :i: index of bin the average magnetization falls into
    """
    cdef:
        long N = nodes.size(), nBins = bins.shape[0]
        double avg = 0
        long i

    for i in range(N):
        avg += states[nodes[i]]

    avg /= N

    for i in range(nBins):
        if avg <= bins[i]:
            break

    return i

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef dict getSnapShots(Model model, int nSamples, int steps = 1,\
                   int burninSteps = int(1e3), int nThreads = -1):
    """
    Determines the state distribution of the :model: in parallel. The model is reset
    to random state and simulated for :step: + :burninSteps: steps after which
    a single sample is drawn and added to the output :snapshots:

    Input:
        :model: a model according to :Models.models:
        :nSamples: number of state samples to draw
        :step: number of steps between samples
    Returns:
        :snapshots: dict containing the idx of the state as keys, and probability as values
    """
    cdef:
        # unordered_map[int, double] snapshots
        # unordered_map[int, vector[int]] msnapshots
        dict snapshots = {}
        int step, sample
        int N          = nSamples * steps
        # long[:, ::1] r = model.sampleNodes(N)
        double Z       = <double> nSamples
        int idx # deprc?
        unordered_map[int, vector[int]].iterator got
        double past    = timer()
        list modelsPy  = []
        vector[PyObjectHolder] models_
        Model tmp
        cdef int tid,

    nThreads = mp.cpu_count() if nThreads == -1 else nThreads
    # threadsafe model access; can be reduces to n_threads
    for sample in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.reset()
        # TODO: remove this
        try:
            tmp.burnin(burninSteps)
        except:
            pass
        tmp.seed += sample # enforce different seeds
        # modelsPy.append(tmp)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    # init rng buffers
    cdef int sampleSize = model.nNodes if model.updateType != 'single' else 1

    cdef long[:, :, ::1] r    = np.ndarray((nThreads, steps, sampleSize), \
                                           dtype = long)
    # cdef cdef vector[vector[vector[int][sampleSize]][nTrial]][nThreads] r    = 0
    # cdef long[:, :, ::1] r = np.ndarray((nThreds, steps, sampleSize), dtype = long)
    cdef PyObject *modelptr
    pbar = tqdm(total = nSamples)
    cdef tuple state
    cdef int counter = 0
    for sample in prange(nSamples, nogil = True, \
                         schedule = 'static', num_threads = nThreads):

        tid      = threadid()
        modelptr = models_[tid].ptr
        r[tid] = (<Model> modelptr).sampleNodes(steps)
        # r[sample] = (<Model> models_[sample].ptr).sampleNodes(steps)
        # perform n steps
        for step in range(steps):
            (<Model> modelptr)._updateState(\
                                                    r[tid, step]
                                                        )
        with gil:
            state = tuple((<Model> modelptr)._states.base)
            snapshots[state] = snapshots.get(state, 0) + 1 / Z
            (<Model> modelptr).reset()
            pbar.update(1)
    print('done')
    pbar.close()
    print(f'Found {len(snapshots)} states')
    print(f"Delta = {timer() - past: .2f} sec")
    return snapshots


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef unordered_map[string, double] getSystemSnapshots(Model model, long[::1] nodes, \
              long nSnapshots = int(1e3), long burninSteps = int(1e3), \
              long distSamples = int(1e3), int threads = -1, int initStateIdx = -1):
    """
    Simulates the system in equilibrium, and extract snapshots of the given nodes.
    The number of parallel MCMC simulations corresponds to the requested number of threads

    Input:
        :model: a model according to :Models.models:
        :nodes: nodes of interest
        :nSnapshots: number of snapshots to be extracted
        :burninSteps: number of initial simulation steps to discard in order to reach equilibrium
        :distSamples: number of simulation steps between two consecutive snapshots
        :threads: number of threads to be used. Corresponds to the number of concurrent
                  Monte Carlo chains from which snapshots are extracted
        :initStateIdx: index to the state in :model.agentStates: to which all
                       system components should be initialized. If -1, the model
                      is initialized with a random system state.

    Output:
        :snapshots: mapping from system states (encoded as byte string) to
                    observation frequencies
    """
    cdef:
        unordered_map[string, double] snapshots
        int numStates = model.agentStates.size
        long i, s, rep, sample
        long[:,::1] initialState
        vector[long] nodesIdx = [model.mapping[n] for n in nodes]
        string state
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads


    # initialize thread-safe models. nThread MC chains will be run in parallel.
    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.resetAllToAgentState(initStateIdx, rep)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    # burnin samples
    for rep in prange(nThreads, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        (<Model>modelptr).simulateNSteps(burninSteps)

    pbar = tqdm(total = nSnapshots)
    for rep in prange(nSnapshots, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        (<Model>modelptr).simulateNSteps(distSamples)

        with gil:
            state = (<Model> modelptr).encodeStateToString(nodesIdx)
            snapshots[state] += 1 # each index corresponds to one system state, the array contains the count of each state
        with gil: pbar.update(1)

    return snapshots


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef vector[unordered_map[string, unordered_map[string, double]]] getSystemSnapshotsSets(Model model,
              vector[vector[long]] systemNodesG, vector[vector[long]] condNodesG, \
              long nSnapshots = int(1e3), long burninSteps = int(1e3), \
              long distSamples = int(1e3), int threads = -1, int initStateIdx = -1):
    """
    Simulates the system in equilibrium. For all node sets in :systemNodesG:,
    the joint distribution over these nodes, conditioned on the respective node set in :condNodesG:
    is estimated from :nSnapshots: system states.
    The number of parallel MCMC simulations corresponds to the requested number of threads

    Input:
        :model: a model according to :Models.models:
        :systemNodesG: vector of vectors containing IDs of nodes for which the
                       conditional joint distribution should be estimated.
                       The IDs are not internal model IDs but refer
                       to the original graph object given as input to the :model:
        :condNodesG: vector of vectors containing IDs of nodes to be conditioned on.
                     The IDs are not internal model IDs but refer
                     to the original graph object given as input to the :model:
        :nSnapshots: number of snapshots to be extracted
        :burninSteps: number of initial simulation steps to discard in order to reach equilibrium
        :distSamples: number of simulation steps between two consecutive snapshots
        :threads: number of threads to be used. Corresponds to the number of concurrent
                  Monte Carlo chains from which snapshots are extracted
        :initStateIdx: index to the state in :model.agentStates: to which all
                       system components should be initialized. If -1, the model
                      is initialized with a random system state.

    Output:
        :snapshots: mapping from system states (encoded as byte string) to
                    observation frequencies
    """
    cdef:
        long n, i, rep, sample, set, numSets = condNodesG.size()
        vector[long] arr
        vector[unordered_map[string, unordered_map[string, double]]]  snapshots =
                  vector[unordered_map[string, unordered_map[string, double]]](numSets)
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

    # burnin samples
    for rep in prange(nThreads, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        (<Model>modelptr).simulateNSteps(burninSteps)

    pbar = tqdm(total = nSnapshots)
    for rep in prange(nSnapshots, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        (<Model>modelptr).simulateNSteps(distSamples)

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
cpdef tuple getmagFreqs_switch(Model model, long[::1] nodesG, \
              unordered_map[long, unordered_map[long, vector[long]]] neighboursG, \
              long nSteps      = 1000, \
              int maxDist = 1,\
              long burninSteps  = 1000, \
              double threshold    = 0.05, \
              long nBins = 100, \
              int threads = -1):
    """
    Simulates the system in equilibrium. Detects switches between positive and
    negative system magnetization and stores frequencies of system states for
    positive, negative and close to zero magnetization separately.
    """
    cdef:
        int tid, nThreads = mp.cpu_count() if threads == -1 else threads
        long nNodes = model._nNodes
        long nNodesG = nodesG.shape[0]
        long[:,:,::1] states = np.zeros((nThreads, nSteps, nNodes), int)
        double[:,::1] mags = np.zeros((nThreads, nSteps))

        long node
        long[::1] nodesIdx = np.zeros(nNodesG, 'int')

        long[:,:,:,:,::1] magFreqsPos = np.zeros((nThreads, nNodesG, maxDist, model.agentStates.shape[0], nBins), int)
        long[:,:,:,:,::1] magFreqsNeg = np.zeros((nThreads, nNodesG, maxDist, model.agentStates.shape[0], nBins), int)
        long[:,:,:,:,::1] magFreqsSwitch = np.zeros((nThreads, nNodesG, maxDist, model.agentStates.shape[0], nBins), int)
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
        tmp.resetAllToAgentState(-1, rep) # initialize at random
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    # burnin samples
    for rep in prange(nThreads, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        (<Model>modelptr).simulateNSteps(burninSteps)


    for rep in prange(nSteps, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        states[tid] = (<Model>modelptr)._simulate(nSteps)

        for step in range(nSteps):
            m = mean(states[tid][step], nNodes, abs=0)
            mAbs = m if m > 0 else -m
            mags[tid][step] = m

            for n in range(nNodesG):
                nodeSpin = idxer[states[tid][step][nodesIdx[n]]]
                for d in range(maxDist):
                    avg = encodeStateToAvg(states[tid][step], neighboursIdx[n][d+1], bins)
                    if mAbs > threshold:
                        if m > 0:
                            magFreqsPos[tid][n][d][nodeSpin][avg] +=1
                            Z[0] += 1
                        else:
                            magFreqsNeg[tid][n][d][nodeSpin][avg] +=1
                            Z[1] += 1
                    else:
                        magFreqsSwitch[tid][n][d][nodeSpin][avg] +=1
                        Z[2] += 1

    return magFreqsPos.base, magFreqsNeg.base, magFreqsSwitch.base, Z.base, mags.base




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getSnapshotsPerDist(Model model, long[::1] nodesG, \
              long nSnapshots = int(1e3), \
              long burninSteps = int(1e3), int maxDist = 1, int threads = -1, \
              int initStateIdx = -1):
    """
    Simulates the system in equilibrium. For each node in :nodesG:,
    take snapshots of its neighbourhood shells up to distance :maxDist: and count
    the number of observations per neighbourhood state.

    Input:
        :model: a model according to :Models.models:
        :nodesG: vector of IDs of the node of interest. The IDs are not the internal
                 model indices but refer to the original graph object given as
                 input to the :model:
        :nSnapshots: number of snapshots to be extracted
        :burninSteps: number of initial simulation steps to discard in order to reach equilibrium
        :maxDist: maximum distance for neighbourhood shells
        :threads: number of threads with concurrent simulations to be used
        :initStateIdx: index to the state in :model.agentStates: to which all
                       system components should be initialized. If -1, the model
                      is initialized with a random system state.

    Output:
        :snapshots: vector containing the mappings from neighbourhood shell states
                    (encoded as byte string) to their probabilities, for each
                    distance respectively
        :neighboursG: vector containing the mappings from distance d to the respective neighbourhood
                      shell, for each node in :nodesG: respectively. Neighbourhood shells are
                      given as vectors of node IDs that refer to the original graph object.
    """
    cdef:
        long nNodes = nodesG.shape[0]
        long[::1] nodesIdx = np.zeros(nNodes, 'int')

        vector[vector[unordered_map[string, double]]] snapshots = vector[vector[unordered_map[string, double]]](nNodes)
        vector[unordered_map[long, vector[long]]] neighboursIdx = vector[unordered_map[long, vector[long]]](nNodes)
        vector[unordered_map[long, vector[long]]] neighboursG = vector[unordered_map[long, vector[long]]](nNodes)

        long d, i, b, sample, rep, n
        double part = 1 / (<double> nSnapshots)
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


    for sample in prange(nSnapshots, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        (<Model>modelptr).simulateNSteps(burninSteps)

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
cpdef tuple getMagsPerDist(Model model, long[::1] nodesG, \
              unordered_map[long, unordered_map[long, vector[long]]] neighboursG, \
              long nSamples = int(1e3), \
              long burninSteps = int(1e3), long distSamples=100, \
              int maxDist = 1, long nBins=10, int threads = -1, \
              int initStateIdx = -1, int getFullSnapshots = 0):

    """
    Simulates the system in equilibrium. For each node in :nodesG:,
    draw samples of the binned magnetization of neighbourhood shells up to distance :maxDist: and count
    the number of observations per neighbourhood state.

    Input:
        :model: a model according to :Models.models:
        :nodesG: vector of IDs of the node of interest. The IDs are not the internal
                 model indices but refer to the original graph object given as
                 input to the :model:
        :nSamples: number of samples to be drawn
        :burninSteps: number of initial simulation steps to discard in order to reach equilibrium
        :distSamples: number of simulation steps between two consecutive samples
        :maxDist: maximum distance for neighbourhood shells
        :nBins: number of bins
        :threads: number of threads with concurrent simulations to be used
        :initStateIdx: index to the state in :model.agentStates: to which all
                       system components should be initialized. If -1, the model
                       is initialized with a random system state.
        :getFullSnapshots: if True, also take snapshots of the entire system to
                           be able to determine pairwise MI and correlation

    Output:
        :magFreqs: array containing frequencies of the binned neighbourhood shell magnetization
                   for all nodes and distances, conditioned on the state of the respective node of interest
        :systemMagFreqs: array containing frequencies of the binned system magnetization, conditioned on the state of
                         each of the nodes in :nodesG:
        :fullSnapshots: array containing the states of all nodes of the :model:, for each of the :nSamples:
                        Only returned if getFullSnapshots==True
    """
    cdef:
        int nThreads = mp.cpu_count() if threads == -1 else threads

        long node, nNodes = nodesG.shape[0]
        long[::1] nodesIdx = np.zeros(nNodes, 'int')

        long[:,:,:,:,::1] magFreqs = np.zeros((nThreads, nNodes, maxDist, model.agentStates.shape[0], nBins), int)
        long[:,:,:,::1] systemMagFreqs = np.zeros((nThreads, nNodes, model.agentStates.shape[0], nBins), int)
        unordered_map[int, int] idxer
        vector[unordered_map[long, vector[long]]] neighboursIdx = vector[unordered_map[long, vector[long]]](nNodes)
        vector[long] allNodes = list(model.mapping.values())

        long d, i, b, sample, rep, n
        string state
        double past    = timer()
        PyObject *modelptr
        vector[PyObjectHolder] models_
        int tid, nodeSpin, s, mag, systemMag


        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins + 1)[1:] # values represent upper bounds for bins
        long[:,::1] fullSnapshots

    if getFullSnapshots: fullSnapshots = np.zeros((nSamples, model._nNodes), int)

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx


    for n in range(nNodes):
        node = nodesG[n]
        nodesIdx[n] = model.mapping[node]
        for d in range(maxDist):
            neighboursIdx[n][d+1] = [model.mapping[neighbour] for neighbour in neighboursG[node][d+1]]

    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += rep
        tmp.resetAllToAgentState(initStateIdx, rep)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    # burnin samples
    for rep in prange(nThreads, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr
        (<Model>modelptr).simulateNSteps(burninSteps)

    pbar = tqdm(total = nSamples)
    for rep in prange(nSamples, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        modelptr = models_[tid].ptr

        (<Model>modelptr).simulateNSteps(distSamples)
        systemMag = (<Model> modelptr).encodeStateToAvg(allNodes, bins)

        for n in range(nNodes):
            nodeSpin = idxer[(<Model> modelptr)._states[nodesIdx[n]]]
            for d in range(maxDist):
                mag = (<Model> modelptr).encodeStateToAvg(neighboursIdx[n][d+1], bins)
                magFreqs[tid][n][d][nodeSpin][mag] +=1

            systemMagFreqs[tid][n][nodeSpin][systemMag] += 1

        if getFullSnapshots:
            # TODO: use string encoding?
            fullSnapshots[rep] = (<Model>modelptr)._states

        with gil: pbar.update(1)

    if getFullSnapshots:
        return magFreqs.base, systemMagFreqs.base, fullSnapshots.base
    else:
        return magFreqs.base, systemMagFreqs.base




cpdef np.ndarray monteCarloFixedNeighbours(Model model, string snapshot, long nodeIdx, \
               vector[long] neighboursIdx, long nTrials, long burninSteps, \
               long nSamples = 10, long distSamples = 10):
      """
      python wrapper
      """
      return _monteCarloFixedNeighbours(model, snapshot, nodeIdx, \
                     neighboursIdx, nTrials, burninSteps, \
                     nSamples, distSamples).base



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[::1] _monteCarloFixedNeighbours(Model model, string snapshot, long nodeG, \
               vector[long] neighboursG, long nTrials, long burninSteps = int(1e3), \
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
        if initStateIdx == -2:
            with gil:
                i = np.mod(trial, model.agentStates.shape[0])
                initialState = np.ones(model._nNodes, int) * model.agentStates[i]
        elif initStateIdx == -1:
            with gil: initialState = np.random.choice(model.agentStates, size = model._nNodes)
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
        model.simulateNSteps(burninSteps) # go to equilibrium

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
cdef long[:,::1] _monteCarloFixedNeighboursStates(Model model, string snapshot, long nodeIdx, \
               vector[long] neighboursIdx, long nTrials, long burninSteps = int(1e3), \
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

        if initStateIdx == -2:
            #with gil: initialState = np.random.choice(model.agentStates, size = model._nNodes)
            with gil:
                i = np.mod(trial, model.agentStates.shape[0])
                initialState = np.ones(model._nNodes, int) * model.agentStates[i]
        elif initStateIdx == -1:
            with gil: initialState = np.random.choice(model.agentStates, size = model._nNodes)
        else:
            with gil: initialState = np.ones(model._nNodes, int) * model.agentStates[initStateIdx]

        for idx in range(length):
            n = neighboursIdx[idx]
            initialState[n] = decodedStates[idx]


        with gil: model.setStates(initialState)
        with gil: model.seed += 1

        model.simulateNSteps(burninSteps) # go to equilibrium

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
              long nTrials, long burninSteps, long nSamples, long distSamples, int threads = -1, int initStateIdx = -1, \
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
                          keys[idx], nodeG, allNeighboursG[dist], nTrials, burninSteps, nSamples, distSamples, initStateIdx)

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
                        keys[idx], nodeIdx, neighboursIdx, nTrials, burninSteps, nSamples, distSamples, initStateIdx)

            with gil: pbar.update(1)

        return snapshotsDict, states.base




@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef np.ndarray magTimeSeries(Model model, long burninSteps, \
                                long nSamples, int abs=0):
    """
    python wrapper
    """
    return _magTimeSeries(model, burninSteps, nSamples, abs).base

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cdef double[::1] _magTimeSeries(Model model, long burninSteps, \
                                long nSamples, int abs=0):
    """
    simulate the system in equilibrium,
    determine the system magnetization at each time step
    """
    cdef:
        double[::1] mags = np.zeros(nSamples)
        long sample

    model.simulateNSteps(burninSteps)

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
cpdef np.ndarray magnetizationParallel(Model model,
                      np.ndarray temps, long n = int(1e3),
                      long burninSteps = 100, int threads = -1):
    """
    Computes the magnetization as a function of temperatures
    Input:
          :model: the model to use for simulations
          :temps: a range of temperatures
          :n:     number of samples to simulate for
          :burninSteps: number of samples to throw away before sampling
    Output:
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
        long totalSteps = n + burninSteps
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
            (<Model>modelptr).resetAllToAgentState(1)

        # simulate until equilibrium reached
        (<Model>modelptr).simulateNSteps(burninSteps)

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
        double slope, intercept, r_value, p_value, std_err

    counter = 0
    allMags = np.array(mean(model.states, nNodes, abs=1))

    # simulate until mag has stabilized
    # remember mixing time needed
    if checkMixing:
        while True:
            mags = _magTimeSeries(model, 0, burninSteps, abs=1)
            allMags = np.hstack((allMags, mags.base))
            if counter >= nStepsRegress :
                # linear regression
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
        double intercept, meanMag = 0
        long idx
        np.ndarray tmp, mags, autocorr, initialConfigs = np.linspace(0.5, 1, nInitialConfigs)
        double prob, t
        long corrTime, mixingTimeMax = 0, corrTimeMax = 0
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
        if tmp.size > 0:
            corrTime = tmp[0]
        else:
            corrTime = nStepsCorr # set to maximum
            warnings.warn(f'autocorrelation did not drop below threshold thr={thresholdCorr}', Warning)
        #corrTime = tmp[0] if tmp.size > 0 else np.inf
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

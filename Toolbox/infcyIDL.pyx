# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Fiona Lippert'

# MODELS
from Models.models cimport Model
from Models.fastIsing cimport Ising

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


def checkDistribution():
    '''Warning statement'''
    from platform import platform
    if 'windows' in platform().lower():
        print('Warning: Windows detected. Please remember to respect the GIL'\
              ' when using multi-core functions')
checkDistribution()


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
cpdef tuple getSnapshotsPerDist2(Model model, long node, \
          long nSamples = int(1e2), long burninSamples = int(1e3), int maxDist = 1):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Take nSamples snapshots for each set of neighbours at distance d, ranging from d=1 to d=maxDist
    """
    cdef:
        vector[unordered_map[string, double]] snapshots = vector[unordered_map[string, double]](maxDist)
        long d, sample
        double part = 1/(<double> nSamples)
        string state
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count(), nodeSpin

        unordered_map[long, vector[long]] allNeighbours_G, allNeighbours_idx

    node = model.mapping[node] # map to internal index
    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)


    for tid in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    for sample in prange(nSamples, nogil = True, schedule = 'static', num_threads = nThreads):
        tid = threadid()
        with gil:
            (<Model>models_[tid].ptr).seed += sample # enforce different seeds
            (<Model>models_[tid].ptr).reset()

        (<Model>models_[tid].ptr).simulateNSteps(burninSamples)

        for d in range(maxDist):
            with gil: state = (<Model> models_[tid].ptr).encodeStateToString(allNeighbours_idx[d+1])
            snapshots[d][state] += part # each index corresponds to one system state, the array contains the probability of each state

    return snapshots, allNeighbours_G, allNeighbours_idx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple getJointSnapshotsPerDist(Model model, long node, \
          long nSamples = int(1e3), long burninSamples = int(1e3), int maxDist = 1, int nBins=10, double threshold = 0.05):
    """
    Extract snapshots from MC for large network, for which the decimal encoding causes overflows
    Take snapshots for each set of neighbours at distance d, ranging from d=1 to d=maxDist.
    Store snapshots for different spin states of the central node separately to estimate the joint pdf.
    Additionally, the discretized average magnetization of neighbours at distance d is sampled.
    Every nSamples, compute the KL-divergence between the previous sample distribution and the new one. Stop sampling if it drops below threshold.
    """
    cdef:
        vector[unordered_map[int, unordered_map[string, double]]] snapshots = vector[unordered_map[int, unordered_map[string, double]]](maxDist)
        vector[unordered_map[int, unordered_map[string, double]]] oldSnapshots = vector[unordered_map[int, unordered_map[string, double]]](maxDist)
        vector[unordered_map[int, unordered_map[int, double]]] avgSnapshots = vector[unordered_map[int, unordered_map[int, double]]](maxDist)
        vector[unordered_map[int, unordered_map[int, double]]] oldAvgSnapshots = vector[unordered_map[int, unordered_map[int, double]]](maxDist)


        long d, i, sample
        double Z       = 0
        string state
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count(), nodeSpin, s, avg
        np.ndarray KL = np.ones(maxDist)
        double KL_d
        double[::1] bins = np.linspace(np.min(model.agentStates), np.max(model.agentStates), nBins)

        unordered_map[long, vector[long]] allNeighbours_G, allNeighbours_idx

    node = model.mapping[node]
    allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(node, maxDist)

    for rep in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    while True:
        for sample in prange(nSamples, nogil = True, schedule = 'static', num_threads = nThreads):
            tid = threadid()
            with gil:
                (<Model>models_[tid].ptr).seed += sample # enforce different seeds
                (<Model>models_[tid].ptr).reset()

            (<Model>models_[tid].ptr).simulateNSteps(burninSamples)

            nodeSpin = (<Model> models_[tid].ptr)._states[node]
            for d in range(maxDist):
                with gil: state = (<Model> models_[tid].ptr).encodeStateToString(allNeighbours_idx[d+1])
                avg = (<Model> models_[tid].ptr).encodeStateToAvg(allNeighbours_idx[d+1], bins)
                snapshots[d][nodeSpin][state] += 1
                avgSnapshots[d][nodeSpin][avg] +=1

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

    return snapshots, avgSnapshots, Z



cpdef np.ndarray monteCarloFixedNeighbours(Model model, string snapshot, long node, \
               vector[long] neighbours, \
               long nSamples = 10, long distSamples = 10):

      return _monteCarloFixedNeighbours(model, snapshot, node, \
                     neighbours, \
                     nSamples, distSamples).base



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[::1] _monteCarloFixedNeighbours(Model model, string snapshot, long node, \
               vector[long] neighbours, \
               long nSamples = int(1e3), long distSamples = int(1e2)) nogil:


    cdef:
       double Z = <double> nSamples
       double part = 1/Z
       long idx, rep, sample, length = neighbours.size()
       long nodeState
       unordered_map[int, int] idxer
       double[::1] probCondArr
       long[::1] decodedStates

    for idx in range(model.agentStates.shape[0]):
        idxer[model.agentStates[idx]] = idx

    with gil:
        decodedStates = np.frombuffer(snapshot).astype(int)
        probCondArr = np.zeros(idxer.size())

    model._loadStatesFromString(decodedStates, neighbours) # keeps all other node states as they are
    #model.simulateNSteps(burninSamples)

    for sample in range(nSamples):
        model.simulateNSteps(distSamples)
        nodeState = model._states[node]
        probCondArr[idxer[nodeState]] += part

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple neighbourhoodMI(Model model, long node, vector[long] neighbours, unordered_map[string, double] snapshots, \
              long nSamples, long distSamples):
    cdef:
        Model tmp
        vector[PyObjectHolder] models_
        int tid, nThreads = mp.cpu_count()
        long n, sample, idx, nNeighbours = neighbours.size()
        string state
        double[::1] pY, pX
        double HXgiveny, HXgivenY = 0, MI = 0

    for idx in range(nNeighbours):
        n = neighbours[idx]
        neighbours[idx] = model.mapping[n] # map from graph to model index

    for tid in range(nThreads):
       tmp = copy.deepcopy(model)
       tmp.fixedNodes = neighbours
       models_.push_back(PyObjectHolder(<PyObject *> tmp))

    node = model.mapping[node]

    cdef:
        dict snapshotsDict = snapshots
        vector[string] keys = list(snapshotsDict.keys())
        double[::1] probs = np.array([snapshots[k] for k in keys])
        double[:,::1] container = np.zeros((keys.size(), model.agentStates.shape[0]))

    # determine conditional probabilities
    pbar = tqdm(total = keys.size())
    for idx in prange(keys.size(), nogil = True, schedule = 'dynamic', num_threads = nThreads):
        tid = threadid()
        container[idx] = _monteCarloFixedNeighbours((<Model>models_[tid].ptr), \
                        keys[idx], node, neighbours, nSamples, distSamples)
        HXgiveny = entropyFromProbs(container[idx])
        HXgivenY -= probs[idx] * HXgiveny

        with gil: pbar.update(1)

    # compute MI based on conditional probabilities
    pX = np.sum(np.multiply(probs.base, container.base.transpose()), axis=1)
    print(pX.base)
    MI = HXgivenY + entropyFromProbs(pX)

    print(f'MI= {MI}')

    return snapshotsDict, container.base, MI



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef long[:,::1] equilibriumSampling(Model model, long repeats, long burninSamples, long nSamples, long distSamples):
    cdef:
        vector[PyObjectHolder] models_
        Model tmp
        long start, sample, idx, step, rep
        int tid, nThreads = mp.cpu_count()
        long[:,::1] snapshots = np.zeros((repeats * nSamples, model._nNodes), int) #np.intc)


    for tid in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    pbar = tqdm(total = repeats) # init  progbar
    for rep in prange(repeats, nogil = True, schedule = 'dynamic', num_threads = nThreads):
        tid = threadid()
        with gil:
            (<Model>models_[tid].ptr).reset()
            (<Model>models_[tid].ptr).seed += rep # enforce different seeds
        (<Model>models_[tid].ptr).simulateNSteps(burninSamples)
        start = rep * nSamples

        for sample in range(nSamples):
            # raw system states are stored, because for large systems encoding of snapshots does not work (overflow)
            snapshots[start + sample] = (<Model>models_[tid].ptr).simulateNSteps(distSamples)

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
                      int switch=1, double threshold=0.05):
    cdef:
        vector[PyObjectHolder] models_
        Model tmp
        long start, sample = 0, idx, step, rep, nNodes = model._nNodes
        int tid, nThreads = mp.cpu_count()
        double mu
        long[:,::1] snapshots = np.zeros((repeats * nSamples, nNodes), int) #np.intc)

    for tid in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    pbar = tqdm(total = repeats) # init  progbar
    for rep in prange(repeats, nogil = True, schedule = 'dynamic', num_threads = nThreads):
        tid = threadid()
        with gil:
            (<Model>models_[tid].ptr).reset()
            (<Model>models_[tid].ptr).seed += rep
        (<Model>models_[tid].ptr).simulateNSteps(burninSamples)
        start = rep * nSamples

        for sample in range(nSamples):
            mu = mean((<Model>models_[tid].ptr).simulateNSteps(distSamples), nNodes, abs=1)
            while ((switch and mu >= threshold) and (not switch and mu < threshold)):
                # continue simulating until system state reached where intended avg mag level is reached
                mu = mean((<Model>models_[tid].ptr).simulateNSteps(1), nNodes, abs=1)
            snapshots[start + sample] = (<Model>models_[tid].ptr)._states

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
cpdef double mutualInformationIDL(long[:,::1] snapshots, double[::1] binEntropies, long node1, long node2) nogil:
    cdef:
        long idx, nSamples = snapshots.shape[0]
        vector[long] states
        vector[long] jointDistr = vector[long](nSamples, 0)
        double mi, jointEntropy


    for idx in range(nSamples):
        jointDistr[idx] = snapshots[idx][node1] + snapshots[idx][node2]*2 # -3,-1,1,3 represent the 4 possible combinations of spins

    with gil:
        jointEntropy = entropy(jointDistr)

    mi = binEntropies[node1] + binEntropies[node2] - jointEntropy
    return mi


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
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
              double[::1] entropies, long node, vector[long] neighbours) nogil:

    cdef:
        long idx, n = neighbours.size()
        double[::1] out

    with gil: # TODO possible without GIL?
        out = np.full(model._nNodes, np.nan)

    if n > 0:
        for idx in range(n):
            out[idx] = mutualInformationIDL(snapshots, entropies, node, neighbours[idx])

    return out


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

    model.simulateNSteps(burninSamples)

    for sample in range(nSamples):
        mags[sample] = mean(model.simulateNSteps(1), model._nNodes, abs)

    return mags


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef tuple runMI(Model model, long repeats, long burninSamples, long nSamples, \
                  long distSamples, np.ndarray nodes, int distMax, \
                  double magThreshold=0):

    cdef:
        long[::1] cv_nodes = nodes
        long[:,::1] snapshots
        double[::1] entropies
        long n, d, nNodes = nodes.shape[0]
        long[::1] nodesIdx = np.array([model.mapping[n] for n in nodes])
        double[:,:,::1] MI = np.zeros((nNodes, distMax, model._nNodes))
        int nThreads = mp.cpu_count()
        unordered_map[long, vector[long]] allNeighbours
        int[::1] neighbours

    # run multiple MC chains in parallel and sample snapshots
    if magThreshold == 0:
        snapshots = equilibriumSampling(model, repeats, burninSamples, nSamples, distSamples)
    elif magThreshold > 0:
        # only sample snapshots with abs avg mag larger than magThreshold
        snapshots = equilibriumSamplingMagThreshold(model, repeats, burninSamples, nSamples, distSamples, switch=0, threshold=magThreshold)
    else:
      # only sample snapshots with abs avg mag smaller than magThreshold
        snapshots = equilibriumSamplingMagThreshold(model, repeats, burninSamples, nSamples, distSamples, switch=1, threshold=np.abs(magThreshold))

    entropies = binaryEntropies(snapshots)

    for n in prange(nNodes, nogil = True, \
                         schedule = 'dynamic', num_threads = nThreads):

        with gil: _, allNeighbours = model.neighboursAtDist(nodesIdx[n], distMax)

        for d in range(distMax):
            MI[n][d] = MIAtDist(model, snapshots, entropies, nodesIdx[n], allNeighbours[d+1])

    degrees = [model.graph.degree(n) for n in nodes]

    return snapshots.base, MI.base, degrees



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
cpdef np.ndarray magnetizationParallel(Model model,\
                          np.ndarray temps  = np.logspace(-3, 2, 20),\
                      long n             = int(1e3),\
                      long burninSamples = 100):
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
        vector[PyObjectHolder] models_
        Model tmp
        long idx, nNodes = model._nNodes, t, step, start
        int tid, nThreads = mp.cpu_count()
        long nTemps = temps.shape[0]
        long totalSteps = n + burninSamples
        double m
        vector[double] mag_sum
        np.ndarray betas = np.array([1 / t if t != 0 else np.inf for t in temps])
        double[::1] cview_betas = betas
        np.ndarray results = np.zeros((2, nTemps))
        double[:,::1] cview_results = results


    # threadsafe model access
    for tid in range(nThreads):
        tmp = copy.deepcopy(model)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    pbar = tqdm(total = nTemps)
    for t in prange(nTemps, nogil = True, \
                         schedule = 'static', num_threads = nThreads): # simulate with different temps in parallel
        tid = threadid()
        with gil:
            (<Model>models_[tid].ptr).reset()
            (<Model>models_[tid].ptr).seed += t
            (<Model>models_[tid].ptr).t = temps[t]

        # simulate until equilibrium reached
        (<Model>models_[tid].ptr).simulateNSteps(burninSamples)

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
                      int checkMixing = 1):

    cdef:
        long s, nNodes = model._nNodes
        double[::1] mags = np.zeros(burninSteps)
        np.ndarray allMags  # for regression
        long lag, h, counter, mixingTime # tmp var and counter
        double beta        # slope value
        np.ndarray magSeries, autocorr, x = np.arange(nStepsRegress)# for regression
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

    # measure correlation time (autocorrelation for varying lags)
    magSeries = magTimeSeries(model, 0, nStepsCorr)
    autocorr = autocorrelation(magSeries)

    return allMags, mixingTime, autocorr


cpdef tuple determineCorrTime(Model model, \
              int nInitialConfigs = 10, \
              long burninSteps = 10, \
              long nStepsRegress = int(1e3), \
              double thresholdReg = 0.05, \
              long nStepsCorr = int(1e3), \
              double thresholdCorr = 0.05, \
              int checkMixing = 1):
    cdef:
        long mixingTime, mixingTimeMax = 0, idx, nNodes = model._nNodes
        double corrTime, prob, t, corrTimeMax = 0 # double because it might be infinity
        np.ndarray tmp, mags, autocorr, initialConfigs = np.linspace(0.5, 1, nInitialConfigs)

    for prob in tqdm(initialConfigs):
        model.states = np.random.choice([-1,1], size = nNodes, p=[prob, 1-prob])
        mags, mixingTime, autocorr = determineMixingTime(model,\
                              burninSteps,\
                              nStepsRegress,\
                              thresholdReg,\
                              nStepsCorr, \
                              checkMixing)
        if mixingTime > mixingTimeMax: mixingTimeMax = mixingTime
        tmp = np.where(np.abs(autocorr) < thresholdCorr)[0]
        corrTime = tmp[0] if tmp.size > 0 else np.inf
        if corrTime > corrTimeMax: corrTimeMax = corrTime

    return mixingTimeMax, corrTimeMax


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
        double sum, sum_square
        vector[double] out = vector[double](2,0)

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

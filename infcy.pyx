# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Casper van Elteren'
import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport parallel, prange, threadid
# cimport numpy as np
# cimport cython
import IO, plotting as plotz, networkx as nx, functools, itertools, platform, pickle,\
fastIsing, copy, time
# from pathos import multiprocessing as mp
import multiprocessing as mp
from models cimport Model
# import multiprocess as mp
# import pathos.multiprocessing as mp
from tqdm import tqdm   #progress bar
#from joblib import Parallel, delayed, Memory
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
# from libcpp.concurrent_unordered_map cimport concurrent_unordered_map
from libc.stdio cimport printf
import ctypes
#
# # TODO: the numpy approach should be re-written in a dictionary only approach in order to prevent memory issues;
# # the general outline would be to yield the results and immediately bin them accordingly and write state to disk
# from models cimport Model
# # array = functools.partial(np.array, dtype = np.float16) # tmp hack
# from libc.stdio cimport printf
# cdef int _CORE = 1  # for imap
# INT16 = np.int16
def checkDistribution():
    '''Warning statement'''
    from platform import platform
    if 'windows' in platform().lower():
        print('Warning: Windows detected. Please remember to respect the GIL'\
              ' when using multi-core functions')
checkDistribution() # print it only once


# cdef int encodeState(long[::1] state) :
#     cdef int i, N = state.shape[0]
#     cdef int out = 1
#     for i in range(N):
#         if state[i] != -1:
#             out *=  2 ** i
#     return out
# def encodeState(state, nStates):
#     return int(''.join(format(1 if i == 1 else 0, f'0{nStates - 1}b') for i in state), 2)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int encodeState(long[::1] state):

    cdef int binNum = 1
    cdef int N = state.shape[0]
    cdef int i
    cdef int dec = 0
    for i in range(N):
        if state[i] == 1:
            dec += binNum
        binNum *= 2
    return dec

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef vector[long] decodeState(int dec, int N):
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




import time
import faulthandler
faulthandler.enable()
from cython.operator cimport dereference as deref, preincrement as pre

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dict getSnapShots(Model model, int nSamples, int step = 1,\
                   int burninSamples = int(1e3)):
    # start sampling
    cdef unordered_map[int, double] snapshots
    # cdef dict snapshots = {}
    cdef int i
    cdef int N = nSamples * step
    cdef long[:, ::1] r = model.sampleNodes(N )
    cdef double Z = <double> nSamples
    cdef int idx
    cdef double past = time.process_time()
    pbar = tqdm(total = nSamples)
    for i in range(N):
        if i % step == 0:
            idx             = encodeState(model._states)
            snapshots[idx] += 1 / Z
            # snapshots[idx] = snapshots.get(idx, 0) + 1 / Z
            pbar.update(1)
        model._updateState(r[i])
    pbar.close()
    print(f'Found {len(snapshots)} states')
    print(f'Delta = {time.process_time() - past}')
    return snapshots
@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dict monteCarlo(\
               Model model, dict snapshots,
               int deltas = 10,  int repeats = 11,
               ):
    cdef dict conditional = {}
    # cdef unordered_map[int, double *] conditional
    cdef str nudgeMode = model.__nudgeType
    cdef double[:] copyNudge = model._nudges.copy() # store nudges already there
    # for k in range(repeats):
    cdef long[:, ::1] s = np.array([decodeState(q, model._nNodes) for q in snapshots], ndmin = 2)
    # print(np.unique(s, 1))

    # print(s.base)
    cdef int N = s.shape[0], n
    cdef long[:, ::1] r  # = model.sampleNodes(deltas + 1)

    # loop declarations
    cdef double Z = <double> repeats

    cdef int k, delta, node, statei
    half  = deltas // 2
    cdef bint reset = True
    cdef long[::1] agentStates = model.agentStates
    cdef long  idx
    cdef int jdx
    # cdef long[::1] start = model._states
    cdef int kdx
    pbar = tqdm(total = N)
    print('Starting loops')
    # TODO i hate cython for its for loops
    # cdef double[:, :, :, ::1] out = np.zeros((N, deltas + 1, model._nNodes, model._nStates), dtype = float)
    # cdef double[::1] out

    # a = [np.zeros( (deltas + 1) * model._nNodes * model._nStates, dtype = float).copy() for i in range(N)]
    # cdef double[::1] out
    cdef counter
    for n in range(N):
        kdx = encodeState(s[n])
        # printf('%p \n',&out[0])
        out = np.zeros((deltas + 1) * model._nNodes * model._nStates)
        # print(id(out))
        for k in range(repeats):
            for node in range(model._nNodes):
                model._states[node] = s[n, node]
                model._nudges[node] = copyNudge[node]
            # reset simulation
            reset = True
            r       = model.sampleNodes(deltas + 1)
            counter = 0
            for delta in range(deltas + 1):
                # bin data
                for node in range(model._nNodes):
                    for statei in range(model._nStates):
                        if model._states[node] == agentStates[statei]:
                            out[counter] += 1 / Z
                        counter += 1
                # update
                model._updateState(r[delta])
                # turn-off
                if reset:
                    if nudgeMode == 'pulse' or \
                    nudgeMode == 'constant' and delta >= half:

                        model._nudges[:] = 0
                        reset            = False
        conditional[kdx] = out.reshape((deltas + 1, model._nNodes, model._nStates)).copy()
        # print(out.reshape((deltas + 1, model._nNodes, model._nStates)))
        # conditional[kdx] = &out[0]# replace this with something that can hold the correct markers
        pbar.update(1)
    # cdef unordered_map[int, double*].iterator start = conditional.begin()
    # cdef unordered_map[int, double*].iterator end   = conditional.end()
    # cdef tmp = {}
    # cdef shape =  (deltas + 1, model._nNodes, model._nStates)
    # cdef double[::1] ar = np.zeros((deltas + 1) * model._nNodes * model._nStates, dtype = float)
    # cdef double* ptr
    # cdef int zz
    # while start != end:
    #     ptr = deref(start).second
    #     zz = deref(start).first
    #     for k in range(ar.shape[0]):
    #         ar[k] = deref(ptr)
    #         pre(ptr)
    #     print(ar.base)
    #     tmp[zz] = np.reshape(ar.base, shape)
    #     pre(start)
    # # print(np.reshape(ar.base, shape))
    # print(np.all(tmp[zz] == out.base.reshape(shape)))
    pbar.close()
    tmp = conditional
    # print(f"Delta = {time.process_time() - past}")
    return tmp
# cdef parallWrap(idx)
@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef mutualInformation(dict conditional, int deltas, \
                          dict snapshots, Model model):
    '''
    Returns the node distribution and the mutual information decay
    '''
    cdef  px = np.zeros((deltas + 1, model._nNodes, model._nStates))
    cdef  H     = np.zeros((deltas + 1, model._nNodes))
    print(' > ' ,snapshots.keys())
    for key, p in conditional.items():
        # p    = np.asarray(p)
        H   -= np.nansum(p * np.log2(p), -1) * snapshots[key]
        px  += p  * snapshots[key] # update node distribution
    H += np.nansum(px *  np.log2(px), -1)
    return px, -H

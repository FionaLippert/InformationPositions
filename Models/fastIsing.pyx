# cython: infer_types=True
# distutils: language=c++
# -*- coding: utf-8 -*-
# __author__ = 'Casper van Elteren'
"""
Created on Tue Feb  6 09:36:17 2018

@author: Casper van Elteren
"""
from Models.models cimport Model
# from models cimport Model
import numpy  as np

from scipy.stats import linregress
import networkx as nx, multiprocessing as mp, \
                scipy,  functools, copy, time
from tqdm import tqdm
import copy

# ___CythonImports___
cimport cython
from cython cimport numeric
from cython.parallel cimport parallel, prange, threadid
from cpython.ref cimport PyObject
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF

cimport numpy as np # overwrite some c  backend from above

from libc.math cimport exp
from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement
from libc.stdio cimport printf

# from libc.math cimport max, min

# use external exp
cdef extern from "vfastexp.h":
    double exp_approx "EXP" (double) nogil
cdef extern from *:
    cdef cppclass PyObjectHolder:
        PyObject *ptr
        PyObjectHolder(PyObject *o) nogil


cdef class Ising(Model):
    # def __cinit__(self, *args, **kwargs):
    #     print('cinit fastIsing')
    def __init__(self, \
                 graph,\
                 temperature = 1,\
                 agentStates = [-1 ,1],\
                 nudgeType   = 'constant',\
                 updateType  = 'async', \
                 magSide     = 'neg',\
                 ):
        # print('Init ising')
        super(Ising, self).__init__(\
                  graph       = graph, \
                  agentStates = agentStates, \
                  updateType  = updateType, \
                  nudgeType   = nudgeType)


        cdef np.ndarray H  = np.zeros(self.graph.number_of_nodes(), float)
        for node, nodeID in self.mapping.items():
            H[nodeID] = graph.nodes()[node].get('H', 0)
        # for some reason deepcopy works with this enabled...
        self.states           = np.asarray(self.states.base).copy()
        self.nudges           = np.asarray(self.nudges.base).copy()
        # specific model parameters
        self._H               = H
        self.beta             = np.inf if temperature == 0 else 1 / temperature
        self.t                = temperature
        self.magSideOptions   = {'': 0, 'neg': -1, 'pos': 1}
        self.magSide          = magSide

    @property
    def H(self): return self._H

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.overflowcheck(False)
    cpdef np.ndarray burnin(self,\
                 int samples = int(1e2), \
                 double threshold = 1e-2, ):
        """
        Go to equilibrium distribution; uses magnetization and linear regression
        to see if the system is stabel
        """

        # magnetization function
        magnetization = np.mean
        cdef:
            y  = np.array(magnetization(self.states)) # for regression
            int h, counter = 0 # tmp var and counter
            double beta        # slope value
            np.ndarray x # for regression
            long[::1] states
            long[:, ::1] r

        # print('Starting burnin')
        while True:
            r      = self.sampleNodes(1) # produce shuffle
            states = self.updateState(r[0]) # update state
            # check if magnetization = constant
            y      = np.hstack((y, np.abs(magnetization(states))))
            if counter > samples :
                # do linear regression
                h = len(y + 2) # plus 2 due to starting point
                x = np.arange(h)
                beta = linregress(x, y).slope
                if abs(beta) < threshold:
                    break
            counter += 1
        else:
            print('Number of bunin samples used {0}\n\n'.format(counter))
            print(f'absolute mean magnetization last sample {abs(y[-1])}')
        return y


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.overflowcheck(False)
    cdef double energy(self, \
                        int  node, \
                        long[::1] states) nogil :
                       # cdef double energy(self, \
                       #                    int  node, \
                       #                    long[::1] states)  :
        """
        input:
            :nsyncode: member of nodeIDs
        returns:
                :energy: current energy of systme config for node
        """
        cdef:
            long length            = self._adj[node].neighbors.size()
            long neighbor, i
            double weight
            double energy          = -self._H [node] * states[node]
        for i in range(length):
            neighbor = self._adj[node].neighbors[i]
            weight   = self._adj[node].weights[i]
            energy  -= states[node] * states[neighbor] * weight
        energy -= self._nudges[node] * states[node]
        # energy *= (1 + self._nudges[node])
        return energy

    cpdef long[::1] updateState(self, long[::1] nodesToUpdate):
        return self._updateState(nodesToUpdate)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.overflowcheck(False)
    cdef long[::1] _updateState(self, long[::1] nodesToUpdate) nogil:
    #cdef double _updateState(self, long[::1] nodesToUpdate) nogil:
    # cdef long[::1] _updateState(self, long[::1] nodesToUpdate):
        """
        Determines the flip probability
        p = 1/(1 + exp(-beta * delta energy))
        """
        cdef:
            # long[::1] states    = self._states # alias
            # long[::1] newstates = self._newstates
            int length          = nodesToUpdate.shape[0]
            double Z            = <double> self._nNodes
            long node
            double energy, p
            int n
        # for n in prange(length,  = True): # dont prange this
        for n in range(length):
            node      = nodesToUpdate[n]
            if self._fixedNodes[node] == 0:
                energy    = self.energy(node, self._states)
                # p = 1 / ( 1. + exp_approx(-self.beta * 2. * energy) )
                p  = 1 / ( 1. + exp(-self.beta * 2. * energy))
                # p  = p  +  self._nudges[node]
                # p += self._nudges[node]
                if self.rand() < p:
                    self._newstates[node] = -self._states[node]
        # uggly
        cdef double mu   =  0 # sign
        cdef long   NEG  = -1 # see the self.magSideOptions
        cdef long   POS  =  1
        # printf('%d ', mu)
        # compute mean
        for node in range(self._nNodes):
            self._states[node] = self._newstates[node] # update
            mu          += self._states[node] # normalization not really needed

        # out of state equilibrium?
        if (mu < 0 and self._magSide == POS) or\
         (mu > 0 and self._magSide == NEG):
            # printf('%f %d\n', mu, self._magSide)
            # flip if true
            for node in range(self._nNodes):
                self._states[node] = -self._states[node]
        return self._states

    cpdef np.ndarray[double] magTimeSeries(self, long nSteps, long burninSamples):
        cdef:
            long[:, ::1] r = self.sampleNodes(nSteps)
            long step
            double[::1] mags = np.zeros(nSteps)

        for step in range(nSteps):
            mags[step] = np.mean(self._updateState(r[step]))

        return mags.base


    cpdef np.ndarray[double] computeProb(self):
        """
        Compute the node probability for the current state p_i = 1/z * (1 + exp( -beta * energy))**-1
        """

        probs = np.zeros(self.nNodes)
        for node in self.nodeIDs:
            en = self.energy(node, self.states[node])
            probs[node] = exp(-self.beta * en)
        return probs / np.nansum(probs)



    cpdef  np.ndarray matchMagnetization(self,\
                              np.ndarray temps  = np.logspace(-3, 2, 20),\
                          int n             = int(1e3),\
                          int burninSamples = 100):
        """
        Computes the magnetization as a function of temperatures
        Input:
              :temps: a range of temperatures
              :n:     number of samples to simulate for
              :burninSamples: number of samples to throw away before sampling
        Returns:
              :temps: the temperature range as input
              :mag:  the magnetization for t in temps
              :sus:  the magnetic susceptibility
        """
        cdef double tcopy   = self.t # store current temp
        cdef results = np.zeros((2, temps.shape[0]))
        print("Computing mag per t")
        #for idx, t in enumerate(tqdm(temps)):
        for idx, t in enumerate((temps)):
            self.reset()
            self.t          = t
            # jdx             = self.magSideOptions[self.magSide]
            # if jdx:
                # self.states = jdx
            # else:
                # self.reset()
            # self.states     = jdx if jdx else self.reset() # rest to ones; only interested in how mag is kept
            self.burnin(burninSamples)
            tmp             = self.simulate(n)
            results[0, idx] = abs(tmp.mean())
            results[1, idx] = ((tmp**2).mean() - tmp.mean()**2) * self.beta
        # print(results[0])
        self.t = tcopy # reset temp
        return results



    def __deepcopy__(self, memo):
        tmp = Ising(
                    graph       = copy.deepcopy(self.graph), \
                    temperature = self.t,\
                    agentStates = list(self.agentStates.base),\
                    updateType  = self.updateType,\
                    nudgeType   = self.nudgeType,\
                    magSide     = self.magSide)
        # tmp.states = self.states
        return tmp

    def __reduce__(self):
        return (rebuild, (self.graph, \
                          self.t,\
                          list(self.agentStates.base.copy()),\
                          self.updateType,\
                          self.nudgeType,\
                          self.magSide, self.nudges.base))
    # property states:
    #     def __get__(self):
    #         return self._states.base
    #     def __set__(self, values):
    #         return np.array(self._states)

    # PROPERTIES
    @property
    def magSide(self):
        for k, v in self.magSideOptions.items():
            if v == self._magSide:
                return k
    @magSide.setter
    def magSide(self, value):
        idx = self.magSideOptions.get(value,\
              f'Option not recognized. Options {self.magSideOptions.keys()}')
        if isinstance(idx, int):
            self._magSide = idx
        else:
            print(idx)

    @property
    def H(self): return self._H

    @property
    def t(self):
        return self._t


    @t.setter
    def t(self, value):
        self._t   = value
        if np.isfinite(value):
            self.beta = 1 / value if value != 0 else np.inf
        else:
            self.beta = 0

cpdef Ising rebuild(object graph, double t, list agentStates, str updateType, str nudgeType, str magSide, \
              np.ndarray nudges):
    cdef Ising tmp = copy.deepcopy(Ising(graph, t, agentStates, nudgeType, updateType, magSide))
    tmp.nudges = nudges.copy()
    return tmp

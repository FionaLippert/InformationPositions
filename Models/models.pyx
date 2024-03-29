# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Casper van Elteren and Fiona Lippert'

import numpy as np
cimport numpy as np
import networkx as nx, functools, time
from tqdm import tqdm
import copy

cimport cython
from cython.parallel cimport parallel, prange
from cython.operator cimport dereference, preincrement
from libc.stdlib cimport malloc, free
# from libc.stdlib cimport rand
from libc.string cimport strcmp
from libc.stdio cimport printf
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libc.math cimport lround
cdef extern from "limits.h":
    int INT_MAX
    int RAND_MAX


# SEED SETUP
from posix.time cimport clock_gettime,\
timespec, CLOCK_REALTIME

# from sampler cimport Sampler # mersenne sampler
cdef class Model: # see pxd
    # def __cinit__(self, *args, **kwargs):
        # print(kwargs)
        # print ('cinit model')
        # graph           = kwargs.get('graph', [])
        # agentStates     = kwargs.get('agentStates', [-1, 1])
        # self.construct(graph, agentStates)
        # self.updateType = kwargs.get('updateType', 'single')
        # self.nudgeType  = kwargs.get('nudgeType', 'constant')

    def __init__(self,\
                 object graph, \
                 list agentStates = [-1, 1], \
                 str updateType   = 'single',\
                 str nudgeType    = 'constant'):
        '''
        General class for the models
        It defines the expected methods for the model; this can be expanded
        to suite your personal needs but the methods defined here need are relied on
        by the rest of the package.

        It translates the networkx graph into c++ unordered_map map for speed
        '''
        # print('Init model')
        # use current time as seed for rng
        cdef timespec ts
        clock_gettime(CLOCK_REALTIME, &ts)
        cdef unsigned int seed = ts.tv_sec
        # define rng sampler
        self.dist = uniform_real_distribution[double](0.0,1.0)
        self.seed = seed
        self.gen  = mt19937(self.seed)

        # create adj list
        self.construct(graph, agentStates)
        self.nudgeType  = copy.copy(nudgeType)
        self.updateType = updateType
        # self.sampler    = Sampler(42, 0., 1.)



    cpdef void construct(self, object graph, list agentStates):
        """
        Constructs adj matrix using structs
        """
        # print('Constructing')
        # check if graph has weights or states assigned and or nudges
        # note does not check all combinations
        # input validation / construct adj lists
        # defaults
        DEFAULTWEIGHT = 1.
        DEFAULTNUDGE  = 0.
        # DEFAULTSTATE  = random # don't use; just for clarity

        # forward declaration and init
        cdef:
            dict mapping = {} # made nodelabe to internal
            dict rmapping= {} # reverse
            str delim = '\t'
            np.ndarray states = np.zeros(graph.number_of_nodes(), int, 'C')
            int counter
            # double[::1] nudges = np.zeros(graph.number_of_nodes(), dtype = float)
            np.ndarray nudges = np.zeros(graph.number_of_nodes(), dtype = float)
            unordered_map[long, Connection] adj # see .pxd

            np.ndarray fixedNodes = np.zeros(graph.number_of_nodes(), int, 'C')


        from ast import literal_eval
        connecting = graph.neighbors if isinstance(graph, nx.Graph) else nx.predecessors
        #print(graph.nodes())
        #print([n for n in graph.neighbors('0')])

        # cdef dict _neighbors = {}
        # cdef dict _weights   = {}
        # generate adjlist
        for line in nx.generate_multiline_adjlist(graph, delim):
            add = False # tmp for not overwriting doubles
            # input validation
            lineData = []
            #print(line)
            # if second is not dict then it must be source
            for prop in line.split(delim):
                try:
                    i = literal_eval(prop) # throws error if only string
                    lineData.append(i)
                except:
                    lineData.append(prop) # for strings
            node, info = lineData
            #print(node)
            #print(graph.node[node])
            # check properties, assign defaults
            if 'state' not in graph.node[node]:
                idx = np.random.choice(agentStates)
                # print(idx, agentStates)
                graph.node[node]['state'] = idx
            if 'nudge' not in graph.node[node]:
                graph.node[node]['nudge'] =  DEFAULTNUDGE

            # if not dict then it is a source
            if isinstance(info, dict) is False:
                # add node to seen
                if node not in mapping:
                    # append to stack
                    counter             = len(mapping)
                    mapping[node]       = counter
                    rmapping[counter]   = node

                # set source
                source   = node
                sourceID = mapping[node]

                states[sourceID] = <long> graph.node[node]['state']
                nudges[sourceID] = <double> graph.node[node]['nudge']
            # check neighbors
            else:
                if 'weight' not in info:
                    graph[source][node]['weight'] = DEFAULTWEIGHT
                if node not in mapping:
                    counter           = len(mapping)
                    mapping[node]     = counter
                    rmapping[counter] = node

                # check if it has a reverse edge
                if graph.has_edge(node, source):
                    sincID = mapping[node]
                    weight = graph[node][source]['weight']
                    # check if t he node is already in stack
                    if sourceID in set(adj[sincID]) :
                        add = False
                    # not found so we should add
                    else:
                        add = True
                # add source > node
                sincID = <long> mapping[node]
                adj[sourceID].neighbors.push_back(<long> mapping[node])
                adj[sourceID].weights.push_back(<double> graph[source][node]['weight'])
                # add reverse
                if add:
                    adj[sincID].neighbors.push_back( <long> sourceID)
                    adj[sincID].weights.push_back( <double> graph[node][source]['weight'])

        # public and python accessible
        self.graph       = graph
        self._mapping     = mapping
        self._rmapping    = rmapping
        self._adj        = adj

        self.agentStates = np.asarray(agentStates, dtype = int).copy()
        # print(states, agentStates)

        self._nudges     = nudges.copy()
        self._nStates    = len(agentStates)


        #private
        # note nodeids will be shuffled and cannot be trusted for mapping
        # use mapping to get the correct state for the nodes
        _nodeids        = np.arange(graph.number_of_nodes(), dtype = long)
        self._nodeids   = _nodeids.copy()
        self._states    = states.copy()
        self._newstates = states.copy()
        self._nNodes    = graph.number_of_nodes()

        self._fixedNodes = fixedNodes.copy()
        # print(f'Done {id(self)}')

    # cdef long[::1]  _updateState(self, long[::1] nodesToUpdate) :
    cdef long[::1]  _updateState(self, long[::1] nodesToUpdate) nogil:
        return self._nodeids


    cpdef long[::1] updateState(self, long[::1] nodesToUpdate):
        return self._nodeids


    cdef double rand(self) nogil:
        return self.dist(self.gen)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.overflowcheck(False)
    cdef long [:, ::1] sampleNodes(self, long  nSamples) nogil:
    # cdef long [:, ::1] sampleNodes(self, long  nSamples):
        """
        Shuffles nodeids only when the current sample is larger
        than the shuffled array
        N.B. nodeids are mutable
        """
        # check the amount of samples to get
        cdef int sampleSize
        if self._updateType == 'single':
            sampleSize = 1
        # elif self._updateType == 'serial':
        #     return self._nodeids[None, :]
        else:
            sampleSize = self._nNodes
        cdef:
            # TODO replace this with a nogil version
            # long _samples[nSamples][sampleSize]
            long [:, ::1] samples
            # long sample
            long start
            long i, j, k
            long samplei
            int correcter = nSamples * sampleSize
        # replace with nogil variant
        with gil:
            samples = np.ndarray((nSamples, sampleSize), dtype = int)
        for samplei in range(nSamples):
            # shuffle if the current tracker is larger than the array
            start  = (samplei * sampleSize) % self._nNodes
            if self._updateType != 'serial' and (start + sampleSize >= self._nNodes or correcter == 1):
                for i in range(self._nNodes):
                    # shuffle the array without replacement
                    j                = lround(self.rand() * (self._nNodes - 1))
                    k                = self._nodeids[j]
                    self._nodeids[j] = self._nodeids[i]
                    self._nodeids[i] = k
                    # enforce atleast one shuffle in single updates; otherwise same picked
                    if correcter == 1 : break
            # assign the samples; will be sorted in case of serial
            for j in range(sampleSize):
                samples[samplei, j]    = self._nodeids[start + j]
        return samples

    cpdef void reset(self):
        self.states = np.random.choice(\
            self.agentStates, size = self._nNodes)

    cpdef void resetAllToAgentState(self, int stateIdx, int i=0):
        if stateIdx == -2:
            # uniformly random
            #self.states = np.random.choice(\
            #    self.agentStates, size = self._nNodes)
            i = np.mod(i, self.agentStates.shape[0])
            self.states = np.ones(self._nNodes, int) * self.agentStates[i]
        elif stateIdx == -1:
            # uniformly random
            self.states = np.random.choice(\
                self.agentStates, size = self._nNodes)
        else:
            # all nodes the same
            assert stateIdx < self.agentStates.shape[0]
            self.states = np.ones(self._nNodes, int) * self.agentStates[stateIdx]


    cpdef tuple neighboursAtDist(self, long nodeG, int maxDist):
        #assert nodeIdx < self._nNodes and nodeIdx >= 0

        cdef:
            #long[::1] neighbours
            #unordered_map[long, vector[long]] allNeighbours_G, allNeighbours_idx
            dict allNeighboursG = {d : [] for d in range(1, maxDist+1)}
            dict allNeighboursIdx = {d : [] for d in range(1, maxDist+1)}
            int undir = not nx.is_directed(self.graph)
            #long node = self.rmapping[node_idx]
            #nx.Graph total, inner
            int d

        total = nx.ego_graph(self.graph, nodeG, radius=0, undirected=undir)

        for d in range(1, maxDist+1):
            inner = total
            total = nx.ego_graph(self.graph, nodeG, radius=d, undirected=undir)
            for n in (set(total.nodes()) - set(inner.nodes())):
                #allNeighbours_G[d].push_back(n)
                allNeighboursG[d].append(n)
                allNeighboursIdx[d].append(self.mapping[n])
                #allNeighbours_idx[d].push_back(self.mapping[n])
            #allNeighbours[d] = np.array([self.mapping[n] for n in (set(total.nodes()) - set(inner.nodes()))], dtype=np.intc)
            #allNeighbours[d].push_back(neighbours)
        #print("neighbors: {}".format(neighbors))

        return allNeighboursG, allNeighboursIdx

    cpdef unordered_map[long, unordered_map[long, vector[long]]] neighboursAtDistAllNodes(self, long[::1] nodesG, int maxDist):

        cdef:
            long nNodes = nodesG.size
            #unordered_map[long, unordered_map[long, vector[long]]] neighboursIdx = vector[unordered_map[long, vector[long]]](nNodes)
            unordered_map[long, unordered_map[long, vector[long]]] neighboursG

        for n in nodesG:
            neighboursG[n] = self.neighboursAtDist(n, maxDist)[0]

        return neighboursG


    def removeAllNudges(self):
        """
        Sets all nudges to zero
        """
        self.nudges[:] = 0

    def releaseFixedNodes(self):
        """
        Frees all fixed nodes
        """
        self._fixedNodes[:] = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.overflowcheck(False)
    cdef long[:,::1] _simulate(self, long long int  samples) nogil:
        cdef:
            long[:, ::1] results
            long[:, ::1] r = self.sampleNodes(samples)
            int i

        with gil: results = np.zeros((samples, self._nNodes), int)

        for i in range(samples):
            results[i] = self._updateState(r[i])
        return results

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.overflowcheck(False)
    cpdef np.ndarray simulate(self, long long int  samples):
        return self._simulate(samples).base


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.overflowcheck(False)
    cdef long[::1] simulateNSteps(self, long nSteps) nogil:
        cdef:
            long[:, ::1] r = self.sampleNodes(nSteps)
            long step

        for step in range(nSteps):
            #with gil: print(self._states.base)
            #with gil: print(self._newstates.base)
            self._updateState(r[step])

        return self._states


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef bytes encodeStateToString(self, vector[long] nodes):
        """Maps states of given nodes to string"""
        cdef:
            int N = nodes.size()
            long[::1] subset = np.zeros(N, int)
            long i, n
            bytes s

        for i in range(N):
            subset[i] = self._states[nodes[i]]

        #print(subset.base, np.frombuffer(subset.base.astype(float).tobytes()))
        s = subset.base.astype(float).tobytes()
        return s

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef int encodeStateToAvg(self, vector[long] nodes, double[::1] bins) nogil:
        """Maps states of given nodes to binned avg magnetization"""
        cdef:
            long N = nodes.size(), nBins = bins.shape[0]
            #long[::1] subset = np.zeros(N, int)
            double avg = 0
            long i, n
            #bytes s

        for i in range(N):
            avg += self._states[nodes[i]]

        avg /= N

        for i in range(nBins):
            if avg <= bins[i]:
                avg = i
                break

        #with gil: print(avg, i, bins.base)

        return <int>avg


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void loadStatesFromString(self, bytes statesString, vector[long] nodes):
        """Maps string back to system state. States of nodes not included in the snapshot remain the same (--> burnin samples needed to forget them)"""
        cdef:
            int N = nodes.size()
            long i, n

        dec = np.frombuffer(statesString).astype(int)
        #print(f'decoded string: {dec}')

        #self.reset() # assign random states

        for i in range(N):
            n = nodes[i]
            self.states[n] = dec[i]
        #print(f'reconstructed state: {self._states.base}')



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void _loadStatesFromString(self, long[::1] snapshot, vector[long] nodes) nogil:
        """Maps string back to system state. States of nodes not included in the snapshot remain the same (--> burnin samples needed to forget them)"""
        cdef:
            int N = nodes.size()
            long i, n

        #dec = np.frombuffer(statesString).astype(int)
        #print(f'decoded string: {dec}')

        #self.reset() # assign random states

        for i in range(N):
            n = nodes[i]
            self._newstates[n] = snapshot[i]
            self._states[n]    = snapshot[i]
        #print(f'reconstructed state: {self._states.base}')


    # TODO: make class pickable
    # hence the wrappers
    @property
    def adj(self)       : return self._adj
    @property
    def states(self)    : return self._states
    @property
    def updateType(self): return self._updateType
    @property
    def nudgeType(self) : return self._nudgeType
    @property #return mem view of states
    def states(self)    : return self._states
    @property
    def nodeids(self)   : return self._nodeids
    @property
    def nudges(self)    : return self._nudges
    @property
    def nNodes(self)    : return self._nNodes
    @property
    def nStates(self)   : return self._nStates
    @property
    def nodeids(self)   : return self._nodeids
    @property
    def seed(self)      : return self._seed
    @property
    def fixedNodes(self): return self.fixedNodes
    @property
    def mapping(self)   : return self._mapping
    @property
    def rmapping(self)  : return self._rmapping

    @seed.setter
    def seed(self, value):
        if isinstance(value, int) and value >= 0:
            self._seed = value
            self.gen   = mt19937(self.seed)
        else:
            print("Value is not unsigned long")

    #cdef void _incrSeed(self, long value) nogil:
    #    self._seed = self._seed + value
    #    self.gen = mt19937(self._seed)


    # TODO: reset all after new?
    @nudges.setter
    def nudges(self, vals):
        """
        Set nudge value based on dict using the node labels
        """
        self._nudges[:] =  0
        if isinstance(vals, dict):
            for k, v in vals.items():
                idx = self.mapping[k]
                self._nudges[idx] = v

    @fixedNodes.setter
    def fixedNodes(self, nodes):
        """
        Set bit in fixedNodes to True for all given nodes (internal index used!)
        """
        self._fixedNodes[:] =  0
        #if isinstance(nodes, vector[int]): # TODO make this more general. Any type of list, array etc could be passed as long as elements are integers
        #print(self._fixedNodes.shape[0])
        #print(nodes)
        for n in nodes:
            self._fixedNodes[n] = 1
        #else:
        #    print("Nodes are not given as vector of integers")

    @updateType.setter
    def updateType(self, value):
        """
        Input validation of the update of the model
        Options:
            - sync  : synchronous; update independently from t > t + 1
            - async : asynchronous; update n Nodes but with mutation possible
            - single: update 1 node random
            - serial: like like scan
        """
        assert value in 'sync async single serial'
        self._updateType = value
        # allow for mutation if async else independent updates
        if value == 'async':
            self._newstates = self._states
        else:
            if value == 'serial':
                self._nodeids = np.sort(self._nodeids) # enforce  for sampler
            self._newstates = self._states.copy()

    @nudgeType.setter
    def nudgeType(self, value):
        assert value in 'constant pulse'
        self._nudgeType = value

    # watch out: does not copy array but references
    cdef void _setStates(self, long[::1] newStates) nogil:
        self._newstates = newStates
        self._states = newStates

    cpdef void setStates(self, long[::1] newStates):
        self._newstates = newStates.copy()
        self._states = newStates.copy()

    @states.setter
    def states(self, value):
        if isinstance(value, int):
            self._newstates[:] = value
            self._states   [:] = value

        elif isinstance(value, np.ndarray) or isinstance(value, list):
            assert len(value) == self.nNodes
            value = np.asarray(value)
            self._newstates = value
            self._states    = value

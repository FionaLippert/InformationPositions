# __author__ = 'Fiona Lippert'
# distutils: language=c++
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
cimport numpy as np
from Models.models cimport Model

cpdef double entropyEstimateH2(long[::1] counts)

cdef double entropyFromProbs(double[::1] probs) nogil

cpdef double[::1] binaryEntropies(long[:,::1] snapshots)

cpdef double pairwiseMI(long[:,::1] snapshots, double[::1] binEntropies, long nodeIdx1, long nodeIdx2) nogil

cpdef double spinCorrelation(long[:,::1] snapshots, long nodeIdx1, long nodeIdx2) nogil

cpdef tuple computeMI_jointPDF(np.ndarray jointDistr, long Z)

cpdef tuple computeMI_jointPDF_fromDict(unordered_map[int, unordered_map[string, long]] jointDistrDict, long Z)

cpdef tuple pairwiseMI_allNodes(Model model, np.ndarray nodesG, long[:,::1] snapshots, \
                int threads=*)

cpdef tuple pairwiseMI_oneNode(Model model, long nodeG, long[:,::1] snapshots, \
                int threads=*)

cpdef tuple processJointSnapshots_allNodes(np.ndarray avgSnapshots, long Z, np.ndarray nodesG, long maxDist, np.ndarray avgSystemSnapshots=*)

cpdef tuple processJointSnapshots_oneNode(np.ndarray avgSnapshots, np.ndarray avgSystemSnapshots, long Z, long maxDist)

cpdef double pCond_theory(int dist, double beta, int num_children, int node_state, np.ndarray neighbour_states)

cpdef double MI_tree_theory(int depth, double beta, int num_children)

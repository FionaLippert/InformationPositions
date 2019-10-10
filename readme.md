This is the code base for my Master thesis "An information-theoretical framework to identify potential informants in criminal networks"

# Instructions

Run `python setup.py build_ext --inplace` to build the Cython packages

Use the python scripts `run_condMI_nodelist.py` (more accurate but slower) or `run_jointMI_nodelist.py` (heuristic that is feasible also for large-scale complex networks) to estimate the decay of mutual information for all nodes in the specified network.


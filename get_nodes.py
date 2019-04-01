#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import networkx as nx
import numpy as np
import os
import glob

for filepath in glob.iglob('networkData/ER/*.gpickle'):
    G = nx.read_gpickle(filepath)
    nodes = list(G)
    path = filepath.strip('.gpickle')
    #np.save(path + '_nodes.npy', np.array(nodes))

    with open(path + '_sample_nodes.txt', 'w') as f:
        for i in range(2):
            f.write(f'{nodes[i]}\n')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
import networkx as nx
from Utils import IO


#gtype='small_graphs/N=50'

N=70
k=N*0.05*1.2
gtype=f'small_graphs/N={N}_p=0.05'


#k_list = [1.96, 2.28, 2.80, 2.88, 2.16, 2.72, 3.16, 3.24, 3.52, 3.64]
#k_list = [2.72, 3.16, 3.24, 3.52, 3.64]
#k_list = [3.52]

#for k in k_list:
for i in range(1,5):
    #graph=f'ER_k={k:.2f}_N=50'
    graph=f'ER_k={k:.2f}_N={N}_v{i}'

    G = nx.read_gpickle(f'networkData/{gtype}/{graph}/{graph}.gpickle')
    np.save(f'networkData/{gtype}/{graph}/{graph}_nodes.npy', list(G))

    Tc_path = f'DataTc_new/{gtype}/{graph}/{graph}_Tc.txt'
    with open(Tc_path) as f:
       for cnt, line in enumerate(f):
           T = re.findall(r'[\d\.\d]+', line)[0]
           print(f'run T={T}')
           if cnt == 0:
               magSide = 'pos'
           else:
               magSide = 'fair'

           print(magSide)
           if cnt > 0:
               subprocess.call(['python3', 'run_systemEntropy.py', \
                    str(T), \
                    f'output_systemEntropy/{gtype}/{graph}/T={T}', \
                    f'networkData/{gtype}/{graph}/{graph}.gpickle', \
                    '--nodes', f'networkData/{gtype}/{graph}/{graph}_nodes.npy', \
                    '--k', '1', \
                    '--trials', '1', \
                    '--snapshots', '100000', \
                    '--excludeNodes', \
                    '--magSide', magSide])
"""
gtype = 'small_graphs/N=50'
graph = f'ER_k=2.88_N=50'
gtype = 'small_graphs/N=50_p=0.05'

N = 50
top = int(N * 0.25)

k = 3

for v in range(7,10):
    graph = f'ER_k=3.00_N=50_v{v}'
    Tc_results = IO.TcResult.loadFromPickle(f'DataTc_new/{gtype}/{graph}', f'{graph}_Tc_results')
    Ts = [Tc_results.T_low, Tc_results.T_c, Tc_results.T_high]
    start = 2 if v == 7 else 0
    for i in range(start, 3):
        T = Ts[i]
        if i == 0:
            magSide = 'pos'
        else:
            magSide = 'fair'

        path = f'output_final/{gtype}/{graph}/magMI/T={T:.2f}'
        #mi = np.load([file for file in glob.iglob(f'{path}/MI_meanField_*')][0])
        mi = IO.SimulationResult.loadNewestFromPickle(path, 'avg').mi

        #nodes = np.load(f'networkData/{gtype}/{graph}_nodes.npy')
        nodes = list(mi.keys())

        iv = {}
        for node in mi.keys():
            iv[node] = np.nansum(mi[node])

        ranking = sorted(iv.items(), key=lambda kv: kv[1], reverse=True)
        ranked_nodes, iv_values = zip(*ranking)
        print(ranked_nodes)
        print(iv_values)

        target_dir = f'output_systemEntropySets/{gtype}/{graph}/T={T:.2f}/k={k}/brute_force_top{top}_trials=1'
        print(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        ranked_nodes = np.array(ranked_nodes)
        iv_values = np.array(iv_values)

        #print(ranked_nodes[np.where(iv_values > 0.1)].size)

        #sets = itertools.combinations(ranked_nodes[np.where(iv_values > 0.1)], k)
        #sets = itertools.combinations(ranked_nodes[:10], k)


        np.save(f'{target_dir}/nodes_top{top}.npy', ranked_nodes[:top])

        subprocess.call(['python3', 'run_systemEntropy.py', \
                 str(T), \
                 target_dir, \
                 f'networkData/{gtype}/{graph}/{graph}.gpickle', \
                 '--nodes', f'{target_dir}/nodes_top{top}.npy', \
                 '--snapshots', '100000', \
                 '--excludeNodes',
                 '--magSide', magSide, \
                 '--trials', '1',\
                 '--k', str(k) ])

"""
"""
#nodes = np.array([28, 38, 0])
#os.makedirs(f'output_systemEntropy/{gtype}/{graph}/T={T}/top3', exist_ok=True)
nodes = np.array([0, 40, 55]) # random.choice(nodes, 3)
os.makedirs(f'output_systemEntropy/{gtype}/{graph}/T={T}/random3', exist_ok=True)
np.save(f'output_systemEntropy/{gtype}/{graph}/T={T}/random3/optimal_nodes.npy', nodes)

for i in range(10):
    subprocess.call(['python3', 'run_systemEntropy.py', \
             str(T), \
             f'output_systemEntropy/{gtype}/{graph}/T={T}/random3', \
             f'networkData/{gtype}/{graph}.gpickle', \
             '--nodes', f'output_systemEntropy/{gtype}/{graph}/T={T}/random3/optimal_nodes.npy', \
             '--snapshots', '10000', \
             '--excludeNodes'])
"""

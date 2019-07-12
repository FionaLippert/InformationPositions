#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

"""
gtype='small_graphs/N=50'

k_list = [1.96, 2.28, 2.80, 2.88, 2.16, 2.72, 3.16, 3.24, 3.52, 3.64]
#k_list = [2.72, 3.16, 3.24, 3.52, 3.64]
#k_list = [2.28]

for k in k_list:
    graph=f'ER_k={k:.2f}_N=50'

    Tc_path = f'DataTc_new/{gtype}/{graph}_Tc.txt'
    with open(Tc_path) as f:
       for cnt, line in enumerate(f):
           T = re.findall(r'[\d\.\d]+', line)[0]
           print(f'run T={T}')
           if cnt == 0:
               magSide = 'pos'
               subprocess.call(['python3', 'run_systemEntropy.py', \
                        str(T), \
                        f'output_systemEntropy/{gtype}/{graph}/T={T}', \
                        f'networkData/{gtype}/{graph}.gpickle', \
                        '--nodes', f'networkData/{gtype}/{graph}_nodes.npy', \
                        '--single', \
                        '--snapshots', '10000', \
                        '--excludeNodes', \
                        '--magSide', magSide])
"""
gtype = 'small_graphs/N=50'
graph = f'ER_k=2.28_N=50'
T=0.72
k=3

path = f'output_final/small_graphs/N=50/{graph}/T={T}'
#mi = np.load([file for file in glob.iglob(f'{path}/MI_meanField_*')][0])
mi = IO.SimulationResult.loadNewestFromPickle(path, 'avg').mi

nodes = np.load(f'networkData/{gtype}/{graph}_nodes.npy')

iv = {}
for node in nodes:
    iv[node] = np.nansum(mi[node])

ranking = sorted(iv.items(), key=lambda kv: kv[1], reverse=True)
ranked_nodes, iv_values = zip(*ranking)
print(ranked_nodes)
print(iv_values)

target_dir = f'output_systemEntropySets/{gtype}/{graph}/T={T}/k=3/brute_force_top10_trails=10'
print(target_dir)
os.makedirs(target_dir, exist_ok=True)

ranked_nodes = np.array(ranked_nodes)
iv_values = np.array(iv_values)

#print(ranked_nodes[np.where(iv_values > 0.1)].size)

#sets = itertools.combinations(ranked_nodes[np.where(iv_values > 0.1)], k)
sets = itertools.combinations(ranked_nodes[:10], k)


for i, s in enumerate(sets):
    if i > 3:
        np.save(f'{target_dir}/nodes_{i}.npy', s)

        subprocess.call(['python3', 'run_systemEntropy.py', \
                 str(T), \
                 target_dir, \
                 f'networkData/{gtype}/{graph}.gpickle', \
                 '--nodes', f'{target_dir}/nodes_{i}.npy', \
                 '--snapshots', '10000', \
                 '--excludeNodes',
                 '--magSide', 'pos', \
                 '--trials', '10'])


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

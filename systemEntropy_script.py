#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


import subprocess

gtype='small_graphs/N=50'

k_list = [1.96, 2.80, 2.88, 2.16, 2.72, 3.16, 3.24, 3.52, 3.64]

for k in k_list:
    graph=f'ER_k={k:.2f}_N=50'

    Tc_path = f'DataTc_new/{gtype}/{graph}_Tc.txt'
    with open(Tc_path) as f:
       for cnt, line in enumerate(f):
           print(f'run T={line}')
           subprocess.call(['python3', 'run_systemEntropy.py', \
                    line, \
                    f'output_systemEntropy/{gtype}/{graph}/T={line}', \
                    f'networkData/{gtype}/{graph}.gpickle', \
                    '--nodes', f'networkData/{gtype}/{graph}_nodes.npy', \
                    '--single', \
                    '--snapshots', '10000', \
                    '--excludeNodes'])

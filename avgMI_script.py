#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re

gtype='small_graphs/N=50'
#gtype='BA'
output_dir = 'output_final'

k_list = [1.96, 2.28, 2.80, 2.88, 2.16, 2.72, 3.16, 3.24, 3.52, 3.64]
#k_list = [2.28]
#k_list = [2.0, 4.0]
#k_list = [3]

for k in k_list:
    graph=f'ER_k={k:.2f}_N=50'
    #graph=f'ER_k={k:.2f}_N=1000/ER_k={k:.2f}_N=1000_v0'
    #graph=f'BA_m={k}_N=1000/BA_m={k}_N=1000_v0'

    Tc_path = f'DataTc_new/{gtype}/{graph}_Tc.txt'
    with open(Tc_path) as f:
       for cnt, line in enumerate(f):
           T = re.findall(r'[\d\.\d]+', line)[0]
           print(f'run T={T}')
           if cnt == 0:
               magSide = 'pos'
               subprocess.call(['python3', 'run_jointMI_nodelist.py', \
                        str(T), \
                        f'{output_dir}/{gtype}/{graph}/T={T}', \
                        f'networkData/{gtype}/{graph}.gpickle', \
                        '--nodes', f'networkData/{gtype}/{graph}_nodes.npy', \
                        '--bins', '100', \
                        '--pairwise', \
                        '--magSide', magSide])

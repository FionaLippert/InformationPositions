#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


import subprocess, os, pickle, sys, glob, re
import numpy as np

gtypes = ['BA', 'ER', 'ER', 'WS']
graphs = [
        'BA_m=3_N=1000', \
        'ER_k=2.0_N=1000',
        'ER_k=4.0_N=1000',
        'WS_k=4_N=1000' ]

version = 0

for i, graph in enumerate(graphs):

    Tc_path = f'DataTc_new/{gtypes[i]}/{graph}/{graph}_v{version}_Tc.txt'
    with open(Tc_path) as f:
       for cnt, line in enumerate(f):
           T = re.findall(r'[\d\.\d]+', line)[0]
           print(f'run T={T}')
           if cnt == 0:
               magSide = 'pos'
           else:
               magSide = 'fair'
           subprocess.call(['python3', 'run_jointMI_nodelist.py', \
                    str(T), \
                    f'output_jointMI_final/{gtypes[i]}/{graph}/{graph}_v{version}/T={T}', \
                    f'networkData/{gtypes[i]}/{graph}/{graph}_v{version}.gpickle', \
                    '--neighboursDir', f'networkData/{gtypes[i]}/{graph}/{graph}_v{version}', \
                    '--nodes', f'networkData/{gtypes[i]}/{graph}/{graph}_v{version}_nodes.npy', \
                    '--pairwise', \
                    '--magSide', magSide])

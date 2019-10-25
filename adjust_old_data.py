#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fiona Lippert
"""

import pickle, os, json, time, glob
from Utils import IO



def find_files(dir, type, ext='.pickle'):
    filepaths = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for file in filenames:
            if file.endswith(ext) and type in file:
                filepaths.append(os.path.join(dirpath, file))
        for subdir in dirnames:
            filepaths_sub = find_files(os.path.join(dirpath, subdir), type, ext)
            if len(filepaths_sub) > 0:
                filepaths.extend(filepaths_sub)

    return filepaths

"""

for f in find_files('../masterthesis_casperscode/DataTc_new', '_Tc_results_dict'):
    head, tail = os.path.split(f)
    print(tail)

    data = IO.loadPickle(head, tail)

    gname = os.path.splitext(tail)[0].split('_Tc_results')[0]

    result = IO.TempsResult(data['temps'], data['mags'], data['abs_mags'], \
        data['sus'], data['binder'], data['T_c'], data['T_d'], data['T_o'], gname)
    dir = f'backup/tempsData/{head}'
    os.makedirs(dir, exist_ok=True)
    result.saveToPickle(dir)
"""

directory = 'output_systemEntropyGreedy'
for f in find_files(f'../masterthesis_casperscode/{directory}', 'simulation_results', 'dict.pickle'):
    head, tail = os.path.split(f)
    print(tail)

    data = IO.loadPickle(head, tail)
    #type = data['type']
    #data.pop('type')

    result = IO.SimulationResult(**data)
    t = tail.split('_')[-2]
    print(t)
    dir = f'backup/{directory}/{head}'
    os.makedirs(dir, exist_ok=True)
    result.saveToPickle(dir, timestamp=t)

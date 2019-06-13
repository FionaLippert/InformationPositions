#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fiona Lippert
"""

from numpy import *
from matplotlib.pyplot import *
from dataclasses import dataclass
import pickle, pandas, os, re, json, datetime
import networkx as nx
from collections import defaultdict, OrderedDict



def newest(path):
    """
    Returns sorted files by time
    """
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return sorted(paths, key=os.path.getctime)


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)

def loadPickle(path, fileName):
    if not fileName.endswith('.pickle'):
        fileName += '.pickle'
    with open(os.path.join(path, fileName), 'rb') as f:
        return renamed_load(f)

def savePickle(path, fileName, objects):
    #TODO: warning; apparantly pickle <=3 cannot handle files
    # larger than 4 gb.
    if not fileName.endswith('.pickle'):
        fileName += '.pickle'
    print(f'Saving {fileName}')
    with open(os.path.join(path, fileName), 'wb') as f:
        pickle.dump(objects, f, protocol = pickle.HIGHEST_PROTOCOL)

def saveSettings(targetDirectory, settings, prefix=''):
    print('Saving settings')
    with open(targetDirectory + f'/{prefix}Settings.json', 'w') as f:
        json.dump(settings, f)

def saveResults(targetDirectory, result_dict, name):
    with open(os.path.join(targetDirectory, f'{name}.json'), 'w') as f:
        json.dump(result_dict, f)

def loadResults(targetDirectory, name):
    with open(os.path.join(targetDirectory, f'{name}.json')) as f:
        return json.load(f)

def readSettings(targetDirectory, dataType = '.pickle'):
    try:
        with open(targetDirectory + '/settings.json') as f:
            return json.load(f)

    # TODO: uggly, also lacks all the entries
    # attempt to load from a file
    except FileNotFoundError:
        # use re to extract settings
        settings = {}
        for file in os.listdir(targetDirectory):
            if file.endswith(dataType) and 'mags' not in file:
                file = file.split(dataType)[0].split('_')
                for f in file:
                    tmp = f.split('=')
                    if len(tmp) > 1:
                        key, value = tmp
                        if key in 'nSamples k step deltas':
                            if key == 'k':
                                key = 'repeats'
                            settings[key] = value
            # assumption is that all the files have the same info
            break
        saveSettings(targetDirectory, settings)
        return settings


class SimulationResult:
    def __init__(self, success, iterations, max_dist, gs, psi):
        self.success = success
        self.iterations = iterations
        self.max_dist = max_dist
        self.gs = gs
        self.psi = psi

class TcResult:
    def __init__(self, success, iterations, max_dist, gs, psi):
        self.success = success
        self.iterations = iterations
        self.max_dist = max_dist
        self.gs = gs
        self.psi = psi

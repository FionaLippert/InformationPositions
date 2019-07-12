#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fiona Lippert
"""

from numpy import *
from matplotlib.pyplot import *
from dataclasses import dataclass
import pickle, pandas, os, re, json, datetime, time, glob
import networkx as nx
from collections import defaultdict, OrderedDict

def get_timestamp(path):
    no_ext = os.path.splitext(path)[0]
    timestamp = no_ext.split('_')[-1]
    return timestamp

def newest(dir, filename):
    """
    Returns sorted files by time (descending)
    """
    paths = [file for file in glob.iglob(f'{os.path.join(dir, filename)}*')]
    return sorted(paths, key=get_timestamp, reverse=True)


def loadPickle(path, fileName):
    if not fileName.endswith('.pickle'):
        fileName += '.pickle'
    with open(os.path.join(path, fileName), 'rb') as f:
        return pickle.load(f)

def loadAllPickle(path, filePrefix):
    allFiles = []
    for file in glob.iglob(f'{os.path.join(path, filePrefix)}*'):
        with open(file, 'rb') as f:
            allFiles.append(pickle.load(f))
    return allFiles

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

    def __init__(self, type, **kwargs):
        assert type in { 'vector', 'avg', 'pairwise', 'system'}
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)

    def saveToPickle(self, dir):
        now = time.time()
        savePickle(dir, f'{self.type}_simulation_results_{now}', self)

    def loadFromPickle(dir, filename):
        return loadPickle(dir, filename)

    def loadNewestFromPickle(dir, type):
        paths = newest(dir, f'{type}_simulation_results_')
        print(paths)
        dir, filename = os.path.split(paths[0])
        return loadPickle(dir, filename)

class TcResult:
    def __init__(self, temps, mags, abs_mags, sus, binder, T_c, T_high, T_low, graph):
        self.temps = temps
        self.mags = mags
        self.abs_mags = abs_mags
        self.sus = sus
        self.binder = binder
        self.T_c = T_c
        self.T_high = T_high
        self.T_low = T_low
        self.graph = graph

    def set_T_low(self, T_low):
        self.T_low = T_low

    def saveToPickle(self, path):
        savePickle(path, f'{self.graph}_Tc_results', self)

    def loadFromPickle(path, filename):
        return loadPickle(path, filename)

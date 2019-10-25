#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fiona Lippert
"""

import pickle, os, json, time, glob


def loadPickle(path, filename):
    """
    load a single .pickle file

    Input:
        :path: path to directory containing the .pickle file
        :filename: name of .pickle file
    Output:
        :object: unpickled objects
    """
    if not filename.endswith('.pickle'):
        filename += '.pickle'
    with open(os.path.join(path, filename), 'rb') as f:
        object = pickle.load(f)
    return object

def loadAllPickle(path, prefix):
    """
    load all .pickle files with filename starting with :prefix:

    Input:
        :path: path to directory with .pickle files
        :prefix: required prefix for .pickle file names
    Output:
        :objects: list of unpickled objects
    """
    objects = []
    for file in glob.iglob(f'{os.path.join(path, prefix)}*'):
        with open(file, 'rb') as f:
            objects.append(pickle.load(f))
    return objects

def savePickle(path, filename, objects):
    """
    save python objects as .pickle file

    Input:
        :path: path to target directory
        :filename: name of .pickle file
        :objects: python objects to be pickled
    """
    if not filename.endswith('.pickle'):
        filename += '.pickle'
    print(f'Saving {filename}')
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(objects, f, protocol = pickle.HIGHEST_PROTOCOL)

def saveSettings(targetDirectory, settings, prefix=''):
    filename = f'{prefix}Settings.json'
    print(f'Saving {filename}')
    with open(os.path.join(targetDirectory, f'{filename}'), 'w') as f:
        json.dump(settings, f)

def saveResults(targetDirectory, result_dict, name):
    with open(os.path.join(targetDirectory, f'{name}.json'), 'w') as f:
        json.dump(result_dict, f)

def loadResults(targetDirectory, name):
    with open(os.path.join(targetDirectory, f'{name}.json')) as f:
        return json.load(f)


def newest(dir, filename):
    """
    Returns sorted files by time (descending)
    """
    paths = [file for file in glob.iglob(f'{os.path.join(dir, filename)}*')]
    return sorted(paths, key=get_timestamp, reverse=True)

def get_timestamp(filepath):
    """
    Extract timestamp from SimulationResult file name
    """
    no_ext = os.path.splitext(filepath)[0]
    timestamp = no_ext.split('_')[-1]
    return timestamp


class SimulationResult:

    """
    General class for simulation results
    """

    def __init__(self, type, **kwargs):
        assert type in { 'snapshotMI', 'magMI', 'pairwiseMI', 'systemMI', 'greedy'}
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)

    def saveToPickle(self, dir, timestamp=None):
        if not timestamp:
            now = time.time()
        else:
            now = timestamp
        savePickle(dir, f'{self.type}_simulationResults_{now}', self)


    def loadFromPickle(dir, filename):
        return loadPickle(dir, filename)


    def loadNewestFromPickle(dir, type):
        paths = newest(dir, f'{type}_simulationResults_')
        dir, filename = os.path.split(paths[0])
        return loadPickle(dir, filename)

    def loadAllFromPickle(dir, type):
        ext = f'{type}_simulation_results_'
        return loadAllPickle(dir, ext)




class TempsResult:
    """
    Class for results from temperature estimation

    Attributes:
        :temps: temperature range
        :mags: average sytem magnetization per temperature
        :abs_mags: average absolute system magnetization per temperature
        :sus: susceptibility per temperature
        :binder: Binder coefficient per temperature
        :T_c: critical temperature
        :T_d: selected temperature in the disordered phase
        :T_o: selected temperature in the ordered phase
        :graph: path to networkx graph
    """

    def __init__(self, temps, mags, abs_mags, sus, binder, T_c, T_d, T_o, graph):
        self.temps = temps
        self.mags = mags
        self.abs_mags = abs_mags
        self.sus = sus
        self.binder = binder
        self.T_c = T_c
        self.T_d = T_d
        self.T_o = T_o
        self.graph = graph

    def saveToPickle(self, path):
        savePickle(path, f'{self.graph}_tempsResults', self)

    def loadFromPickle(path, filename):
        return loadPickle(path, filename)

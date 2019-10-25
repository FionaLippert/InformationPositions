#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import numpy as np
import networkx as nx
from scipy import stats, optimize

def IV(mi_per_d, max_d=-1):
    return np.nansum(mi_per_d[:max_d])

def IV_dict(mi_dict):
    return {n : IV(mi) for n, mi in mi_dict.items()}


def preprocess_snapshotMI(snapshotMI_list):
    # combine all snapshot-based MI estimates (previously loaded with IO.loadAllFromPickle()) into single dict
    # average in case of multiple data points per node
    snapshotMI_dict = {}
    counts = {}
    for part in snapshotMI_list:
        for n, vals in part.mi.items():
            if n not in snapshotMI_dict.keys():
                snapshotMI_dict[n] = vals
                counts[n] = 1
            else:
                snapshotMI_dict[n] = snapshotMI_dict[n] + vals
                counts[n] = counts[n] + 1

    for n, vals in snapshotMI_dict.items():
        snapshotMI_dict[n] = vals / counts[n]

    return snapshotMI_dict

def compare_MI_snapshot_mag(magMI_dict, snapshotMI_dict, d_max, thr=0.001):

    error_rel = np.zeros((len(snapshotMI_dict), d_max))
    error = np.zeros((len(snapshotMI_dict), d_max))
    for j, k in enumerate(snapshotMI_dict.keys()):
        v = snapshotMI_dict[k]
        v[np.where(magMI_dict[k]==0)] = np.nan
        error_rel[j,:] = np.divide(v[:d_max] - magMI_dict[k][:d_max], v[:d_max])
        error[j,:] = (v[:d_max] - magMI_dict[k][:d_max])

        # disregard data points with MI < threshold
        idx = np.logical_or(v[:d_max] < thr, MI_avg[k][:d_max] < thr)
        error_rel[j,:][idx] = np.nan

    nodes = list(snapshotMI_dict.keys())
    iv_snapshot = [IV(snapshotMI_dict[n]) for n in nodes]
    iv_mag = [IV(magMI_dict[n]) for n in nodes]
    corr, p = stats.spearmanr(iv_snapshot, iv_mag)
    print(f'spearman rank corr: {corr,p}')

    return np.abs(error), np.abs(error_rel), corr, p

def validate_IV(MI_dict, ground_truth_MI_dict, corr_type='spearman'):

    assert corr_type in ['spearman', 'pearson']

    f = lambda x, a, b: a*x + b

    nodes = list(ground_truth_MI_dict.keys())

    iv = [np.nansum(MI_dict[n]) for n in nodes]
    h = [np.mean(ground_truth_MI_dict[n]) for n in nodes]

    a, cov = optimize.curve_fit(f, np.array(iv), np.array(h))
    if corr_type == 'spearman':
        corr, p = stats.spearmanr(iv, h)
    else:
        r, p = stats.pearsonr(iv, h)
        corr = r*r

    ranks_iv = stats.rankdata(iv, method='ordinal')
    ranks_h = stats.rankdata(h, method='ordinal')
    std_diff = np.std(ranks_iv - ranks_h)
    mean_h = np.mean(h)
    std_h = np.std(h)

    slope = a[0]
    mse = np.mean(np.power(np.divide(f(np.array(iv), *a) - np.array(h), np.max(h)), 2))

    return corr, p, slope, mse, std_diff, mean_h, std_h

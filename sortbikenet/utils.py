# -*- coding: utf-8 -*-
"""
Useful functions.
"""

import numpy as np
from sklearn.metrics import auc


def get_node_positions(G):
    """Find all node positions x (longitude) and y (latitude) from the graph G and put them into a numpy array."""
    return [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G.nodes]


def get_auc(
    xx,
    yy,
    normalize_x=True,
    normalize_y=True,
    max_comparison_y="max",
    yaxis_method="bottom",
    exp_discounting=True,
    exp_gap=10,
    normalize_max_auc=True,
):
    """Get area under the curve to compare the efficiency of a curve"""
    if normalize_x:
        xx = (np.array(xx) - np.min(xx)) / (np.max(xx) - np.min(xx))
    if normalize_y:
        if yaxis_method == "bottom":
            min_y = 0
            max_y = np.max(yy)
        elif yaxis_method == "top":
            min_y = np.min(yy)
            max_y = 1
        elif yaxis_method == "both":
            min_y = 0
            max_y = 1
        elif yaxis_method == "natural":
            min_y = np.min(yy)
            max_y = np.max(yy)
        yy = (np.array(yy) - min_y) / (max_y - min_y)
    # Need to normalize by potential best curve to get AUC from 0 to 1
    if max_comparison_y == "max":
        max_yy = max(yy)
    elif max_comparison_y == "one":
        max_yy = 1
    optimal_yy = np.array([yy[0]] + [max_yy] * (len(yy) - 2) + [yy[-1]])
    if exp_discounting:
        exp_disc = np.exp(-xx * np.log(exp_gap))
        yy *= exp_disc
        optimal_yy *= exp_disc
    if normalize_max_auc:
        return auc(xx, yy) / auc(xx, optimal_yy)
    return auc(xx, yy)


def auc_from_metrics_dict(met_dict, met, **kwargs):
    """Get area under the curve for a metric in a dictionary made by growth.compute_metrics"""
    xx = met_dict["xx"]
    yy = met_dict[met]
    return get_auc(xx, yy, **kwargs)

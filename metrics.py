# -*- coding: utf-8 -*-
"""
Metrics functions natively implemented.
"""

import momepy as mp
import numpy as np
import igraph as ig
import scipy
from haversine import haversine
from .utils import get_node_positions


def coverage(G, buff_size):
    """Get coverage of the graph G."""
    return (
        mp.nx_to_gdf(G, points=False, lines=True)
        .geometry.buffer(buff_size)
        .union_all()
        .area
    )


def directness(G):
    """Get directness (impossible path are counted) of the graph G."""
    mat = get_directness_matrix(G)
    # Mean directness on all non-null value, a null value means in different components or same node
    return np.sum(mat) / (np.size(mat) - max(np.shape(mat)))


def get_directness_matrix(G, lonlat=False, weight="length"):
    """Get the symmetrical directness matrix of a graph G. If lonlat is True, node positions are in geographic CRS."""
    return _avoid_zerodiv_matrix(
        get_euclidean_distance_matrix(G, lonlat=lonlat),
        get_shortest_network_path_length_matrix(G, weight=weight),
    )


def get_shortest_network_path_length_matrix(G, weight="length"):
    """
    Get the symmetric matrix of shortest network path length of a graph G, with weight being called "length". The shortest network path length between the node i and j are in [i, j] and [j, i]. All diagonal values are 0. Value for pairs of nodes from different components is 0.

    Args:
        G (networkx.Graph): Graph on which we want to find shortest network path length for all pairs of nodes.
        weight (str, optional): Weight used in Dijkstra algorithm. Defaults to length.

    Returns:
        numpy.array: Matrix of shortest network path length for all pairs of nodes of G. Matrix with a shape (N, N), with N being the number of nodes in G
    """
    return np.array(
        ig.Graph.from_networkx(G).distances(
            source=None, target=None, weights=weight, mode="all"
        )
    )


def get_euclidean_distance_matrix(G, lonlat=False):
    """
    Get the symmetric matrix of euclidean distance for nodes on a spatial graph G. The euclidean distance between the node i and j are in [i, j] and [j, i]. All diagonal values are 0.

    Args:
        G (networkx.Graph): Graph on which we want to find the euclidean distance for all pairs of nodes.
        lonlat (bool, optional): If True, node positions are in longitude and latitude, else they are values in meters in a projection. Defaults to False.

    Returns:
        numpy.array: Matrix of euclidean distance for all pairs of nodes of G. Matrix with a shape (N, N), with N being the number of nodes in G
    """
    points = get_node_positions(G)
    if lonlat:
        dist = scipy.spatial.distance.cdist(points, points, metric=haversine)
    else:
        dist = scipy.spatial.distance.cdist(points, points, metric="euclidean")
    return np.array(dist)


def _avoid_zerodiv_matrix(num_mat, den_mat):
    """
    Divide one matrix by another while replacing numerator divided by 0 by 0.
    Example: [[1, 2],   divided by [[1, 0],    will give out [[1, 0],
              [3, 4]]               [6, 0]]                   [0.5, 0]]
    """
    return np.divide(num_mat, den_mat, out=np.zeros_like(num_mat), where=den_mat != 0)

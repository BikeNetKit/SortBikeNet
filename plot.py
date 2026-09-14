# -*- coding: utf-8 -*-
"""
Functions to visualize results of the growth of a graph.
"""

import geopandas as gpd
from matplotlib import pyplot as plt
import networkx as nx
import shapely

from .utils import get_node_positions


def plot_growth(
    G,
    growth_steps,
    folder_name,
    built=True,
    color_built="red",
    color_added="blue",
    color_newest="green",
    dpi=200,
    buffer=False,
    buff_size=200,
    buff_alpha=0.2,
    plot_metrics=False,
    growth_dir=None,
    growth_cov=None,
    growth_xx=None,
    figsize=(10, 10),
    **kwargs,
):
    """Plot the growth of the graph G in the order of the added edges from growth_steps."""
    G_init = _init_graph(G, growth_steps, built=built)
    if built:
        edge_color = {edge: color_built for edge in G_init.edges}
    else:
        edge_color = {edge: color_newest for edge in G_init.edges}
    # Make first plot only to get a fixed bounding box for plots, being the one for the final graph
    fig, ax = plot_graph(G, show=False, save=False, close=True, dpi=dpi)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bb_graph = [ylim[0], ylim[1], xlim[0], xlim[1]]
    if buffer:
        bb_graph = [
            bb_graph[0] - buff_size,
            bb_graph[1] + buff_size,
            bb_graph[2] - buff_size,
            bb_graph[3] + buff_size,
        ]
    pad_name = len(str(len(G.edges)))
    if built:
        bci = color_added
    else:
        bci = color_newest
    if plot_metrics:
        fig, axs = plt.subplots(1, 4, figsize=(figsize[0], figsize[1] * 4))
        ax = axs[0]
        for idx, met in enumerate([growth_cov, growth_dir]):
            a = axs[idx + 1]
            a.set_xlim(0, max(growth_xx) * 1.1)
            pad = (max(met) - min(met)) * 0.1
            a.set_ylim(min(met) - pad, max(met) + pad)
            a.plot([growth_xx[0]], [met[0]], marker="o")
    else:
        fig, ax = _init_fig(ax=None, figsize=figsize)
    fig, ax = plot_graph(
        G_init,
        ax=ax,
        edge_color=edge_color,
        save=False,
        show=False,
        bbox=bb_graph,
        close=False,
        dpi=dpi,
        buffer=buffer,
        buff_size=buff_size,
        buff_alpha=buff_alpha,
        buff_color=bci,
        **kwargs,
    )
    if plot_metrics:
        plt.tight_layout()
    _show_save_close(
        fig,
        show=False,
        save=True,
        close=True,
        filepath=folder_name + f"/step_{0:0{pad_name}}.png",
        dpi=dpi,
    )
    if not built:
        edge_color = {edge: color_added for edge in G_init.edges}
    actual_edges = list(G_init.edges)
    old_edge = None
    if buffer:
        geom = [G.edges[edge]["geometry"].buffer(buff_size) for edge in G_init.edges]
    for ids, edge in enumerate(growth_steps):
        actual_edges.append(edge)
        edge_color[edge] = color_newest
        # Replace color of previous edge from being the newest to being an added one
        if old_edge is not None:
            edge_color[old_edge] = color_added
        G_actual = G.edge_subgraph(actual_edges)
        if plot_metrics:
            fig, axs = plt.subplots(1, 4, figsize=(figsize[0], figsize[1] * 4))
            ax = axs[0]
        else:
            fig, ax = _init_fig(ax=None, figsize=figsize)
        if buffer:
            fig, ax = plot_graph(
                G_actual,
                ax=ax,
                edge_color=edge_color,
                buffer=buffer,
                buff_size=buff_size,
                buff_alpha=buff_alpha,
                save=False,
                show=False,
                bbox=bb_graph,
                close=False,
                dpi=dpi,
                **kwargs,
            )
            buff_bef = gpd.GeoSeries(geom).union_all()
            geom.append(G.edges[edge]["geometry"].buffer(buff_size))
            buff_aft = gpd.GeoSeries(geom).union_all()
            diff = shapely.difference(buff_aft, buff_bef)
            if diff.area > 0:
                buff_added = gpd.GeoSeries(diff)
                buff_added.plot(ax=ax, color=color_newest, alpha=buff_alpha, zorder=1)
        else:
            fig, ax = plot_graph(
                G_actual,
                ax=ax,
                edge_color=edge_color,
                save=False,
                show=False,
                bbox=bb_graph,
                close=False,
                dpi=dpi,
                **kwargs,
            )
        if plot_metrics:
            for idx, met in enumerate([growth_cov, growth_dir]):
                a = axs[idx + 1]
                a.set_xlim(0, max(growth_xx) * 1.1)
                pad = (max(met) - min(met)) * 0.1
                a.set_ylim(min(met) - pad, max(met) + pad)
                a.plot(growth_xx[: ids + 2], met[: ids + 2], marker="o")
            plt.tight_layout()
        _show_save_close(
            fig,
            show=False,
            save=True,
            close=True,
            filepath=folder_name + f"/step_{ids + 1:0{pad_name}}.png",
            dpi=dpi,
        )
        old_edge = edge


def plot_graph(
    G,
    edge_linewidth=2,
    node_size=6,
    edge_color="steelblue",
    node_color="black",
    ax=None,
    figsize=(16, 9),
    show=True,
    save=False,
    close=False,
    filepath=None,
    dpi=200,
    bbox=None,
    buffer=False,
    buff_size=200,
    buff_color="steelblue",
    buff_alpha=0.2,
):
    """Plot the graph G with some specified matplotlib parameters, using Geopandas."""
    fig, ax = _init_fig(ax=ax, figsize=figsize)
    if bbox is not None:
        ax.set_ylim(bbox[0], bbox[1])
        ax.set_xlim(bbox[2], bbox[3])
    geom_node = [shapely.Point(x, y) for x, y in get_node_positions(G)]
    geom_edge = list(nx.get_edge_attributes(G, "geometry").values())
    if isinstance(edge_color, dict):
        edgeidx = [edge for edge in G.edges]
        new_edge_color = {}
        for edge in edge_color:
            if edge not in edgeidx:
                if len(edge) == 3:
                    reverse = tuple([edge[1], edge[0], edge[2]])
                else:
                    reverse = tuple(reversed(edge))
                if reverse in edgeidx:
                    new_edge_color[reverse] = edge_color[edge]
                else:
                    raise ValueError(f"{edge} in edge_color is not in G")
            else:
                new_edge_color[edge] = edge_color[edge]
        edge_color = new_edge_color
    gdf_node = gpd.GeoDataFrame(index=[node for node in G.nodes], geometry=geom_node)
    gdf_node = gdf_node.assign(color=node_color)
    gdf_edge = gpd.GeoDataFrame(index=[edge for edge in G.edges], geometry=geom_edge)
    gdf_edge = gdf_edge.assign(color=edge_color)
    gdf_edge.plot(ax=ax, color=gdf_edge["color"], zorder=2, linewidth=edge_linewidth)
    gdf_node.plot(ax=ax, color=gdf_node["color"], zorder=3, markersize=node_size)
    if buffer:
        buff = gpd.GeoSeries(gdf_edge.geometry.buffer(buff_size).unary_union)
        buff.plot(ax=ax, color=buff_color, alpha=buff_alpha, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    _show_save_close(fig, show=show, save=save, close=close, filepath=filepath, dpi=dpi)
    return fig, ax


def _init_fig(ax=None, figsize=(16, 9)):
    """Initialize the matplotlib figure if not already given."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.get_figure()
    return fig, ax


def _show_save_close(fig, show=True, save=False, close=False, filepath=None, dpi=1000):
    """Show, save, and close the given figure."""
    if show:
        plt.show()
    if save:
        fig.savefig(filepath, dpi=dpi)
    if close:
        plt.close()


def _init_graph(G, growth_steps, built=True):
    if built:
        init_edges = [edge for edge in G.edges if G.edges[edge]["built"] == 1]
    # Find initial edges if not built by finding the ones that are not on the growth steps. Supposedly it's a single random edge from the highest closeness node, see growth.order_network_growth
    else:
        init_edges = [edge for edge in G.edges if edge not in growth_steps]
    return G.edge_subgraph(init_edges)

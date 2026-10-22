# -*- coding: utf-8 -*-
"""
Class to grow a graph.
"""

import numpy as np
import networkx as nx
import momepy as mp
import osmnx as ox
import shapely
from metrics import directness


class SortBike:
    def __init__(
        self,
        G,
        preset="random",
        buff_size=200,
        built=False,
        keep_connected=True,
        seed=None,
    ):
        self.G = G.copy()
        self.gdf_edges = mp.nx_to_gdf(self.G, points=False, lines=True)
        self.preset = preset
        self.preset_dict = self._add_preset_dict()
        self.main_func, self.upda_func, self.type_func = self._use_preset()
        self.buff_size = buff_size
        self.seed = seed
        self.random_generator = np.random.default_rng(seed=self.seed)
        self.keep_connected = keep_connected
        self.edge_to_id = {edge: idx for idx, edge in enumerate(self.G.edges)}
        self.id_to_edge = {idx: edge for idx, edge in enumerate(self.G.edges)}
        self.built = built
        self.hierarchy_level = 0
        if self.built:
            self.G_init = self._build_graph(preset)
        else:
            self.G_init = self._init_graph(preset)
        self.G_actual = self.G_init.copy()
        self.edges_actual = list(self.G_actual.edges)
        self.actual_buff_geom = (
            mp.nx_to_gdf(self.G_actual, points=False, lines=True)
            .buffer(buff_size)
            .union_all()
        )
        self.max_area = self.gdf_edges.buffer(buff_size).union_all().area
        self.num_step = len(self.G.edges) - len(self.G_init.edges)
        self.directness = 0
        self.xx = 0
        self.num_cc = 0
        self.length_lcc = 0
        self.metrics_dict = {
            "xx": [],
            "directness": [],
            "coverage": [],
            "num_cc": [],
            "length_lcc": [],
        }
        self._compute_metrics()
        self.growth_order = []
        self.edges_tested = []
        self.values_found = []
        self.G_tested = None
        self.grown = False
        self.keep_searching = True

    @staticmethod
    def download_osmnx(city_query, network_type="all_public", simplify=True, ):
        """ Download a street network from OpenStreetMap using OSMnx.
         Parameters
         ----------
         city_query : str Place name understood by OSMnx, e.g. "Copenhagen, Denmark".
         network_type : str OSMnx network type, e.g. "bike", "walk", or "drive".
         simplify : bool Whether OSMnx should simplify the downloaded graph.

         Returns
         -------
         nx.MultiDiGraph The raw OSMnx graph. """

        for custom_tag in ["highway", "cycleway", "bicycle", "cycleway:right", "cycleway:left", "cycleway:both",
                           "cyclestreet", "access", "area", "service", "motor_vehicle",
                           "motorcar"]:  # This list should contain all tags used in any custom filters
            if custom_tag not in ox.settings.useful_tags_way:
                ox.settings.useful_tags_way.extend(custom_tag)

        custom_filter = ['["highway"~"cycleway|living_street"]',
    '["highway"~"path|pedestrian|footway"]["bicycle"~"designated|yes|permissive"]["access"!~"private"]',
    '["cyclestreet"~"yes"]',
    '["bicycle_road"~"yes"]',
    '["cycleway"~"track"]',
    '["cycleway:right"~"track|opposite_track"]', # opposite_track is deprecated, but could still exist
    '["cycleway:left"~"track|opposite_track"]', # opposite_track is deprecated, but could still exist
    '["cycleway:both"~"track|opposite_track"]', # opposite_track is deprecated, but could still exist
]

        g = ox.graph_from_place(
            city_query, network_type=network_type, custom_filter=custom_filter, retain_all=False, simplify=simplify
        )

        return g

    @staticmethod
    def _prepare_osmnx_graph(G):
        """ Prepare an OSMnx graph for use with Orderbike. The graph is converted to an undirected graph and the edge attributes required by Orderbike are added.
        Parameters
        ----------
        G : nx.MultiDiGraph Graph downloaded with OSMnx.

        Returns
        -------
        nx.MultiGraph Prepared graph. """
        G = ox.projection.project_graph(G)
        G = ox.convert.to_undirected(G)

        gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs( G, nodes=True, edges=True, fill_edge_geometry=True, )

        if "length" not in gdf_edges.columns: gdf_edges["length"] = gdf_edges.geometry.length

        # Sortbike expects a "built" attribute.
        # By default, none of the downloaded network is considered
        # initially built.
        gdf_edges["built"] = 0

        # This is a provisional mapping for road hierarchies.
        hierarchy_map = {"primary": 4,
                          "secondary": 3,
                          "tertiary": 2,
                          "residential": 1,
                          "living_street": 1,
                          "service": 1,
                          "unclassified": 1, }
        def get_highway_value(value):
            if isinstance(value, list):
                return value[0]
            return value

        if "highway" in gdf_edges.columns:
            gdf_edges["highway"] = gdf_edges["highway"].apply(
                get_highway_value
            )
            gdf_edges["hierarchy"] = (
                gdf_edges["highway"]
                .map(hierarchy_map)
                .fillna(1)
                .astype(int)
            )
        else:
            # Fall back to the lowest hierarchy level if OSM data does not contain highway information.
            gdf_edges["hierarchy"] = 1

        # Reconstruct the NetworkX graph with the modified edge attributes.
        G = ox.convert.graph_from_gdfs(
            gdf_nodes,
            gdf_edges,
            graph_attrs=G.graph,
        )
        # make Graph undirected again
        G = nx.MultiGraph(G)

        return G

    @classmethod
    def from_osmnx(
            cls,
            city_query,
            network_type="public_all",
            preset="random",
            buff_size=200,
            keep_connected=True,
            seed=None,
            simplify=True,
    ):
        """ Download and prepare an OSMnx network for Orderbike.
        This is method combines download_osmnx()
        and _prepare_osmnx_graph(). """

        G = cls.download_osmnx(city_query=city_query, network_type=network_type, simplify=simplify, )
        G = cls._prepare_osmnx_graph(G)
        return cls(G, preset=preset, buff_size=buff_size, built=False, keep_connected=keep_connected, seed=seed, )

    def _add_preset_dict(self):
        return {
            "directness": {
                "main_func": self.main_directness,
                "upda_func": None,
                "type_func": "dynamic",
            },
            "coverage": {
                "main_func": self.main_coverage,
                "upda_func": self.upda_coverage,
                "type_func": "dynamic",
            },
            "betweenness": {
                "main_func": self.main_betweenness,
                "upda_func": None,
                "type_func": "static",
            },
            "dual_betweenness": {
                "main_func": self.main_dual_betweenness,
                "upda_func": None,
                "type_func": "static",
            },
            "closeness": {
                "main_func": self.main_closeness,
                "upda_func": None,
                "type_func": "static",
            },
            "dual_closeness": {
                "main_func": self.main_dual_closeness,
                "upda_func": None,
                "type_func": "static",
            },
            "hierarchy": {
                "main_func": self.main_hierarchy,
                "upda_func": None,
                "type_func": "static",
            },
            "hierarchy_coverage": {
                "main_func": self.main_hierarchy_coverage,
                "upda_func": self.upda_hierarchy_coverage,
                "type_func": "dynamic",
            },
            "hierarchy_directness": {
                "main_func": self.main_hierarchy_directness,
                "upda_func": self.upda_hierarchy_directness,
                "type_func": "dynamic",
            },
            "random": {
                "main_func": self.main_random,
                "upda_func": None,
                "type_func": "static",
            },
        }

    def _use_preset(self):
        "Use a preset of functions instead of manually entering each of them."
        if self.preset not in self.preset_dict:
            raise ValueError(
                f"Preset not existing! Please choose between the following options: {list(self.preset_dict.keys())}"
            )
        else:
            return (
                self.preset_dict[self.preset]["main_func"],
                self.preset_dict[self.preset]["upda_func"],
                self.preset_dict[self.preset]["type_func"],
            )

    def _build_graph(self, preset):
        "Use the built argument to initialize the graph with only a specific set of edges."
        if preset in ["hierarchy", "hierarchy_coverage", "hierarchy_directness"]:
            self.hierarchy_level = int(self.gdf_edges["hierarchy"].max())
        return self.G.edge_subgraph(
            [edge for edge in self.G.edges if self.G.edges[edge]["built"] == 1]
        )

    def _init_graph(self, preset):
        "Initialize the graph with the edge of highest average closeness centrality."
        closeness = nx.closeness_centrality(self.G, distance="length")
        if preset in ["hierarchy", "hierarchy_coverage", "hierarchy_directness"]:
            max_hierarchy = int(self.gdf_edges["hierarchy"].max())
            self.hierarchy_level = max_hierarchy
            edge_closeness = {
                edge: (closeness[edge[0]] + closeness[edge[1]]) / 2
                for edge in self.G.edges
                if self.G.edges[edge]["hierarchy"] == max_hierarchy
            }
        else:
            edge_closeness = {
                edge: (closeness[edge[0]] + closeness[edge[1]]) / 2
                for edge in self.G.edges
            }
        return self.G.edge_subgraph(
            [tuple(max(edge_closeness, key=edge_closeness.get))]
        )

    def _compute_metrics(self):
        self.directness = directness(self.G_actual)
        self.xx = sum([self.G_actual.edges[e]["length"] for e in self.edges_actual])
        self.actual_area = self.actual_buff_geom.area
        cc = list(nx.connected_components(self.G_actual))
        self.num_cc = len(cc)
        self.length_lcc = max(
            [
                sum(
                    [
                        self.G_actual.edges[e]["length"]
                        for e in self.G_actual.subgraph(comp).edges
                    ]
                )
                for comp in cc
            ]
        )
        self.metrics_dict["directness"].append(self.directness)
        self.metrics_dict["coverage"].append(self.actual_area)
        self.metrics_dict["xx"].append(self.xx)
        self.metrics_dict["num_cc"].append(self.num_cc)
        self.metrics_dict["length_lcc"].append(self.length_lcc)

    def grow(self):
        "Grow the network."
        if self.type_func == "dynamic":
            self._dynamic_growth()
        elif self.type_func == "static":
            self._static_growth()
        else:
            raise ValueError(
                "Wrong type_func value. type_func can only be either dynamic or static."
            )

    def _dynamic_growth(self):
        "Grow the network using a dynamic metric, recomputing metrics at each step for each valid edge."
        self.growth_order = []
        for _ in range(self.num_step):
            self.edges_tested = []
            self.values_found = []
            self.valid_edges = self._find_valid_edges()
            for edge in self.valid_edges:
                self.edges_tested.append(edge)
                self.G_tested = self.G.edge_subgraph(self.edges_actual + [edge])
                self.values_found.append(self.main_func())
            self._update_status(self._find_optimal_edge())
            if self.upda_func is not None:
                self.upda_func()
        self.grown = True

    def _find_optimal_edge(self):
        "Find the optimal edge."
        m = max(self.values_found)
        optimum = [
            edge for edge, val in zip(self.edges_tested, self.values_found) if val == m
        ]
        # When more than one optimal value, return a random value from all the optimal ones
        if len(optimum) > 1:
            # Need to change to list with built-in function to avoid numpy int type for values, and then to tuple for the list
            return optimum[self.random_generator.choice(len(optimum))]
        return optimum[0]

    def _static_growth(self):
        "Grow the network using a static metric, following the order for each valid edge."
        self.growth_order = []
        res = self.main_func()
        self.edges_tested = list(res.keys())
        self.values_found = np.array(list(res.values()))
        for _ in range(self.num_step):
            self.valid_edges = self._find_valid_edges()
            self._update_status(self._find_highest_rank())
        self.grown = True

    def _find_valid_edges(self):
        "Find all valid edges to add for a given step. For now work only if the final network do not have a component without built part."
        if self.keep_connected:
            return [
                edge
                for edge in self.G.edges
                if not (
                    (not any(node in self.G_actual for node in edge[:2]))
                    or edge in self.G_actual.edges
                )
            ]
        return [edge for edge in self.G.edges if edge not in self.G_actual.edges]

    def _find_highest_rank(self):
        for idx, edge in enumerate(self.edges_tested):
            if edge in self.valid_edges:
                value = self.values_found[idx]
                if len(np.where(self.values_found == value)[0]) > 1:
                    cand = [
                        self.edges_tested[pos]
                        for pos in np.where(self.values_found == value)[0]
                        if self.edges_tested[pos] in self.valid_edges
                    ]
                    return cand[self.random_generator.choice(len(cand))]
                return edge

    def _update_status(self, choice):
        self.growth_order.append(choice)
        self.edges_actual.append(choice)
        self.G_actual = self.G.edge_subgraph(self.edges_actual)
        self.actual_buff_geom = shapely.unary_union(
            [
                self.actual_buff_geom,
                self.G.edges[choice]["geometry"].buffer(self.buff_size),
            ]
        )
        self._compute_metrics()

    def get_metrics_dict(self):
        return self.metrics_dict

    def get_growth_order(self):
        return self.growth_order

    def main_directness(self):
        return directness(self.G_tested)

    def main_coverage(self):
        if self.keep_searching:
            edge = self.edges_tested[-1]
            return (
                shapely.unary_union(
                    [
                        self.actual_buff_geom,
                        self.G.edges[edge]["geometry"].buffer(self.buff_size),
                    ]
                ).area
                - self.actual_area
            ) / self.G.edges[edge]["length"]
        return 0

    def upda_coverage(self):
        if self.actual_area == self.max_area:
            self.keep_searching = False

    def main_betweenness(self):
        return {
            key: val
            for key, val in sorted(
                nx.edge_betweenness_centrality(self.G, weight="length").items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def main_dual_betweenness(self):
        coins = mp.COINS(self.gdf_edges.copy())
        G_dual = mp.coins_to_nx(coins)
        nx.set_node_attributes(
            G_dual,
            dict(nx.betweenness_centrality(G_dual)),
            "stroke_betweenness",
        )
        strokes, _ = mp.nx_to_gdf(G_dual)
        strokes = strokes.explode("edge_indices").set_index("edge_indices")
        gdf_cont = self.gdf_edges.copy()
        gdf_cont["stroke_betweenness"] = strokes["stroke_betweenness"]
        H = mp.gdf_to_nx(gdf_cont, integer_labels=False, preserve_index=True)
        return {
            key: val
            for key, val in sorted(
                nx.get_edge_attributes(H, "stroke_betweenness").items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def main_closeness(self):
        nclo = nx.closeness_centrality(self.G, distance="length")
        return {
            key: val
            for key, val in sorted(
                {e: (nclo[e[0]] + nclo[e[1]]) / 2 for e in self.G.edges}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def main_dual_closeness(self):
        coins = mp.COINS(self.gdf_edges.copy())
        G_dual = mp.coins_to_nx(coins)
        nx.set_node_attributes(
            G_dual,
            dict(nx.closeness_centrality(G_dual)),
            "stroke_closeness",
        )
        strokes, _ = mp.nx_to_gdf(G_dual)
        strokes = strokes.explode("edge_indices").set_index("edge_indices")
        gdf_cont = self.gdf_edges.copy()
        gdf_cont["stroke_closeness"] = strokes["stroke_closeness"]
        H = mp.gdf_to_nx(gdf_cont, integer_labels=False, preserve_index=True)
        return {
            key: val
            for key, val in sorted(
                nx.get_edge_attributes(H, "stroke_closeness").items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def main_hierarchy(self):
        return {
            key: val
            for key, val in sorted(
                {
                    edge: self.G.edges[edge]["hierarchy"] for edge in self.G.edges
                }.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def main_hierarchy_coverage(self):
        edge = self.edges_tested[-1]
        if self.G.edges[edge]["hierarchy"] < self.hierarchy_level:
            if any(
                self.gdf_edges.iloc[[self.edge_to_id[idx] for idx in self.valid_edges]][
                    "hierarchy"
                ]
                == self.hierarchy_level
            ):
                return 0
            else:
                lower_hierarchy = self.hierarchy_level - 1
                while not any(
                    self.gdf_edges.iloc[
                        [self.edge_to_id[idx] for idx in self.valid_edges]
                    ]["hierarchy"]
                    == lower_hierarchy
                ):
                    lower_hierarchy = lower_hierarchy - 1
                if self.G.edges[edge]["hierarchy"] < lower_hierarchy:
                    return 0
                else:
                    return (
                        shapely.unary_union(
                            [
                                self.actual_buff_geom,
                                self.G.edges[edge]["geometry"].buffer(self.buff_size),
                            ]
                        ).area
                        - self.actual_area
                    ) / self.G.edges[edge]["length"]
        elif self.keep_searching:
            return (
                shapely.unary_union(
                    [
                        self.actual_buff_geom,
                        self.G.edges[edge]["geometry"].buffer(self.buff_size),
                    ]
                ).area
                - self.actual_area
            ) / self.G.edges[edge]["length"]
        else:
            return 1

    def upda_hierarchy_coverage(self):
        gdf_edges_actual = mp.nx_to_gdf(self.G_actual, points=False, lines=True)
        if len(
            gdf_edges_actual[gdf_edges_actual["hierarchy"] == self.hierarchy_level]
        ) == len(self.gdf_edges[self.gdf_edges["hierarchy"] == self.hierarchy_level]):
            self.hierarchy_level = self.hierarchy_level - 1
        if self.actual_area == self.max_area:
            self.keep_searching = False

    def main_hierarchy_directness(self):
        edge = self.edges_tested[-1]
        if self.G.edges[edge]["hierarchy"] < self.hierarchy_level:
            if any(
                self.gdf_edges.iloc[[self.edge_to_id[idx] for idx in self.valid_edges]][
                    "hierarchy"
                ]
                == self.hierarchy_level
            ):
                return 0
            else:
                lower_hierarchy = self.hierarchy_level - 1
                while not any(
                    self.gdf_edges.iloc[
                        [self.edge_to_id[idx] for idx in self.valid_edges]
                    ]["hierarchy"]
                    == lower_hierarchy
                ):
                    lower_hierarchy = lower_hierarchy - 1
                if self.G.edges[edge]["hierarchy"] < lower_hierarchy:
                    return 0
                else:
                    return directness(self.G_tested)
        return directness(self.G_tested)

    def upda_hierarchy_directness(self):
        gdf_edges_actual = mp.nx_to_gdf(self.G_actual, points=False, lines=True)
        if len(
            gdf_edges_actual[gdf_edges_actual["hierarchy"] == self.hierarchy_level]
        ) == len(self.gdf_edges[self.gdf_edges["hierarchy"] == self.hierarchy_level]):
            self.hierarchy_level = self.hierarchy_level - 1

    def main_random(self):
        edgelist = list(self.G.edges)
        self.random_generator.shuffle(edgelist)
        return {key: -idx for idx, key in enumerate(edgelist)}

    def to_geodataframe(self):
        """
        Export the final grown network as a GeoDataFrame

        Parameters
        ----------

        Returns
        -------
        geopandas.GeoDataFrame
            Edges of the grown network, including their attributes
            and growth step.
        """

        # Convert the final grown graph to an edge GeoDataFrame.
        gdf = ox.convert.graph_to_gdfs(
            self.G_actual,
            nodes=False,
            edges=True,
            fill_edge_geometry=True,
        )

        # Record the order in which edges were added.
        # Step 0 represents the initial edge; subsequent steps follow
        # the order recorded by the growth algorithm.
        growth_steps = {
            frozenset(edge[:2]): step
            for step, edge in enumerate(self.growth_order, start=1)
        }

        gdf["growth_step"] = [
            growth_steps.get(frozenset((u, v)), 0)
            for u, v in zip(
                gdf.index.get_level_values("u"),
                gdf.index.get_level_values("v"),
            )
        ]

        # Reset the index so u, v, and key are ordinary columns.
        gdf = gdf.reset_index()

        return gdf

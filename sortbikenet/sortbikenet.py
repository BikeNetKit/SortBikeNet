
"""
Run the Sortbike growth process on an OpenStreetMap network.

Edit the CONFIGURATION section below before running. The script is deliberately
non-interactive: it does not prompt for input.

Requirements:
    pip install osmnx geopandas networkx numpy momepy shapely

Project files expected beside this script:
    growth.py
    metrics.py

Run:
    python sortbikenet.py
"""

from pathlib import Path
import sys

from growth import SortBike

# =============================================================================
# CONFIGURATION — edit these values before running
# =============================================================================

# Place query understood by OpenStreetMap / OSMnx.
CITY_QUERY = "Paris"

# OSMnx network type"
NETWORK_TYPE = "all_public"

# Must be one of the presets defined in SortBike._add_preset_dict():
# "random", "directness", "coverage", "betweenness", "dual_betweenness",
# "closeness", "dual_closeness", "hierarchy", "hierarchy_coverage",
# "hierarchy_directness"
PRESET = "betweenness"

# Buffer distance in metres. The graph is projected before SortBike uses it.
BUFFER_SIZE = 200

# Whether to constrain growth to edges adjacent to the current grown network.
KEEP_CONNECTED = True

# Integer for reproducible random choices, or None for a non-fixed seed.
RANDOM_SEED = 42

# Simplify the downloaded OSM network.
SIMPLIFY = True

# Output files. Use None to skip the corresponding output.
GEOJSON_PATH = "grown_network.geojson"

# Print the first N rows of the resulting GeoDataFrame.
PREVIEW_ROWS = 5

# =============================================================================
# RUN
# =============================================================================

VALID_PRESETS = {
    "random",
    "directness",
    "coverage",
    "betweenness",
    "dual_betweenness",
    "closeness",
    "dual_closeness",
    "hierarchy",
    "hierarchy_coverage",
    "hierarchy_directness",
}


def main() -> int:
    if PRESET not in VALID_PRESETS:
        raise ValueError(
            f"Unknown PRESET={PRESET!r}. Choose one of: "
            + ", ".join(sorted(VALID_PRESETS))
        )

    try:
        # from_osmnx uses the download and graph-preparation methods already
        # defined in the uploaded growth.py.
        sortbike = SortBike.from_osmnx(
            city_query=CITY_QUERY,
            network_type=NETWORK_TYPE,
            preset=PRESET,
            buff_size=BUFFER_SIZE,
            keep_connected=KEEP_CONNECTED,
            seed=RANDOM_SEED,
            simplify=SIMPLIFY,
        )

        print(
            "Input graph prepared: "
            f"{sortbike.G.number_of_nodes():,} nodes, "
            f"{sortbike.G.number_of_edges():,} edges"
        )

        print("Running growth process...")
        sortbike.grow()

        # growth.py's current to_geodataframe() returns the GeoDataFrame.
        # GeoJSON is saved here because that method currently takes no path.
        gdf_result = sortbike.to_geodataframe()

        print("\nGrowth complete.")
        print(f"Output edges: {len(gdf_result):,}")
        print(f"CRS: {gdf_result.crs}")
        print("\nGeoDataFrame preview:")
        print(gdf_result.head(PREVIEW_ROWS).to_string(index=False))

        if GEOJSON_PATH:
            geojson_path = Path(GEOJSON_PATH)
            geojson_path.parent.mkdir(parents=True, exist_ok=True)
            # GeoJSON should use WGS84 longitude/latitude coordinates.
            gdf_result.to_crs("EPSG:4326").to_file(
                geojson_path, driver="GeoJSON"
            )
            print(f"GeoJSON saved to: {geojson_path.resolve()}")

        print("\nFinal metrics:")
        for metric, values in sortbike.get_metrics_dict().items():
            if values:
                print(f"  {metric}: {values[-1]}")

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "\nCheck the place/network settings, installed dependencies, and "
            "that growth.py and metrics.py are importable.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


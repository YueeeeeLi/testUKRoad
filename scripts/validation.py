# %%
from pathlib import Path
import geopandas as gpd
import pandas as pd
from utils import load_config
import warnings

pd.set_option("display.max_columns", None)
warnings.simplefilter("ignore")

incoming_path = Path(load_config()["paths"]["incoming_data"])
base_path = Path(load_config()["paths"]["base_path"])

# %%
dft_file = pd.read_csv(
    incoming_path / "road" / "mrdb-traffic counts" / "dft_traffic_counts_aadf_2021.csv"
)
# cars and buses
dft_file["commuters_vehicles"] = (
    dft_file["Cars_and_taxis"] + dft_file["Buses_and_coaches"]
)

# major roads - attributes (2021)
major_traffic_counts = dft_file[dft_file["Road_type"] == "Major"]
major_traffic_counts.reset_index(drop=True, inplace=True)
major_traffic_counts_gdf = gpd.GeoDataFrame(
    major_traffic_counts,
    geometry=gpd.points_from_xy(
        x=major_traffic_counts.Longitude, y=major_traffic_counts.Latitude
    ),
    crs="4326",
)
major_traffic_counts_dict = major_traffic_counts.set_index("Count_point_id")[
    "commuters_vehicles"
].to_dict()

# %%
# major roads - shapefile (2021)
major_roads = gpd.read_file(incoming_path / "road" / "mrdb-2021" / "MRDB_2021.shp")
major_roads.CP_Number = major_roads.CP_Number.astype(int)

# %%
# Attach attribute nodes to major roads shapefile
major_roads["commuters_vehicles"] = major_roads["CP_Number"].map(
    major_traffic_counts_dict
)

# %%
major_roads.to_parquet(
    incoming_path / "road" / "mrdb-traffic counts" / "dft_major_roads.geoparquet"
)

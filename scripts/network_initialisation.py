# %%
from pathlib import Path

import geopandas as gpd
import numpy as np

import json
from utils import load_config

output_path = Path(load_config()["paths"]["outputs"])
base_path = Path(load_config()["paths"]["base_path"])
# %%
road_link_file = gpd.read_parquet(
    base_path / "networks" / "road" / "road_link_file.geoparquet"
)
observation = gpd.read_parquet(
    base_path / "networks" / "road" / "link_traffic_counts.geoparquet"
)
with open(base_path / "parameters" / "free_flow_speed_dict.json", "r") as f:
    free_flow_speed_dict = json.load(f)

# %%
# initialisation
road_link_file.drop(columns=["average_toll_cost"], inplace=True)
road_link_file.insert(len(road_link_file.columns) - 1, "combined_label", np.nan)
# %%
road_link_file.loc[
    road_link_file.road_classification == "Motorway", "combined_label"
] = "M"
road_link_file.loc[road_link_file.road_classification == "B Road", "combined_label"] = (
    "B"
)
road_link_file.loc[road_link_file.road_classification == "A Road", "combined_label"] = (
    "A_single"
)
road_link_file.loc[
    (
        (road_link_file.road_classification == "A Road")
        & (
            road_link_file.form_of_way.isin(
                ["Dual Carriageway", "Collapsed Dual Carriageway"]
            )
        )
    ),
    "combined_label",
] = "A_dual"

# %%
observation.loc[observation.road_classification == "Motorway", "combined_label"] = "M"
observation.loc[observation.road_classification == "B Road", "combined_label"] = "B"
observation.loc[observation.road_classification == "A Road", "combined_label"] = (
    "A_single"
)
observation.loc[
    (
        (observation.road_classification == "A Road")
        & (
            observation.form_of_way.isin(
                ["Dual Carriageway", "Collapsed Dual Carriageway"]
            )
        )
    ),
    "combined_label",
] = "A_dual"

# %%
observation["capacities"] = observation["Cars_and_taxis"]
observation.loc[
    (observation.capacities.isnull()) & (observation.combined_label == "M"),
    "capacities",
] = 205_664

observation.loc[
    (observation.capacities.isnull()) & (observation.combined_label == "A_dual"),
    "capacities",
] = 184_546


observation.loc[
    (observation.capacities.isnull()) & (observation.combined_label == "A_single"),
    "capacities",
] = 111_578

observation.loc[
    (observation.capacities.isnull()) & (observation.combined_label == "B"),
    "capacities",
] = 162_949

obs_dict = observation.set_index("e_id")["capacities"].to_dict()


# %%
road_link_file["acc_flow"] = 0.0
road_link_file["acc_capacity"] = road_link_file.e_id.map(obs_dict)
road_link_file["ave_flow_rate"] = road_link_file.combined_label.map(
    free_flow_speed_dict
)

# %%
road_link_file.to_parquet(
    base_path / "networks" / "road" / "road_link_file_updated.geoparquet"
)

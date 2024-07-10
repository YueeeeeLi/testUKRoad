# %%
from pathlib import Path
import geopandas as gpd
import pandas as pd
import random
from utils import load_config
from collections import defaultdict
from tqdm.auto import tqdm
import pickle
import warnings

warnings.simplefilter("ignore")

incoming_path = Path(load_config()["paths"]["incoming_data"])
base_path = Path(load_config()["paths"]["base_path"])

# %%
# UK buildings
verisk = gpd.read_file(
    incoming_path / "census" / "VERISK_gb" / "data" / "edition_15_online_version.gpkg",
    engine="pyogrio",
)  # geometry: polygon (5mins)

# Roads
road_node_file = gpd.read_parquet(
    base_path / "networks" / "road" / "road_node_file.geoparquet"
)  # nd_id

# OA (2021)
admin_file = gpd.read_parquet(
    base_path
    / "census_datasets"
    / "admin_census_boundary_stats"
    / "gb_oa_2021_estimates.geoparquet",
)  # OA21CD

# OD Matrix at OA level (2021)
od_file = pd.read_csv(base_path / "census_datasets" / "od_matrix" / "od_gb_oa_2021.csv")
od_file.rename(columns={"Car21": "Count21"}, inplace=True)  # 11,143,891 records
od_file.reset_index(drop=True, inplace=True)

rule_file = pd.read_csv(
    incoming_path / "census" / "VERISK_gb" / "reclassification_rules.csv",
)
class_dict = rule_file.set_index("use")["class"].to_dict()
uniheight_dict = rule_file.set_index("use")["unit_height_m"].to_dict()


# %%
# attach OD trips to road nodes
# based on OA centroids (in general, not population-based centroids)
def find_nearest_node_centroid(
    zones: gpd.GeoDataFrame, road_nodes: gpd.GeoDataFrame
) -> dict:
    nearest_node_dict = {}  # node_idx: zone_idx
    for zidx, z in tqdm(zones.iterrows(), desc="Processing", total=zones.shape[0]):
        closest_road_node = road_nodes.sindex.nearest(z.geometry, return_all=False)[1][
            0
        ]
        # for the first [x]:
        #   [0] represents the index of geometry;
        #   [1] represents the index of gdf
        # the second [x] represents the No. of closest item in the returned list,
        #   which only return one nearest node in this case
        nearest_node_dict[zidx] = closest_road_node

    zone_to_node = {}
    for zidx in range(zones.shape[0]):
        z = zones.loc[zidx, "OA21CD"]
        nidx = nearest_node_dict[zidx]
        n = road_nodes.loc[nidx, "nd_id"]
        zone_to_node[z] = n

    return zone_to_node


# based on gross area of building footprint
def find_nearest_node(zones: gpd.GeoDataFrame, road_nodes: gpd.GeoDataFrame) -> dict:
    nearest_node_dict = {}  # node_idx: zone_idx
    for zidx, z in zones.iterrows():
        closest_road_node = road_nodes.sindex.nearest(z.geometry, return_all=False)[1][
            0
        ]
        # for the first [x]:
        #   [0] represents the index of geometry;
        #   [1] represents the index of gdf
        # the second [x] represents the No. of closest item in the returned list,
        #   which only return one nearest node in this case
        nearest_node_dict[zidx] = closest_road_node

    weight_dict = defaultdict(lambda: defaultdict(list))
    for zidx in range(zones.shape[0]):
        admin = zones.loc[zidx, "OA21CD"]
        area = zones.loc[zidx, "gross_area_m2"]
        nidx = nearest_node_dict[zidx]
        n = road_nodes.loc[nidx, "nd_id"]
        weight_dict[n][admin].append(area)

    return weight_dict


# convert from defaultdict to dict
def convert_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_to_dict(v) for k, v in d.items()}
    return d


# %%
# extract the centroids of buildings link them to OA
verisk["centroid"] = verisk.geometry.centroid
verisk_centroids_gdf = verisk.drop(columns=["geometry"]).rename(
    columns={"centroid": "geometry"}
)
verisk_centroids_gdf = gpd.GeoDataFrame(
    verisk_centroids_gdf, geometry=verisk_centroids_gdf.geometry
)
temp = gpd.overlay(verisk_centroids_gdf, admin_file)  # intersection, 25mins

# extract GB buildings
verisk_gb = verisk[
    verisk["unique_property_number"].isin(temp["unique_property_number"])
]
verisk_gb.reset_index(drop=True, inplace=True)  # unique_property_number

# attach OA to GB buildings
dict_build_to_oa = temp.set_index("unique_property_number")["OA21CD"].to_dict()
verisk_gb["OA21CD"] = verisk_gb["unique_property_number"].map(dict_build_to_oa)

# %%
# classify buildings into Residential (R) vs Non-residential (NR)
verisk_gb.loc[(verisk_gb["height"] == 0) | (verisk_gb["height"].isnull()), "height"] = 1
verisk_gb["class"] = verisk_gb.use.map(class_dict)
mixeduse = verisk_gb[verisk_gb["class"] == "M"]

# ~25 million buildings
temp = verisk_gb[verisk_gb["class"] == "R"]
residential = pd.concat([temp, mixeduse], axis=0, ignore_index=True)  # 24,948,681
residential["unit_height"] = residential.use.map(uniheight_dict)
residential["num_of_floor"] = residential["height"] / residential["unit_height"]
residential["gross_area_m2"] = (
    residential["property_area"] * residential["num_of_floor"]
)
residential["gross_area_m2"] = residential["gross_area_m2"].astype(int)  # m2

# ~2 million buildings
temp = verisk_gb[verisk_gb["class"] == "N"]
nonresidential = pd.concat([temp, mixeduse], axis=0, ignore_index=True)  # 1,903,032
nonresidential["unit_height"] = nonresidential.use.map(uniheight_dict)
nonresidential["num_of_floor"] = (
    nonresidential["height"] / nonresidential["unit_height"]
)
nonresidential["gross_area_m2"] = (
    nonresidential["property_area"] * nonresidential["num_of_floor"]
)
nonresidential["gross_area_m2"] = nonresidential["gross_area_m2"].astype(int)  # m2

# %%
# attach buildings to OAs and estimate the total non-residential area (m2)
weight_nr = find_nearest_node(
    nonresidential, road_node_file
)  # 4 mins: weight[node][oa] = area
weight_nr2 = defaultdict(lambda: defaultdict(list))
for node, v1 in weight_nr.items():
    for oa, list_of_area in v1.items():
        total_area = sum(list_of_area)
        weight_nr2[node][oa] = total_area  # format
weight_nr3 = convert_to_dict(weight_nr2)

# %%
with open(
    base_path / "census_datasets" / "verisk" / "weight_non_residential.pkl",
    "wb",
) as f:
    pickle.dump(weight_nr3, f)

# %%
# attach buildings to OAs to estimate the total residential areas (m2)
weight_r = find_nearest_node(residential, road_node_file)  # 42 mins
weight_r2 = defaultdict(lambda: defaultdict(list))
for node, v1 in weight_r.items():
    for oa, list_of_area in v1.items():
        total_area = sum(list_of_area)
        weight_r2[node][oa] = total_area
weight_r3 = convert_to_dict(weight_r2)

# %%
with open(
    base_path / "census_datasets" / "verisk" / "weight_residential.pkl",
    "wb",
) as f:
    pickle.dump(weight_r3, f)

# %%
# normalised nodes' weights within each OA
# normalise non_residential properties
transformed_weight_nr = {}
for node, oas in weight_nr3.items():
    for oa, v in oas.items():
        if oa not in transformed_weight_nr:
            transformed_weight_nr[oa] = {}
        transformed_weight_nr[oa][node] = v

node_weight_nr = defaultdict(lambda: defaultdict(list))
for oa, list_of_nodes in transformed_weight_nr.items():
    ttArea = sum(list_of_nodes.values())
    for node, area in list_of_nodes.items():
        p = area / ttArea
        node_weight_nr[oa][node] = round(p, 2)

# %%
node_weight_nr = convert_to_dict(node_weight_nr)
with open(
    base_path / "census_datasets" / "verisk" / "node_weight_non_residential.pkl",
    "wb",
) as f:
    pickle.dump(node_weight_nr, f)

# %%
# normalise residential properties
transformed_weight_r = {}
for node, oas in weight_r3.items():
    for oa, v in oas.items():
        if oa not in transformed_weight_r:
            transformed_weight_r[oa] = {}
        transformed_weight_r[oa][node] = v

node_weight_r = defaultdict(lambda: defaultdict(list))
for oa, list_of_nodes in transformed_weight_r.items():
    ttArea = sum(list_of_nodes.values())
    for node, area in list_of_nodes.items():
        p = area / ttArea
        node_weight_r[oa][node] = round(p, 2)

# %%
node_weight_r = convert_to_dict(node_weight_r)
with open(
    base_path / "census_datasets" / "verisk" / "node_weight_residential.pkl",
    "wb",
) as f:
    pickle.dump(node_weight_r, f)

# %%
"""
For OAs without residential buildings, trip origins were approximated using the OA centroids.
Similarly, for OAs without nonresidential buildings, trip destinations were approximated using the OA centroids.
"""
# find the nearest node for each OA centroid (build the reference dict)
zone_to_node = find_nearest_node_centroid(admin_file, road_node_file)

# %%
# assign OA-level OD to nodes proportionally
origins = []  # 15,818,457
destinations = []
counts = []
# invalid_idx = []  # 331,015 invalid OA (3%)
for idx, row in tqdm(od_file.iterrows(), desc="Processing:", total=od_file.shape[0]):
    c = row["Count21"]
    oa_house = row["Area of usual residence"]
    oa_workplace = row["Area of workplace"]
    if oa_house in node_weight_r.keys():  # house
        list_of_origin_nodes = list(node_weight_r[oa_house].keys())
        list_of_origin_probs = list(node_weight_r[oa_house].values())
    else:  # if origin does not contain residential building
        # find the nearest node?
        # invalid_idx.append(idx)
        # continue
        list_of_origin_nodes = [zone_to_node[oa_house]]
        list_of_origin_probs = [1.0]
    if oa_workplace in node_weight_nr.keys():  # workplace
        list_of_destination_nodes = list(node_weight_nr[oa_workplace].keys())
        list_of_destination_probs = list(node_weight_nr[oa_workplace].values())
    else:  # if destination does not contain non-residential building
        # find the nearest node?
        # invalid_idx.append(idx)
        # continue
        list_of_destination_nodes = [zone_to_node[oa_workplace]]
        list_of_destination_probs = [1.0]

    for _ in range(int(c)):
        selected_origin_node = random.choices(
            list_of_origin_nodes, list_of_origin_probs
        )[0]
        selected_destination_node = random.choices(
            list_of_destination_nodes, list_of_destination_probs
        )[0]
        origins.append(selected_origin_node)
        destinations.append(selected_destination_node)
        counts.append(1)

temp_df = pd.DataFrame(
    {"origin_node": origins, "destination_node": destinations, "count": counts}
)
temp_df_group = temp_df.groupby(
    by=["origin_node", "destination_node"], as_index=False
).agg({"count": sum})

# %%
temp_df_group.to_csv(
    base_path / "census_datasets" / "od_matrix" / "od_gb_oa_2021_node.csv", index=False
)

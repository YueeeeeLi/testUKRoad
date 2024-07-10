"""
Some updates:
- cost function: £ (delete toll costs)
- speed flow curves (B roads: same as A_Single)

List of inputs:
- Parameter dicts
- Road networks (after roadnet proprocess)
- OD matrix (based on road nodes, after OD preproces)

For validation:
- Attach both major and minor roads to road_line_file (max(multiples))
"""

# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd  # type: ignore
from utils import load_config
import functions_revised as func
import json
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])

# %%
if __name__ == "__main__":
    # model parameters
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)
    with open(base_path / "parameters" / "flow_cap_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)
    with open(base_path / "parameters" / "free_flow_speed_dict.json", "r") as f:
        free_flow_speed_dict = json.load(f)
    with open(base_path / "parameters" / "min_speed_cap.json", "r") as f:
        min_speed_dict = json.load(f)
    with open(base_path / "parameters" / "urban_speed_cap.json", "r") as f:
        urban_speed_dict = json.load(f)

    # road networks (urban_filter_revised + tolls)
    road_node_file = gpd.read_parquet(
        base_path / "networks" / "road" / "road_node_file.geoparquet"
    )
    road_link_file = gpd.read_parquet(
        base_path / "networks" / "road" / "road_link_file.geoparquet"
    )  # !!! delete the tolls information

    # O-D matrix (2021)
    od_node_2021 = pd.read_csv(
        base_path / "census_datasets" / "od_matrix" / "od_gb_oa_2021_node.csv"
    )
    print(f"total flows: {od_node_2021.Car21.sum()}")

    # generate OD pairs
    list_of_origin_nodes, dict_of_destination_nodes, dict_of_origin_supplies = (
        func.extract_od_pairs(od_node_2021)
    )
    # network creation (igragh)
    test_net_ig, edge_cost_dict, edge_timeC_dict, edge_operateC_dict = (
        func.create_igraph_network(road_link_file, road_node_file, free_flow_speed_dict)
    )  # this returns a network and edge weights dict(edge_name, edge_weight)
    edge_index_to_name = {
        idx: name for idx, name in enumerate(test_net_ig.es["edge_name"])
    }

    # network initialisation
    road_link_file = func.initialise_igraph_network(
        road_link_file,
        flow_capacity_dict,
        free_flow_speed_dict,
        col_road_classification="road_classification",
    )

    # flow simulation
    speed_dict, acc_flow_dict, acc_capacity_dict = func.network_flow_model(
        test_net_ig,  # network
        edge_cost_dict,  # total cost
        edge_timeC_dict,  # value of time
        edge_operateC_dict,  # vehicle operating cost
        road_link_file,  # road
        list_of_origin_nodes,  # od
        dict_of_origin_supplies,  # od
        dict_of_destination_nodes,  # od
        free_flow_speed_dict,  # speed
        flow_breakpoint_dict,  # speed
        min_speed_dict,  # speed
        urban_speed_dict,  # speed
    )
    # append estimation of: speeds, flows, and remaining capacities
    road_link_file.ave_flow_rate = road_link_file.e_id.map(speed_dict)
    road_link_file.acc_flow = road_link_file.e_id.map(acc_flow_dict)
    road_link_file.acc_capacity = road_link_file.e_id.map(acc_capacity_dict)

    # change field types
    road_link_file.acc_flow = road_link_file.acc_flow.astype(int)
    road_link_file.acc_capacity = road_link_file.acc_capacity.astype(int)

    # export files
    road_link_file.to_parquet(
        base_path.parent / "outputs" / "p_road_gb_2021_revised.gpkg"
    )

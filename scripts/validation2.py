# %%
from pathlib import Path
import geopandas as gpd
import pandas as pd
from utils import load_config
import warnings
from collections import defaultdict

pd.set_option("display.max_columns", None)
warnings.simplefilter("ignore")

incoming_path = Path(load_config()["paths"]["incoming_data"])
base_path = Path(load_config()["paths"]["base_path"])


dft_file = pd.read_csv(
    incoming_path / "road" / "mrdb-traffic counts" / "dft_traffic_counts_aadf_2021.csv"
)

# cars and buses
dft_file["commuters_vehicles"] = dft_file["Cars_and_taxis"]
dft_count_dict = dft_file.set_index("Count_point_id")["Cars_and_taxis"].to_dict()
dft_gdf = gpd.GeoDataFrame(
    dft_file,
    geometry=gpd.points_from_xy(x=dft_file.Longitude, y=dft_file.Latitude),
    crs="4326",
)
dft_gdf = dft_gdf.to_crs("27700")
road_link_file = gpd.read_parquet(
    base_path / "networks" / "road" / "road_link_file.geoparquet"
)


# %%
# find nearest road link for each traffic count point
def find_nearest_node(
    count_points: gpd.GeoDataFrame, road_lines: gpd.GeoDataFrame
) -> dict:
    nearest_node_dict = {}
    # find the nearest link for each counting point
    for zidx, z in count_points.iterrows():
        closest_road_line = road_lines.sindex.nearest(z.geometry, return_all=False)[1][
            0
        ]
        # for the first [x]:
        #   [0] represents the index of geometry;
        #   [1] represents the index of gdf
        # the second [x] represents the No. of closest item in the returned list,
        #   which only return one nearest node in this case
        nearest_node_dict[zidx] = closest_road_line

    cp_to_link = defaultdict(list)
    for zidx in range(count_points.shape[0]):
        z = count_points.loc[zidx, "Count_point_id"]
        nidx = nearest_node_dict[zidx]
        n = road_lines.loc[nidx, "e_id"]
        cp_to_link[n].append(z)

    return cp_to_link


# %%
cp_to_road = find_nearest_node(dft_gdf, road_link_file)
# %%
cp_to_road_counts = defaultdict(list)
for k, v in cp_to_road.items():
    for cp in v:
        car = dft_count_dict[cp]
        cp_to_road_counts[k].append(car)

cp_to_road_max_count = {}
for k, v in cp_to_road_counts.items():
    maxv = max(v)
    cp_to_road_max_count[k] = maxv

# %%
road_link_file["Cars_and_taxis"] = road_link_file.e_id.map(cp_to_road_max_count)
# %%
road_link_file.to_parquet(
    base_path / "networks" / "road" / "link_traffic_counts.geoparquet"
)

# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler

simulation = gpd.read_parquet(
    r"C:\Oxford\Research\DAFNI\local\outputs\p_road_gb_2021_revised.gpkg"
)
observation = gpd.read_parquet(
    r"C:\Oxford\Research\DAFNI\local\processed_data\networks\road\link_traffic_counts.geoparquet"
)

# %%
temp = pd.DataFrame(
    {
        "type": simulation.combined_label,
        "simu": simulation.acc_flow,
        "obs": observation.Cars_and_taxis,
    }
)
temp = temp[temp.obs.notnull()]

# %%
temp_ASingle = temp[temp.type == "A_single"]
temp_ADual = temp[temp.type == "A_dual"]
temp_B = temp[temp.type == "B"]
temp_M = temp[temp.type == "M"]
temp_A = temp[(temp.type == "A_single") | (temp.type == "A_dual")]

# %%
simulation_values = np.array(temp_A.simu)
observation_values = np.array(temp_A.obs)

# Pearson's correlation coefficient
pearson_corr, _ = pearsonr(simulation_values, observation_values)

# Spearman's rank correlation coefficient
spearman_corr, _ = spearmanr(simulation_values, observation_values)

# Normalize the data for visual comparison
scaler = MinMaxScaler()
simulation_values_normalized = scaler.fit_transform(
    simulation_values.reshape(-1, 1)
).flatten()
observation_values_normalized = scaler.fit_transform(
    observation_values.reshape(-1, 1)
).flatten()

plt.figure(figsize=(12, 8))  # Width: 10 inches, Height: 6 inches
plt.rcParams.update({"font.size": 14})

# Plot the normalized data
plt.plot(simulation_values_normalized, label="Normalized Simulation Values", marker="o")
plt.plot(
    observation_values_normalized, label="Normalized Observation Values", marker="x"
)

# Add labels and legend
plt.xlabel("Index")
plt.ylabel("Normalized Value")
plt.legend(loc="upper left")

# Display the plot
plt.show()

# Print the correlation results
print(f"Pearson's Correlation Coefficient: {pearson_corr}")
print(f"Spearman's Rank Correlation Coefficient: {spearman_corr}")

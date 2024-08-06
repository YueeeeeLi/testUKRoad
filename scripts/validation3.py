# %%
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler

from utils import load_config

output_path = Path(load_config()["paths"]["outputs"])
base_path = Path(load_config()["paths"]["base_path"])

simulation = gpd.read_parquet(
    output_path / "p_road_gb_2021_revised_doubled_changed_0717.gpkg"
)
observation = gpd.read_parquet(
    base_path / "networks" / "road" / "link_traffic_counts.geoparquet"
)

# %%
temp_dict = observation.set_index("e_id")["Cars_and_taxis"].to_dict()
simulation["obs"] = simulation["e_id"].map(temp_dict)
simulation["simu-obs"] = simulation.acc_flow - simulation.obs * 1.6
simulation["diff"] = np.nan
simulation.loc[simulation["simu-obs"] > 0, "diff"] = 1
simulation.loc[simulation["simu-obs"] < 0, "diff"] = -1
# simulation.to_parquet(output_path / "diff_0717.geoparquet")

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
simulation_values = np.array(temp_B.sort_values(by=["obs"]).simu)
observation_values = np.array(temp_B.sort_values(by=["obs"]).obs)

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

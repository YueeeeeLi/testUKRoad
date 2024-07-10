# %%
"""
Workday population estimation
"""
import os
import pandas as pd
from collections import defaultdict

base_path = r"C:\Oxford\Research\DAFNI\local\processed_data\census_datasets\od_matrix"

# %%
# od matrix with origins in Scotland (including cars and other methods)
od11_gb = pd.read_csv(
    os.path.join(base_path, "od_gb_2011_updated.csv")
)  # source: WU03BUK_oa_wz_v4 (2011)

od11_Scot_oa = od11_gb[od11_gb.from_region == "oa_scot"]
od11_Scot_oa.reset_index(drop=True, inplace=True)
od11_Scot_oa.rename(columns={"travel to work": "travel11"}, inplace=True)
od11_Scot_oa.drop(columns=["car"], inplace=True)

# %%
# population 11 at LSOA and OA levels
pop11_Scot = pd.read_csv(os.path.join(base_path, "pop11_scot.csv"))
pop11_Scot_lsoa = pop11_Scot.groupby("LSOA11CD", as_index=False).agg({"Popcount": sum})
pop11_Scot_oa = pop11_Scot[["OA11CD", "Popcount"]]
# population 21 at LSOA
pop21_Scot_lsoa = pd.read_excel(
    os.path.join(base_path, "sape-2021.xlsx"), sheet_name="Sheet1"
)
pop21_Scot_lsoa.rename(
    columns={"Data zone code": "LSOA21CD", "Total population": "Popcount"}, inplace=True
)

# -> to estimate population 21 at OA
temp_pop = pd.merge(pop11_Scot, pop11_Scot_lsoa, on="LSOA11CD", how="left")
temp_pop["oa/lsoa"] = temp_pop.Popcount_x / temp_pop.Popcount_y
dict_ratio = defaultdict(lambda: defaultdict(list))
for idx, row in temp_pop.iterrows():
    oa = row["OA11CD"]
    lsoa = row["LSOA11CD"]
    rt = row["oa/lsoa"]
    dict_ratio[lsoa][oa] = rt

oaList = []
popList = []
for idx, row in pop21_Scot_lsoa.iterrows():
    lsoa = row["LSOA21CD"]
    pop = row["Popcount"]
    if lsoa in dict_ratio.keys():
        for oa, ratio in dict_ratio[lsoa].items():
            oaList.append(oa)
            popList.append(pop * ratio)

pop21_Scot_oa = pd.DataFrame({"OA21CD": oaList, "Popcount": popList})

# %%
# od projection from 2011 to 2021 for Scotland
pop11_scot_dict = pop11_Scot_oa.set_index("OA11CD")["Popcount"].to_dict()
pop21_scot_dict = pop21_Scot_oa.set_index("OA21CD")["Popcount"].to_dict()
od11_Scot_oa["Pop11"] = od11_Scot_oa["Area of usual residence"].map(pop11_scot_dict)
od11_Scot_oa["Pop21"] = od11_Scot_oa["Area of usual residence"].map(pop21_scot_dict)
od11_Scot_oa["travel21"] = (
    od11_Scot_oa.travel11 * od11_Scot_oa.Pop21 / od11_Scot_oa.Pop11
)

# %%
# od disaggregation to OA level for Scotland
# workday population = total population + inflow - outflow
### remove travels within the same OA
od11_Scot_oa = od11_Scot_oa[
    od11_Scot_oa["Area of usual residence"] != od11_Scot_oa["Area of workplace"]
]
od11_Scot_oa.reset_index(drop=True, inplace=True)

# %%
### disaggregating to OA levels
od11_scotoa_scotoa = od11_Scot_oa[od11_Scot_oa.to_region != "msoa_enw"]
od11_scotoa_scotoa.reset_index(drop=True, inplace=True)
od11_scotoa_enwmsoa = od11_Scot_oa[od11_Scot_oa.to_region == "msoa_enw"]
od11_scotoa_enwmsoa.reset_index(drop=True, inplace=True)

# %%
# look-up table: msoa11 to oa11 (geometry)
# disaggregate travel21 from combined levels to OA level

# ENW: 2021 Population at OA level (dict)
pop21_enw_oa21 = pd.read_csv(
    os.path.join(base_path, "gb_population_2021_estimates.csv")
)
pop21_enw_oa_dict = pop21_enw_oa21.set_index("OA21CD")["OA_POP_2021"].to_dict()

# From MSOA11 to OA11 (each MSOA contains multiple OAs)
lut_enw_msoa11_oa11 = pd.read_csv(
    os.path.join(base_path, "lut\\_lut_enw_msoa11_oa11.csv")
)
msoa11_to_oa11_dict = (
    lut_enw_msoa11_oa11.groupby(by=["MSOA11CD"], as_index=False)
    .agg({"OA11CD": list})
    .set_index("MSOA11CD")["OA11CD"]
    .to_dict()
)
pop11_enw_oa = pd.read_csv(os.path.join(base_path, "pop11_enw_oa.csv"))
pop11_enw_oa_dict = pop11_enw_oa.set_index("OA11CD")["POP11"].to_dict()
msoa11_oa11_pop11 = defaultdict(lambda: defaultdict(list))
for msoa11 in msoa11_to_oa11_dict.keys():
    for oa11 in msoa11_to_oa11_dict[msoa11]:
        pop11 = pop11_enw_oa_dict[oa11]
        msoa11_oa11_pop11[msoa11][oa11] = pop11

# From OA11 to OA21 (some OA11 can have multiple OA21)
lut_enw_oa11_oa21 = pd.read_csv(os.path.join(base_path, "lut\\_lut_enw_oa11_oa21.csv"))
oa11_to_oa21_dict = (
    lut_enw_oa11_oa21.groupby(by=["OA11CD"]).agg({"OA21CD": list}).to_dict()["OA21CD"]
)
pop21_enw_oa = pd.read_csv(os.path.join(base_path, "gb_population_2021_estimates.csv"))
pop21_enw_oa_dict = pop21_enw_oa.set_index("OA21CD")["OA_POP_2021"].to_dict()
oa11_oa21_pop21 = defaultdict(lambda: defaultdict(list))
for oa11 in oa11_to_oa21_dict.keys():
    for oa21 in oa11_to_oa21_dict[oa11]:
        pop21 = pop21_enw_oa_dict[oa21]
        oa11_oa21_pop21[oa11][oa21] = pop21

# %%
scot11List = []
oa21List = []
travel21List = []
for idx, row in od11_scotoa_enwmsoa.iterrows():
    scot11 = row["Area of usual residence"]
    msoa11 = row["Area of workplace"]
    travel21 = row["travel21"]
    if msoa11 in msoa11_oa11_pop11.keys():
        ttpop_msoa11 = sum(msoa11_oa11_pop11[msoa11].values())
        for oa11, pop11 in msoa11_oa11_pop11[msoa11].items():
            ratio11 = pop11 / ttpop_msoa11
            count11 = travel21 * ratio11
            if oa11 in oa11_oa21_pop21.keys():
                ttpop_oa11 = sum(oa11_oa21_pop21[oa11].values())
                for oa21, pop21 in oa11_oa21_pop21[oa11].items():
                    ratio21 = pop21 / ttpop_oa11
                    count21 = count11 * ratio21

                    scot11List.append(scot11)
                    oa21List.append(oa21)
                    travel21List.append(count21)
            else:
                print(f"cannot find oa21 for oa11: {oa11}")
    else:
        print(f"cannot find oa11 for msoa11: {msoa11}")

# %%
temp_df = pd.DataFrame(
    {
        "Area of usual residence": scot11List,
        "Area of workplace": oa21List,
        "travel21": travel21List,
    }
)

temp_df = temp_df.groupby(
    by=["Area of usual residence", "Area of workplace"],
    as_index=False,
).agg({"travel21": sum})

# %%
# combined dataframe
# Scotland (from OA to OA: estimated travels 2021)
temp_od11_scotoa_scotoa = od11_scotoa_scotoa[
    ["Area of usual residence", "Area of workplace", "travel21"]
]
od21_scot_oa = pd.concat([temp_df, temp_od11_scotoa_scotoa], axis=0, ignore_index=True)

# %%
# England and Wales (from OA to OA: real observation data)
od21_enw_oa = pd.read_csv(os.path.join(base_path, "od_enw_2021\\ODWP01EW_OA.csv"))
od21_enw_oa.rename(
    columns={
        "Output Areas code": "Area of usual residence",
        "OA of workplace code": "Area of workplace",
        "Count": "travel21",
    },
    inplace=True,
)
od21_enw_oa = od21_enw_oa[
    (od21_enw_oa["Place of work indicator (4 categories) code"] == 3)
    & (od21_enw_oa["Area of usual residence"] != od21_enw_oa["Area of workplace"])
]
od21_enw_oa = od21_enw_oa[["Area of usual residence", "Area of workplace", "travel21"]]
od21_enw_oa.reset_index(drop=True, inplace=True)
od21_gb_oa = pd.concat([od21_scot_oa, od21_enw_oa], axis=0, ignore_index=True)

# attach total population 2021 to the od21_gb_oa
od21_gb_oa["POP21"] = od21_gb_oa["Area of usual residence"].map(pop21_enw_oa_dict)

# %%
### estimate workday population 2021
total_pop = (
    od21_gb_oa.groupby(by=["Area of usual residence"], as_index=False)
    .agg({"POP21": "first"})
    .rename(columns={"Area of usual residence": "CODE"})
)
inflow_pop = (
    od21_gb_oa.groupby(by=["Area of workplace"], as_index=False)
    .agg({"travel21": sum})
    .rename(columns={"Area of workplace": "CODE", "travel21": "inflow"})
)
outflow_pop = (
    od21_gb_oa.groupby(by=["Area of usual residence"], as_index=False)
    .agg({"travel21": sum})
    .rename(columns={"Area of usual residence": "CODE", "travel21": "outflow"})
)

# %%
temp_merge = pd.merge(total_pop, inflow_pop, on="CODE", how="left")
gb_merge = pd.merge(temp_merge, outflow_pop, on="CODE", how="left")
gb_merge = gb_merge.fillna(0.0)
gb_merge["workday_pop_2021"] = gb_merge.POP21 + gb_merge.inflow - gb_merge.outflow
gb_merge.to_csv(
    os.path.join(base_path, "gb_workday_population_2021_float.csv"), index=False
)

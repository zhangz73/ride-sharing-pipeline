import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCENARIO_NAME = "st-stc_12car4region_xi=0.5" #"12car4region48chargers15mins_fullycharged_nyc_combo" #"10car5region" #

trip_demand_df = pd.read_csv(f"Data/TripDemand/trip_demand_{SCENARIO_NAME}.tsv", sep = "\t")
map_df = pd.read_csv(f"Data/Map/map_{SCENARIO_NAME}.tsv", sep = "\t")

time_horizon = map_df["T"].max() + 1
num_regions = map_df["Origin"].max() + 1

inout_mat = np.zeros((num_regions, time_horizon))

for i in range(trip_demand_df.shape[0]):
    t, origin, dest, rate = trip_demand_df.iloc[i]["T"].astype(int), trip_demand_df.iloc[i]["Origin"].astype(int), trip_demand_df.iloc[i]["Destination"].astype(int), trip_demand_df.iloc[i]["Count"]
    trip_time = map_df[(map_df["T"] == t) & (map_df["Origin"] == origin) & (map_df["Destination"] == dest)].iloc[0]["TripTime"].astype(int)
    inout_mat[origin, t] -= rate
    if t + trip_time < time_horizon:
        inout_mat[dest, t + trip_time] += rate

inout_mat = inout_mat.cumsum(axis = 1)
tot = inout_mat.sum(axis = 0)

for region in range(num_regions):
    plt.plot(inout_mat[region,:], label = f"Region {region}")
plt.plot(tot, color = "gray")
plt.axhline(y = 0, color = "black")
plt.xlabel("Time Steps")
plt.ylabel("In/Out Flow")
plt.legend()
plt.savefig(f"DataPlots/balancedness_{SCENARIO_NAME}.png")
plt.clf()
plt.close()

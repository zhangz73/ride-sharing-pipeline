import numpy as np
from pyproj import Proj, Transformer
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap

## Setup ESPG transformer from GPS to coordinates
transformer = Transformer.from_crs(4326, 2263, always_xy = True)

## Load data from files
df = gpd.read_file("Data/Map/taxi_zones/taxi_zones.shp")
df_data = pd.read_parquet("Data/Map/fhvhv_tripdata_2022-07.parquet")
df_chargers = pd.read_csv("Data/Map/ny_charging_stations.csv")
df_chargers = df_chargers[(df_chargers["City"] == "New York") & (df_chargers["Restricted Access"] == False)]
df_zone_lookup = pd.read_csv("Data/Map/taxi_zone_lookup.csv")

## Construct the dataframe of charging locations
geometry = [Point((transformer.transform(x, y))) for x, y in zip(df_chargers["Longitude"], df_chargers["Latitude"])]
df_chargers = gpd.GeoDataFrame(geometry = geometry)
df_chargers.crs = df.crs

## Construct the dataframe of trip links among taxi zones
centroids = df["geometry"].apply(lambda x: list(x.centroid.coords)[0])
df_link = df[["LocationID", "borough"]].copy()
df_link["Centroids"] = centroids
df_link["key"] = 0
df_link = df_link.merge(df_link, on = "key").drop("key", axis = 1)
df_link = gpd.GeoDataFrame(
    data = df_link,
    geometry = df_link.apply(
        lambda x: shapely.geometry.LineString(
            [x["Centroids_x"], x["Centroids_y"]]
        ), axis = 1
    )
)
df_link = df_link[["LocationID_x", "LocationID_y", "borough_x", "borough_y", "geometry"]]
df_link.columns = ["PULocationID", "DOLocationID", "borough_x", "borough_y", "geometry"]

## Map charger locations to taxi zones
zones = []
boroughs = []
for i in range(df_chargers.shape[0]):
    charger_loc = df_chargers.iloc[i]["geometry"]
    for j in range(df.shape[0]):
        taxi_zone = df.iloc[j]["LocationID"]
        geom = df.iloc[j]["geometry"]
        region = df.iloc[j]["borough"]
        if geom.contains(charger_loc):
            zones.append(taxi_zone)
            boroughs.append(region)
            break
df_chargers["LocationID"] = zones
df_chargers["borough"] = boroughs

## Define helper functions
### Generate heatmap
def get_heatmap(df_plot, title, fname, overlay_chargers = True, df_chargers = None):
    if overlay_chargers:
        assert df_chargers is not None
    fig, ax = plt.subplots()
    df_plot.plot(ax = ax, edgecolor = "black", cmap = "OrRd", column = "TripCounts", legend = True)
    if overlay_chargers:
        df_chargers.plot(ax = ax, marker = "o", color = "green", markersize = 2)
    ax.set_aspect("equal")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"MapExplore/{fname}.png")
    plt.clf()
    plt.close()

### Generate trip demand heatmap for given regions
def get_tripdemand_heatmap(regions = None):
    if regions is not None and len(regions) > 0:
        df_sub = df[df["borough"].isin(regions)]
        df_chargers_sub = df_chargers[df_chargers["borough"].isin(regions)]
        name = "-".join(regions)
    else:
        df_sub = df
        df_chargers_sub = df_chargers
        name = "nyc"
    df_merged = df_sub.merge(df_data, left_on = "LocationID", right_on = "PULocationID")
    df_merged["TripCounts"] = 1
    df_tripcounts = df_merged[["LocationID", "TripCounts"]].groupby("LocationID").sum().reset_index()
    df_tripdemands = df_sub.merge(df_tripcounts, on = "LocationID")
    get_heatmap(df_tripdemands, f"{name.upper()} Trip Demands 2022-07", f"TripDemand/{name.lower()}_tripdemand_202207", overlay_chargers = True, df_chargers = df_chargers_sub)
    return df_tripdemands

### Generate trip destination heatmap for given regions
def get_tripdest_heatmap(regions = None):
    if regions is not None and len(regions) > 0:
        df_sub = df_data[df_data["PULocationID"].isin(regions)]
        region_names = [df_zone_lookup[df_zone_lookup["LocationID"] == x].iloc[0]["Zone"].replace(" ", "_").replace("/", " ") for x in regions]
        name = "-".join(region_names)
    else:
        df_sub = df_data
        name = "nyc"
    df_merged = df.merge(df_sub, left_on = "LocationID", right_on = "DOLocationID")
    df_merged["TripCounts"] = 1
    df_tripcounts = df_merged[["LocationID", "TripCounts"]].groupby("LocationID").sum().reset_index()
    df_tripdests = df.merge(df_tripcounts, on = "LocationID")
    get_heatmap(df_tripdests, f"{name.upper()} Trip Destinations 2022-07", f"TripDest/{name.lower()}_tripdest_202207", overlay_chargers = False, df_chargers = None)
    return df_tripdests

### Generate histograms
def get_hist(val, title, fname, x = None):
    if x is not None:
        plt.bar(x, val)
    else:
        plt.hist(val, bins = 100, density = True)
#        plt.xscale("log")
    plt.title(title)
    plt.savefig(f"MapExplore/Histograms/{fname}.png")
    plt.clf()
    plt.close()

### Generate map for travel volumes
def get_travel_link_plot(title, fname, regions = None, overlay_chargers = True):
    if regions is not None and len(regions) > 0:
        df_sub = df_link[(df_link["borough_x"].isin(regions)) & (df_link["borough_y"].isin(regions))]
        df_cp = df[df["borough"].isin(regions)]
        df_chargers_sub = df_chargers[df_chargers["borough"].isin(regions)]
        name = "-".join(regions)
    else:
        df_sub = df_link
        df_cp = df
        df_chargers_sub = df_chargers
        name = "nyc"
    df_triplinks = df_sub.merge(df_data, on = ["PULocationID", "DOLocationID"])
    df_triplinks["TripCounts"] = 1
    df_triplinkcounts = df_triplinks[["PULocationID", "DOLocationID", "TripCounts"]].groupby(["PULocationID", "DOLocationID"]).sum().reset_index()
    df_triplinks = df_sub.merge(df_triplinkcounts, on = ["PULocationID", "DOLocationID"])
    df_triplinks["TripCountsLog"] = np.log10(df_triplinks["TripCounts"])
    df_triplinks = df_triplinks.sort_values("TripCounts", ascending = False)
    
#    plt.hist(df_triplinks["TripCounts"], bins = 100)
#    plt.title(df_triplinks["TripCounts"].max())
#    plt.show()

    df_triplinksplot = df_triplinks[df_triplinks["PULocationID"] != df_triplinks["DOLocationID"]].copy()
    df_triplinksplot["TripCountsBin"] = df_triplinksplot["TripCounts"].apply(lambda x: 11000 if x >= 10000 else 9000 if x >= 8000 else 7000 if x >= 6000 else 5000 if x >= 4000 else 3000 if x >= 2000 else 1500 if x >= 1000 else 500)
    df_triplinksplot = df_triplinksplot[df_triplinksplot["TripCounts"] >= 1000]
    df_triplinksplot = df_triplinksplot.sort_values("TripCounts", ascending = True)
    
    cmap = plt.cm.get_cmap("gist_heat_r") #plt.cm.OrRd #RdBu_r #hot_r
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    
    fig, ax = plt.subplots()
    df_cp.plot(ax = ax, edgecolor = "black", color = "lightskyblue")
    links = df_triplinksplot.plot(ax = ax, cmap = my_cmap, column = "TripCounts", legend = True)
    
#    cb = ax.get_figure().get_axes()[0]
#    print(cb.get_yticklabels())
#    print(links)
#    leg = links.get_legend()
#    for lbl in leg.get_texts():
#        print(lbl.get_text())
    
    if overlay_chargers:
        df_chargers_sub.plot(ax = ax, marker = "o", color = "green", markersize = 2)
    ax.set_aspect("equal")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"MapExplore/TripLinks/{fname}.png")
    plt.clf()
    plt.close()
    return df_triplinks

"""
## Visualize the trip demand heatmaps
df_tripdemands = get_tripdemand_heatmap(regions = [])
get_tripdemand_heatmap(regions = ["Manhattan"])
df_tripdemands[["LocationID", "zone", "TripCounts"]].drop_duplicates().sort_values("TripCounts", ascending = False).to_csv("MapExplore/TripDemand/trip_demands.csv", index=False)

## Visualize the trip destinations for representative regions
### (LocationID, Zone)
### (138, LaGuardia Airport), (132, JFK Airport), (79, East Village), (230, Times Sq/Theatre District), (161, Midtown Center), (42, Central Harlem), (186, Penn Station), (236, Upper East Side North), (239, Upper West Side South), (74, East Harlem)
regions = [None, 138, 132, 79, 230, 161, 42, 186, 236, 239, 74]
for region in regions:
    if region is None:
        input = None
    else:
        input = [region]
    get_tripdest_heatmap(regions = input)

## Visualize the request density across time
df_vis = df_data.copy()
df_vis["Hours"] = [x.hour for x in df_vis["request_datetime"]]
df_vis["TripRequests"] = 1
df_vis = df_vis[["Hours", "TripRequests"]].groupby("Hours").sum().reset_index()
get_hist(df_vis["TripRequests"], "# Trip Requests", "trip_request", x = df_vis["Hours"])

## Visualize the trip time, trip distance, trip waiting time densities
df_vis = df_data.copy()
## Remove erroneous data
### Remove entries where driver arrives before the request (1.1% records)
df_vis = df_vis[df_vis["on_scene_datetime"] >= df_vis["request_datetime"]]
### Remove entries where trip distance is ridiculously long (0.01% records)
df_vis = df_vis[df_vis["trip_miles"] <= 100]

df_vis["TripTime"] = (df_vis["dropoff_datetime"] - df_vis["pickup_datetime"]).apply(lambda x: x.seconds / 60)
df_vis["WaitingTime"] = (df_vis["on_scene_datetime"] - df_vis["request_datetime"]).apply(lambda x: x.seconds / 60)
### Visualize trip time
get_hist(df_vis["TripTime"], "Trip Time (Minutes)", "trip_time")
### Visualize trip distance
get_hist(df_vis["trip_miles"], "Trip Distance (Miles)", "trip_distance")
### Visualize trip time
get_hist(df_vis["WaitingTime"], "Trip Waiting Time (Minutes)", "wait_time")
"""

## Visualize the travel volumes
df_triplinks = get_travel_link_plot("NYC Trip Links", "nyc_trip_links", regions = None, overlay_chargers = False)
#df_triplinks.to_csv("MapExplore/TripLinks/trip_links.csv", index = False)
get_travel_link_plot("Manhattan Trip Links", "manhattan_trip_links", regions = ["Manhattan"], overlay_chargers = False)

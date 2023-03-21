import holidays
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
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

## Select Scope
SCOPE = "All"
dct_key = {
    "Uber": ["HV0003"],
    "Lyft": ["HV0005"],
    "All": ["HV0003", "HV0005"]
}

## Setup ESPG transformer from GPS to coordinates
transformer = Transformer.from_crs(4326, 2263, always_xy = True)

## Load data from files
df = gpd.read_file("Data/Map/taxi_zones/taxi_zones.shp")
df_data = pd.read_parquet("Data/Map/fhvhv_tripdata_2022-07.parquet")
df_chargers = pd.read_csv("Data/Map/ny_charging_stations.csv")
df_chargers = df_chargers[(df_chargers["City"] == "New York") & (df_chargers["Restricted Access"] == False)]
df_zone_lookup = pd.read_csv("Data/Map/taxi_zone_lookup.csv")

df_data = df_data[df_data["hvfhs_license_num"].isin(dct_key[SCOPE])]

## Construct the dataframe of charging locations
charger_offset = 0#0.005
geometry = [Point((transformer.transform(x, y))) for x, y in zip(df_chargers["Longitude"], df_chargers["Latitude"])]
geometry_level2 = [Point((transformer.transform(x, y))) for x, y in zip(df_chargers["Longitude"] - charger_offset, df_chargers["Latitude"])]
geometry_dcFast = [Point((transformer.transform(x, y))) for x, y in zip(df_chargers["Longitude"] + charger_offset, df_chargers["Latitude"])]
geometry_long = geometry_level2 + geometry_dcFast
df_chargers_long = df_chargers[["EV Level2 EVSE Num", "EV DC Fast Count"]].copy().fillna(0)
df_chargers_long.columns = ["ChargerNum_Level2", "ChargerNum_DCFast"]
df_chargers_long["Geometry_Level2"] = geometry_level2
df_chargers_long["Geometry_DCFast"] = geometry_dcFast
df_chargers_long["geometry_orig"] = geometry
df_chargers_long["id"] = df_chargers_long.index
df_chargers_long = pd.wide_to_long(df_chargers_long, stubnames = ["ChargerNum", "Geometry"], i = "geometry_orig", j = "ChargerType", sep = "_", suffix = r"\w+").reset_index()
df_chargers_long = df_chargers_long[df_chargers_long["ChargerNum"] > 0]
df_chargers_long = gpd.GeoDataFrame(df_chargers_long, crs = df.crs, geometry = df_chargers_long["Geometry"])

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

zones = []
boroughs = []
for i in range(df_chargers_long.shape[0]):
    charger_loc = df_chargers_long.iloc[i]["geometry_orig"]
    for j in range(df.shape[0]):
        taxi_zone = df.iloc[j]["LocationID"]
        geom = df.iloc[j]["geometry"]
        region = df.iloc[j]["borough"]
        if geom.contains(charger_loc):
            zones.append(taxi_zone)
            boroughs.append(region)
            break
df_chargers_long["LocationID"] = zones
df_chargers_long["borough"] = boroughs

## Define helper functions
### Generate heatmap
def get_heatmap(df_plot, title, fname, overlay_chargers = True, df_chargers = None, save = True, cmap = "OrRd", norm = None):
    if overlay_chargers:
        assert df_chargers is not None
    fig, ax = plt.subplots()
    if norm is not None:
        df_plot.plot(ax = ax, edgecolor = "black", cmap = cmap, column = "TripCounts", legend = True, norm = norm)
    else:
        df_plot.plot(ax = ax, edgecolor = "black", cmap = cmap, column = "TripCounts", legend = True)
    if overlay_chargers:
        df_chargers.plot(ax = ax, marker = "o", color = "green", markersize = 2)
    if save:
        ax.set_aspect("equal")
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"MapExplore/{SCOPE}/{fname}.png")
        plt.clf()
        plt.close()
    return fig, ax

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

### Generate trip destination heatmap for given regions
def get_tripdest_time_heatmap(df_data, hours = None, desc = "all_time", save = True, cmap = "OrRd"):
    if hours is not None and len(hours) > 0:
        df_sub = df_data[df_data["Hours"].isin(hours)]
    else:
        df_sub = df_data
    df_merged = df.merge(df_sub, left_on = "LocationID", right_on = "DOLocationID")
    df_merged["TripCounts"] = 1
    df_tripcounts = df_merged[["LocationID", "TripCounts"]].groupby("LocationID").sum().reset_index()
    df_tripdests = df.merge(df_tripcounts, on = "LocationID")
    fig, ax = get_heatmap(df_tripdests, f"{desc.replace('_', ' ').title()} Trip Destinations 2022-07", f"TripDestTime/{desc.lower()}_tripdesttime_202207", overlay_chargers = False, df_chargers = None, save = save, cmap = cmap)
    if save:
        return df_tripdests
    return fig, ax

### Generate trip destination heatmap for given regions
def get_triporig_time_heatmap(df_data, hours = None, desc = "all_time", save = True, cmap = "OrRd"):
    if hours is not None and len(hours) > 0:
        df_sub = df_data[df_data["Hours"].isin(hours)]
    else:
        df_sub = df_data
    df_merged = df.merge(df_sub, left_on = "LocationID", right_on = "PULocationID")
    df_merged["TripCounts"] = 1
    df_tripcounts = df_merged[["LocationID", "TripCounts"]].groupby("LocationID").sum().reset_index()
    df_triporigs = df.merge(df_tripcounts, on = "LocationID")
    fig, ax = get_heatmap(df_triporigs, f"{desc.replace('_', ' ').title()} Trip Origins 2022-07", f"TripOrigTime/{desc.lower()}_triporigtime_202207", overlay_chargers = False, df_chargers = None, save = save, cmap = cmap)
    if save:
        return df_triporigs
    return fig, ax

### Generate trip destination heatmap for given regions
def get_supply_time_heatmap(df_data, hours = None, desc = "all_time", save = True, cmap = "OrRd"):
    if hours is not None and len(hours) > 0:
        df_sub = df_data[df_data["Hours"].isin(hours)]
    else:
        df_sub = df_data
    df_merged_out = df.merge(df_sub, left_on = "LocationID", right_on = "PULocationID")
    df_merged_out["OutCounts"] = 1
    df_outcounts = df_merged_out[["LocationID", "OutCounts"]].groupby("LocationID").sum().reset_index()
    df_merged_in = df.merge(df_sub, left_on = "LocationID", right_on = "DOLocationID")
    df_merged_in["InCounts"] = 1
    df_incounts = df_merged_in[["LocationID", "InCounts"]].groupby("LocationID").sum().reset_index()
    df_tripcounts = df_outcounts.merge(df_incounts, on = "LocationID")
    df_tripcounts["TripCounts"] = df_tripcounts["InCounts"] - df_tripcounts["OutCounts"]
    df_triporigs = df.merge(df_tripcounts, on = "LocationID")
    bounds = np.linspace(-5000, 5000, 101)
    colors = plt.get_cmap('bwr')(np.linspace(0,1,len(bounds)+1))
    cmap = mcolors.ListedColormap(colors[1:-1])
    # set upper/lower color
    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds)-1)
    fig, ax = get_heatmap(df_triporigs, f"{desc.replace('_', ' ').title()} Trip Supply 2022-07", f"SupplyTime/{desc.lower()}_tripsupplytime_202207", overlay_chargers = False, df_chargers = None, save = save, cmap = cmap, norm = norm)
    if save:
        return df_triporigs
    return fig, ax

### Generate trip destination heatmap for given regions
def get_tripstats_time_heatmap(df_data, stats = "wait_time", hours = None, desc = "all_time", regions = None):
    if hours is not None and len(hours) > 0:
        df_sub = df_data[df_data["Hours"].isin(hours)]
    else:
        df_sub = df_data
    if regions is not None and len(regions) > 0:
        df_region_sub = df[df["borough"].isin(regions)]
    else:
        df_region_sub = df
    df_merged = df_region_sub.merge(df_sub, left_on = "LocationID", right_on = "PULocationID")
    metric = ""
    if stats == "wait_time":
        metric = "WaitingTime"
    elif stats == "trip_miles":
        metric = "trip_miles"
    else:
        metric = "trip_time"
    df_merged["TripCounts"] = df_merged[metric]
    df_tripcounts = df_merged[["LocationID", "TripCounts"]].groupby("LocationID").mean().reset_index()
    df_tripstats = df.merge(df_tripcounts, on = "LocationID")
    get_heatmap(df_tripstats, f"{desc.replace('_', ' ').title()} {stats.replace('_', ' ').title()} 2022-07", f"TripStatsTime/{desc.lower()}_{stats}_202207", overlay_chargers = False, df_chargers = None)
    return df_tripstats

### Generate histograms
def get_hist(val, title, fname, x = None):
    if x is not None:
        plt.bar(x, val)
    else:
        plt.hist(val, bins = 100, density = True)
#        plt.xscale("log")
    plt.title(title)
    plt.savefig(f"MapExplore/{SCOPE}/Histograms/{fname}.png")
    plt.clf()
    plt.close()

### Generate map for travel volumes
def get_travel_link_plot(title, fname, regions = None, overlay_chargers = True, hours = None, desc = "all_time", df_data = df_data):
    fig, ax = get_tripdest_time_heatmap(df_data, hours = hours, desc = desc, save = False, cmap = "winter_r")
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
    if hours is not None and len(hours) > 0:
        df_data_sub = df_data[df_data["Hours"].isin(hours)]
    else:
        df_data_sub = df_data
    df_triplinks = df_sub.merge(df_data_sub, on = ["PULocationID", "DOLocationID"])
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
#    df_triplinksplot = df_triplinksplot[df_triplinksplot["TripCounts"] >= 1000]
    df_triplinksplot = df_triplinksplot.sort_values("TripCounts", ascending = True)
        
    cmap = plt.cm.get_cmap("gist_heat_r") #plt.cm.OrRd #RdBu_r #hot_r
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 0.5, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    
#    fig, ax = plt.subplots()
#    df_cp.plot(ax = ax, edgecolor = "black", color = "thistle")
    links = df_triplinksplot.plot(ax = ax, cmap = my_cmap, column = "TripCounts", legend = True)
    
    if overlay_chargers:
        df_chargers_sub.plot(ax = ax, marker = "o", color = "green", markersize = 2)
    ax.set_aspect("equal")
    plt.title(f"{desc.replace('_', ' ').title()} {title}")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"MapExplore/{SCOPE}/TripLinks/{desc.lower()}_{fname}.png")
    plt.clf()
    plt.close()
    return df_triplinks

def get_boxplot(data, labels, colors, title, fname):
    assert len(data) == len(labels)
    assert len(data) == len(colors)
    
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
     
    # Creating axes instance
    bp = ax.boxplot(data, patch_artist = True, notch ='True', vert = 0, showfliers = False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
     
    # changing color and linewidth of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B', linewidth = 1.5, linestyle =":")
    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
         
    # x-axis labels
    ax.set_yticklabels(labels, wrap = True)
    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Adding title
    plt.title(title)
    plt.savefig(f"MapExplore/{SCOPE}/Boxplots/{fname}.png")
    plt.clf()
    plt.close()

### Visualize Chargers
def visualize_chargers():
    df_cp = df[df["borough"].isin(["Manhattan"])]
    fig, ax = plt.subplots()
    df_cp.plot(ax = ax, edgecolor = "black", color = "white") # thistle
    
    df_chargers_long_cp = df_chargers_long[df_chargers_long["borough"].isin(["Manhattan"])]
    df_chargers_long_level2 = df_chargers_long_cp[df_chargers_long_cp["ChargerType"] == "Level2"]
    df_chargers_long_dcFast = df_chargers_long_cp[df_chargers_long_cp["ChargerType"] == "DCFast"]
    
    sc = df_chargers_long_level2.plot(ax = ax, marker = "o", color = "green", markersize = df_chargers_long_level2["ChargerNum"] * 3, alpha = 1)
#    df_chargers_long_dcFast.plot(ax = ax, marker = "o", color = "blue", markersize = df_chargers_long_dcFast["ChargerNum"], alpha = 0.5)
    _, bins = pd.cut(df_chargers_long_level2["ChargerNum"], bins=3, precision=0, retbins=True)
    ax.add_artist(
        ax.legend(
            handles=[
                mlines.Line2D(
                    [],
                    [],
                    color="green",
                    lw=0,
                    marker="o",
                    markersize=b,
                    label=str(int(b)),
                )
                for i, b in enumerate(bins)
            ],
            loc=4,
        )
    )
    ax.set_aspect("equal")
    plt.title(f"Chargers")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"MapExplore/{SCOPE}/chargers.png")
    plt.clf()
    plt.close()

visualize_chargers()

"""
## Visualize the trip demand heatmaps
df_tripdemands = get_tripdemand_heatmap(regions = [])
get_tripdemand_heatmap(regions = ["Manhattan"])
#df_tripdemands[["LocationID", "zone", "TripCounts"]].drop_duplicates().sort_values("TripCounts", ascending = False).to_csv("MapExplore/TripDemand/trip_demands.csv", index=False)

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

## Visualize the request density across time categorized by weekday and holiday effects
df_vis = df_data.copy()
df_vis["Hours"] = df_vis["request_datetime"].apply(lambda x: x.hour)
df_vis["Weekday"] = df_vis["request_datetime"].apply(lambda x: 1 if x.weekday() <= 4 else 0)
df_vis["Holiday"] = df_vis["request_datetime"].apply(lambda x: 1 if x.day == 4 else 0)
df_vis["TripRequests"] = 1
df_vis = df_vis[["Hours", "TripRequests", "Weekday", "Holiday"]].groupby(["Hours", "Weekday", "Holiday"]).sum().reset_index()
get_hist(df_vis[(df_vis["Weekday"] == 1) & (df_vis["Holiday"] == 0)]["TripRequests"], "# Trip Requests on Weekdays (Excluding Holidays)", "trip_request_weekday", x = df_vis[(df_vis["Weekday"] == 1) & (df_vis["Holiday"] == 0)]["Hours"])
get_hist(df_vis[(df_vis["Weekday"] == 0) & (df_vis["Holiday"] == 0)]["TripRequests"], "# Trip Requests on Weekends (Excluding Holidays)", "trip_request_weekend", x = df_vis[(df_vis["Weekday"] == 0) & (df_vis["Holiday"] == 0)]["Hours"])
df_vis = df_vis[["Hours", "Holiday", "TripRequests"]].groupby(["Hours", "Holiday"]).sum().reset_index()
get_hist(df_vis[df_vis["Holiday"] == 1]["TripRequests"], "# Trip Requests on Holidays", "trip_request_holiday", x = df_vis[df_vis["Holiday"] == 1]["Hours"])
"""

## Regions to consider: Midtown Center, Penn Station/Madison Sq West, East Harlem, JFK Airport
## Weekday rush v.s. leisure hours: 7am - 11 pm v.s. 0-6 am
## Weekend/Holiday rush v.s. leisure hours: 0-1 am, 10am - 11pm v.s. 2-9 am

"""
## Visualize the trip time, trip distance, trip waiting time densities
df_vis = df_data.copy()
df_vis["Hours"] = df_vis["request_datetime"].apply(lambda x: x.hour)
df_vis["Weekday"] = df_vis["request_datetime"].apply(lambda x: 1 if x.weekday() <= 4 else 0)
df_vis["Holiday"] = df_vis["request_datetime"].apply(lambda x: 1 if x.day == 4 else 0)
df_vis["Busy"] = df_vis["Weekday"] * (1 - df_vis["Holiday"])
## Remove erroneous data
### Remove entries where driver arrives before the request (1.1% records)
df_vis = df_vis[df_vis["on_scene_datetime"] >= df_vis["request_datetime"]]
### Remove entries where trip distance is ridiculously long (0.01% records)
#df_vis = df_vis[df_vis["trip_miles"] <= 100]
### Remove trips coming from or to unknown areas
df_vis = df_vis[(df_vis["PULocationID"] < 264) & (df_vis["DOLocationID"] < 264)]

df_vis["TripTime"] = (df_vis["dropoff_datetime"] - df_vis["pickup_datetime"]).apply(lambda x: x.seconds / 60)
df_vis["WaitingTime"] = (df_vis["on_scene_datetime"] - df_vis["request_datetime"]).apply(lambda x: x.seconds / 60)

arg_lst = [([3,4,5,6], 1, "weekday_early_morning"), ([7,8,9,10], 1, "weekday_morning_rush"), ([11,12,13,14,15,16], 1, "weekday_noon"), ([17,18,19,20], 1, "weekday_evening_rush"), ([0,1,2,21,22,23], 1, "weekday_late_night"), ([2,3,4,5,6,7,8], 0, "weekend_early_morning"), ([9,10,11,12,13,14,15,16,17], 0, "weekend_daytime"), ([0,1,18,19,20,21,22,23], 0, "weekend_evening")]

for hours, busy, desc in arg_lst:
#    get_supply_time_heatmap(df_vis[df_vis["Busy"] == busy], hours = hours, desc = desc)
    get_travel_link_plot("NYC Trip Links", "nyc_trip_links", regions = None, overlay_chargers = False, hours = hours, desc = desc, df_data = df_vis[df_vis["Busy"] == busy])
"""

#for hour in [] + list(range(24)):
#    if hour is None:
#        input = None
#        desc = "all_time"
#    else:
#        input = [hour]
#        desc = hour
#    get_tripdest_time_heatmap(df_vis[df_vis["Busy"] == 1], hours = input, desc = f"weekday_{desc}")
#    get_tripdest_time_heatmap(df_vis[df_vis["Busy"] == 0], hours = input, desc = f"weekend_{desc}")
#    get_triporig_time_heatmap(df_vis[df_vis["Busy"] == 1], hours = input, desc = f"weekday_{desc}")
#    get_triporig_time_heatmap(df_vis[df_vis["Busy"] == 0], hours = input, desc = f"weekend_{desc}")
#    get_travel_link_plot("NYC Trip Links", "nyc_trip_links", regions = None, overlay_chargers = False, hours = input, desc = f"weekday_{desc}", df_data = df_vis[df_vis["Busy"] == 1])
#    get_travel_link_plot("NYC Trip Links", "nyc_trip_links", regions = None, overlay_chargers = False, hours = input, desc = f"weekend_{desc}", df_data = df_vis[df_vis["Busy"] == 0])
#    get_tripstats_time_heatmap(df_vis[df_vis["Busy"] == 1], stats = "wait_time", hours = input, desc = f"weekday_{desc}", regions = ["Manhattan", "Bronx", "Brooklyn", "Queens"])
#    get_tripstats_time_heatmap(df_vis[df_vis["Busy"] == 0], stats = "wait_time", hours = input, desc = f"weekend_{desc}", regions = ["Manhattan", "Bronx", "Brooklyn", "Queens"])
#    get_supply_time_heatmap(df_vis[df_vis["Busy"] == 1], hours = input, desc = f"weekday_{desc}")
#    get_supply_time_heatmap(df_vis[df_vis["Busy"] == 0], hours = input, desc = f"weekend_{desc}")

"""
### (138, LaGuardia Airport), (132, JFK Airport), (79, East Village), (230, Times Sq/Theatre District), (161, Midtown Center), (42, Central Harlem), (186, Penn Station), (236, Upper East Side North), (239, Upper West Side South), (74, East Harlem)
region_lst = [(161, "Midtown Center"), (186, "Penn Station"), (74, "East Harlem"), (132, "JFK Airport")]
hours_w_desc_lst = [(1, list(np.arange(7, 24)), "Weekday Rush"), (1, list(np.arange(0, 7)), "Weekday Leisure"), (0, [0, 1] + list(np.arange(10, 24)), "Weekend/Holiday Rush"), (0, list(np.arange(2, 10)), "Weekend/Holiday Leisure")]

## Draw Histograms
for region in region_lst:
    for hours_w_desc in hours_w_desc_lst:
        df_vis_sub = df_vis[(df_vis["PULocationID"] == region[0]) & (df_vis["Busy"] == hours_w_desc[0]) & (df_vis["Hours"].isin(hours_w_desc[1]))]
        title_suffix = region[1] + " " + hours_w_desc[2]
        fname_suffix = title_suffix.replace("/", " ").replace(" ", "_")
        ### Visualize trip time
        get_hist(df_vis_sub["TripTime"], f"Trip Time (Minutes)\n{title_suffix}", f"trip_time_{fname_suffix}")
        ### Visualize trip distance
        get_hist(df_vis_sub["trip_miles"], f"Trip Distance (Miles)\n{title_suffix}", f"trip_distance_{fname_suffix}")
        ### Visualize trip time
        get_hist(df_vis_sub["WaitingTime"], f"Trip Waiting Time (Minutes)\n{title_suffix}", f"wait_time_{fname_suffix}")

## Draw Boxplots
colors = ["#F75431", "#F5C4B9", "#55B42C", "#BACDB2"]
labels = ["Weekday Rush", "Weekday Non-Rush", "Weekend/Holiday Rush", "Weekend/Holiday Non-Rush"]
for region in region_lst:
    for data_type in ["TripTime", "trip_miles", "WaitingTime"]:
        data = []
        for hours_w_desc in hours_w_desc_lst:
            df_vis_sub = df_vis[(df_vis["PULocationID"] == region[0]) & (df_vis["Busy"] == hours_w_desc[0]) & (df_vis["Hours"].isin(hours_w_desc[1]))]
            data.append(df_vis_sub[data_type])
        name = data_type
        if name == "TripTime":
            name = "trip_time"
        elif name == "WaitingTime":
            name = "wait_time"
        title = f"{name.replace('_', ' ').title()} - {region[1]}"
        fname = f"{name.lower()}_{region[1]}"
        get_boxplot(data, labels, colors, title, fname)

## Visualize the travel volumes
df_triplinks = get_travel_link_plot("NYC Trip Links", "nyc_trip_links", regions = None, overlay_chargers = False)
#df_triplinks.to_csv("MapExplore/TripLinks/trip_links.csv", index = False)
get_travel_link_plot("Manhattan Trip Links", "manhattan_trip_links", regions = ["Manhattan"], overlay_chargers = False)
"""

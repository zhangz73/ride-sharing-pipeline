import holidays
from textwrap import wrap
from itertools import chain
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from joblib import Parallel, delayed
from tqdm import tqdm

## Select Scope
SCOPE = "All"
dct_key = {
    "Uber": ["HV0003"],
    "Lyft": ["HV0005"],
    "All": ["HV0003", "HV0005"]
}
## Locations of interest
###  Airport, working area, restaurant area, residential area
###     Airports: (138, LaGuardia Airport), (132, JFK Airport)
###     Working Area: (161, "Midtown Center"), (230, Times Sq/Theatre District)
###     Restaurants: (158, "Meatpacking/West Village West"), (249, "West Village"), (114, "Greenwich Village South"), (79, "East Village"), (148, "Lower East Side")
###     Residential: (238, "Upper West Side North"), (141, "Lenox Hill West"), (263, Yorkville West)
REGION = "small" # small or big
if REGION == "small":
    LOCATIONS_ID_OF_INTEREST = [132, 230, 79, 238]
else:
    LOCATIONS_ID_OF_INTEREST = [138, 132, 161, 230, 158, 249, 114, 79, 148, 238, 141, 263]
LOCATION_MAP = {}
for i in range(len(LOCATIONS_ID_OF_INTEREST)):
    LOCATIONS_ID = LOCATIONS_ID_OF_INTEREST[i]
    LOCATION_MAP[LOCATIONS_ID] = i

## Time of interest
TIME_RANGE = (8, 20)
TIME_FREQ = 5 # E.g. 5 minutes per decision epoch
PACK_SIZE = 40
MAX_MILES = 149
TOTAL_CARS_ORIG = 5000
TOTAL_CARS_NEW = 200
CHARGING_RATE = 0.128#[0.128, 0.833]
NUM_BATTERY_LEVELS = 264
NUM_PLUGS = TOTAL_CARS_NEW + TOTAL_CARS_NEW ** 0.5
CHARGING_RATE_DIS = 1#[1, 5]
SCENARIO_NAME = f"{TOTAL_CARS_NEW}car{len(LOCATIONS_ID_OF_INTEREST)}region_nyc"

## Compute time horizon
TIME_HORIZON = int((TIME_RANGE[1] - TIME_RANGE[0] + 1) * 60 / TIME_FREQ)
NUM_TS_PER_HOUR = 60 // TIME_FREQ

## Load data from files
df_data = pd.read_parquet("Data/MapData/fhvhv_tripdata_2022-07.parquet")

df_data["trip_time"] = (df_data["dropoff_datetime"] - df_data["pickup_datetime"]).apply(lambda x: x.days / 24 / 60 + x.seconds / 60)
df_data["hour"] = df_data["request_datetime"].apply(lambda x: x.hour)
df_data["date"] = df_data["request_datetime"].apply(lambda x: f"{x.year}-{x.month}-{x.day}")
df_data["day_of_week"] = df_data["request_datetime"].apply(lambda x: x.weekday())
df_data["holiday"] = df_data["request_datetime"].apply(lambda x: 1 if x.day == 4 else 0)

## Filter data of interest
df_data = df_data[(df_data["PULocationID"].isin(LOCATIONS_ID_OF_INTEREST)) & (df_data["DOLocationID"].isin(LOCATIONS_ID_OF_INTEREST))]
df_data = df_data[(df_data["hour"] >= TIME_RANGE[0]) & (df_data["hour"] < TIME_RANGE[1])]
df_data = df_data[df_data["day_of_week"].isin([0, 1, 2, 3])]
df_data = df_data[df_data["holiday"] == 0]
df_data = df_data[df_data["hvfhs_license_num"].isin(dct_key[SCOPE])]

## Remove erroneous data
### Remove entries where driver arrives before the request (1.1% records)
df_data = df_data[df_data["on_scene_datetime"] >= df_data["request_datetime"]]
### Remove entries where trip distance is ridiculously long (0.01% records)
df_data = df_data[df_data["trip_miles"] <= 100]

## Define columns of interests
df_data["Count"] = 1
df_data["Payoff"] = df_data["base_passenger_fare"]
df_data["TripTime"] = df_data["trip_time"]
df_data["Distance"] = df_data["trip_miles"]
df_data["Origin"] = df_data["PULocationID"].apply(lambda x: LOCATION_MAP[x])
df_data["Destination"] = df_data["DOLocationID"].apply(lambda x: LOCATION_MAP[x])

def get_travel_time(weekdays = [0, 1, 2, 3], remove_holiday = True):
    travel_time = df_data[df_data["day_of_week"].isin(weekdays)]
    if remove_holiday:
        travel_time = travel_time[travel_time["holiday"] == 0]
    travel_time = travel_time[["PULocationID", "DOLocationID", "hour", "trip_time"]].groupby(["PULocationID", "DOLocationID", "hour"]).mean().reset_index()
    return travel_time

def get_arrival_rate(weekdays = [0, 1, 2, 3], remove_holiday = True):
    travel_time = df_data[df_data["day_of_week"].isin(weekdays)]
    if remove_holiday:
        travel_time = travel_time[travel_time["holiday"] == 0]
    travel_time["request_num"] = 1
    arrival_rate = travel_time[["PULocationID", "DOLocationID", "hour", "date", "request_num"]].groupby(["PULocationID", "DOLocationID", "date", "hour"]).sum().reset_index()
    arrival_rate = arrival_rate[["PULocationID", "DOLocationID", "hour", "request_num"]].groupby(["PULocationID", "DOLocationID", "hour"]).mean().reset_index()
    arrival_rate = arrival_rate.sort_values("request_num", ascending = False)
    return arrival_rate

def get_battery_consumption_rate():
    avg_miles_per_min = (df_data["trip_miles"] / df_data["trip_time"]).mean()
    avg_kwh_per_min = avg_miles_per_min / MAX_MILES * PACK_SIZE
    return avg_kwh_per_min

def get_attr(attr_name, agg_by_day = False, scale_down_by_car = False, scale_by_freq = None, scale_factor = 1):
    df = df_data.copy()
    if agg_by_day:
        df = df[["Origin", "Destination", "hour", "date", attr_name]].groupby(["Origin", "Destination", "hour", "date"]).sum().reset_index()
    df = df[["Origin", "Destination", "hour", attr_name]].groupby(["Origin", "Destination", "hour"]).mean().reset_index()
    if scale_down_by_car:
        df[attr_name] = df[attr_name] / TOTAL_CARS_ORIG * TOTAL_CARS_NEW
    if scale_by_freq == "down":
        df[attr_name] = df[attr_name] / TIME_FREQ
    elif scale_by_freq == "up":
        df[attr_name] = df[attr_name] / 60 * TIME_FREQ
    df[attr_name] = df[attr_name] * scale_factor
    df_ret = None
    for ts in range(NUM_TS_PER_HOUR):
        df_curr = df.copy()
        df_curr["T"] = df["hour"].apply(lambda x: (x - TIME_RANGE[0]) * NUM_TS_PER_HOUR + ts)
        if df_ret is None:
            df_ret = df_curr
        else:
            df_ret = pd.concat([df_ret, df_curr], axis = 0)
#    df = df.loc[df.index.repeat(NUM_TS_PER_HOUR)].reset_index()
#    df = df.sort_values("hour", ascending = True)
#    df["T"] = np.arange(TIME_HORIZON)
    df_ret = df_ret[["T", "Origin", "Destination", attr_name]].sort_values("T")
    return df_ret

def get_charging_cost(cost_rate_per_min_lst):
    cost_rate_lst = []
    T_lst = []
    region_lst = []
    for cost_rate_per_min in cost_rate_per_min_lst:
        cost_rate, tup = cost_rate_per_min
        lo, hi = tup
        for region in range(len(LOCATIONS_ID_OF_INTEREST)):
            cost_rate_lst += [cost_rate * TIME_FREQ * CHARGING_RATE] * (hi - lo)
            T_lst += list(range(lo, hi))
            region_lst += [region for _ in range(hi - lo)]
    dct = {"T": T_lst, "Payoff": cost_rate_lst, "Region": region_lst}
    df = pd.DataFrame.from_dict(dct)
    df["Rate"] = CHARGING_RATE_DIS
    df = df.sort_values("T")
    return df

def get_region_battery_car_df():
    car_num_per_region = TOTAL_CARS_NEW / len(LOCATIONS_ID_OF_INTEREST)
    region_lst = []
    battery_lst = []
    num_lst = []
    for region in range(len(LOCATIONS_ID_OF_INTEREST)):
        region_lst.append(region)
        battery_lst.append(NUM_BATTERY_LEVELS // 2)
        num_lst.append(car_num_per_region)
    dct = {"region": region_lst, "battery": battery_lst, "num": num_lst}
    return pd.DataFrame.from_dict(dct)

def get_region_rate_plug_df():
    plug_num_per_region = NUM_PLUGS // len(LOCATIONS_ID_OF_INTEREST)
    region_lst = []
    rate_lst = []
    num_lst = []
    for region in range(len(LOCATIONS_ID_OF_INTEREST)):
        region_lst.append(region)
        rate_lst.append(CHARGING_RATE_DIS)
        num_lst.append(plug_num_per_region)
    dct = {"region": region_lst, "rate": rate_lst, "num": num_lst}
    return pd.DataFrame.from_dict(dct)

def get_numcars_prev():
    df = df_data.copy()
    start = pd.Timestamp("2022-07-01 00:00:00")
    end = pd.Timestamp("2022-07-31 23:59:59")
    ts_lst = pd.date_range(start, end, freq = "S")
    cnt_arr = np.zeros(len(ts_lst))
    df["pickup_idx"] = df["pickup_datetime"].apply(lambda x: (x - start).days * 24 * 3600 + (x - start).seconds)
    df["dropoff_idx"] = df["dropoff_datetime"].apply(lambda x: (x - start).days * 24 * 3600 + (x - start).seconds)
    for i in tqdm(range(df.shape[0])):
        s, e = df.iloc[i]["pickup_idx"], df.iloc[i]["dropoff_idx"]
        cnt_arr[s:e] += 1
    df_ret = pd.DataFrame.from_dict({"ts": ts_lst, "cnt": cnt_arr})
    df_ret["time_of_day"] = df_ret["ts"].apply(lambda x: f"{x.hour}-{x.minute}-{x.second}")
    df_ret = df_ret.groupby("time_of_day").mean().reset_index()
    return df_ret["cnt"].median()

def get_numcars():
    df = df_data.copy()
    cnt_arr_all = np.zeros((31, TIME_HORIZON))
    df["T_PU"] = df["pickup_datetime"].apply(lambda x: (x.hour - TIME_RANGE[0]) * NUM_TS_PER_HOUR + x.minute // TIME_FREQ)
    df["T_DO"] = df["dropoff_datetime"].apply(lambda x: (x.hour - TIME_RANGE[0]) * NUM_TS_PER_HOUR + x.minute // TIME_FREQ)
    df["day"] = df["pickup_datetime"].apply(lambda x: x.day - 0)
    cnt_arr_avg = np.zeros(TIME_HORIZON)
    for i in tqdm(range(df.shape[0])):
        s, e = df.iloc[i]["T_PU"], df.iloc[i]["T_DO"]
        d = df.iloc[i]["day"]
        cnt_arr_all[d, s:e] += 1
    cnt_arr = np.mean(cnt_arr_all, axis = 0)
    df_ret = pd.DataFrame.from_dict({"T": np.arange(TIME_HORIZON), "cnt": cnt_arr})
    return df_ret["cnt"].median()

#avg_kwh_per_min = get_battery_consumption_rate()
#print(avg_kwh_per_min)

#travel_time = get_travel_time()
#arrival_rate = get_arrival_rate()
#print(travel_time)
#print(arrival_rate)

## Get median num of cars
median_car_cnt = get_numcars()
scale_factor = TOTAL_CARS_NEW / median_car_cnt
print(median_car_cnt, scale_factor)

## Create trip_demand_df
trip_demand_df = get_attr("Count", agg_by_day = True, scale_down_by_car = False, scale_by_freq = "up", scale_factor = scale_factor)
print(trip_demand_df)

## Create payoff_df
payoff_df = get_attr("Payoff", agg_by_day = False, scale_down_by_car = False)
payoff_df["Type"] = "Travel"
payoff_df["Pickup"] = 1
payoff_df["Region"] = None
payoff_df["Rate"] = None
charging_df = get_charging_cost([(0.19297, (0, 12)), (0.16631, (12, 72)), (0.38498, (72, 144))])
charging_df["Type"] = "Charge"
charging_df["Pickup"] = None
charging_df["Origin"] = None
charging_df["Destination"] = None
payoff_df = pd.concat([payoff_df, charging_df], axis = 0, ignore_index = True)

## Create map df
trip_time_df = get_attr("TripTime", agg_by_day = False, scale_down_by_car = False, scale_by_freq = "down")
trip_time_df["TripTime"] = trip_time_df["TripTime"].round(0).astype(int)
distance_df = get_attr("Distance", agg_by_day = False, scale_down_by_car = False)
distance_df["Distance"] = distance_df["Distance"].round(0).astype(int)
map_df = trip_time_df.merge(distance_df, on = ["T", "Origin", "Destination"])
region_battery_car_df = get_region_battery_car_df()
region_rate_plug_df = get_region_rate_plug_df()

## Write to files
region_battery_car_df.to_csv(f"Data/RegionBatteryCar/region_battery_car_{SCENARIO_NAME}.tsv", index = False, sep = "\t")
region_rate_plug_df.to_csv(f"Data/RegionRatePlug/region_rate_plug_{SCENARIO_NAME}.tsv", index = False, sep = "\t")
trip_demand_df.to_csv(f"Data/TripDemand/trip_demand_{SCENARIO_NAME}.tsv", index = False, sep = "\t")
payoff_df.to_csv(f"Data/Payoff/payoff_{SCENARIO_NAME}.tsv", index = False, sep = "\t")
map_df.to_csv(f"Data/Map/map_{SCENARIO_NAME}.tsv", index = False, sep = "\t")

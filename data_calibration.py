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

## Load data from files
df_data = pd.read_parquet("Data/Map/fhvhv_tripdata_2022-07.parquet")
df_data = df_data[df_data["hvfhs_license_num"].isin(dct_key[SCOPE])]
df_data["trip_time"] = (df_data["dropoff_datetime"] - df_data["pickup_datetime"]).apply(lambda x: x.days / 24 / 60 + x.seconds / 60)
df_data["hour"] = df_data["request_datetime"].apply(lambda x: x.hour)
df_data["date"] = df_data["request_datetime"].apply(lambda x: f"{x.year}-{x.month}-{x.day}")
df_data["day_of_week"] = df_data["request_datetime"].apply(lambda x: x.weekday())
df_data["holiday"] = df_data["request_datetime"].apply(lambda x: 1 if x.day == 4 else 0)
## Remove erroneous data
### Remove entries where driver arrives before the request (1.1% records)
df_data = df_data[df_data["on_scene_datetime"] >= df_data["request_datetime"]]
### Remove entries where trip distance is ridiculously long (0.01% records)
df_data = df_data[df_data["trip_miles"] <= 100]
PACK_SIZE = 40
MAX_MILES = 149

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

avg_kwh_per_min = get_battery_consumption_rate()
print(avg_kwh_per_min)

travel_time = get_travel_time()
arrival_rate = get_arrival_rate()
print(travel_time)
print(arrival_rate)

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

N = 300
TIME_HORIZON = 288

df = pd.read_csv("Tables/table_300car_10region_3000charger_chargetime3_5min_halfcharged_nyc_combo_fullday_ppo_reduced.csv")
df["t"] = df["t"] % TIME_HORIZON
df["num_cars_outside_region_3"] = 0
for i in range(10):
    if i != 3:
        df["num_cars_outside_region_3"] += df[f"num_cars_region_{i}"]
df = df[["t", "num_cars_region_3", "num_cars_outside_region_3"]].copy()
#df = df[["hour"] + [f"num_cars_region_{i}" for i in range(10)]].copy()
df = df.groupby("t").mean().reset_index()

fig, ax = plt.subplots()
interval = int(24 * 60 / TIME_HORIZON)
start = datetime(1900, 1, 1, 0, 0, 0)
step = timedelta(minutes = interval)
date_arr = [start]
for _ in range(df.shape[0] - 1):
    date_arr.append(date_arr[-1] + step)
#for i in range(10):
#    color = "gray"
#    if i == 3:
#        label = "Midtown Region"
#        color = "red"
#    elif i == 0:
#        label = "Non-Midtown Regions"
#    else:
#        label = None
#    plt.plot(df["hour"], df[f"num_cars_region_{i}"], label = label, color = color)
plt.plot(date_arr, df["num_cars_region_3"] / N * 100, label = "Midtown Region", color = "red", alpha = 0.5)
#plt.plot(df["hour"], df["num_cars_outside_region_3"], label = "Non-Midtown Regions", color = "gray")
#plt.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.gcf().axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.autofmt_xdate()
plt.xlabel("Time of day")
plt.ylabel("Fraction of cars")
plt.savefig("TablePlots/midtown_cars.png")
plt.clf()
plt.close()

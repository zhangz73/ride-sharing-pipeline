import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import Utils.train as train
import Utils.setup as setup
import Utils.mdp as mdp

def get_table_plot(args, json_name):
    if "neural" in args and "state_reduction" in args["neural"] and args["neural"]["state_reduction"]:
        descriptor = "reduced"
    else:
        descriptor = "full"
    map = setup.Map(**args["map"])
    time_horizon = args["mdp"]["time_horizon"]
    trip_demands = setup.TripDemands(time_horizon = time_horizon, **args["trip_demand"])
    reward_query = mdp.Reward(**args["reward"])
    markov_decision_process = mdp.MarkovDecisionProcess(map, trip_demands, reward_query, **args["mdp"])
    solver_type = args["solver"]["type"]
    report_factory = train.ReportFactory(markov_decision_process = markov_decision_process)
    df_table_all = pd.read_csv(f"Tables/table_{json_name}_{descriptor}.csv")
    if "d-closest" in json_name:
        df_table_all["frac_rerouting_cars"] = 0
        df_table_all["frac_idling_cars"] = 1 - df_table_all["frac_passenger-carrying_cars"] - df_table_all["frac_rerouting_cars"] - df_table_all["frac_charging_cars"]
    report_factory.visualize_table(df_table_all, f"{json_name}_{descriptor}", detailed = True)
    vis_day = 3
    t_range = (time_horizon * vis_day, time_horizon * (vis_day + 1))
    df_table_all = df_table_all[(df_table_all["t"] >= t_range[0]) & (df_table_all["t"] < t_range[1])]
    report_factory.visualize_table(df_table_all, f"{json_name}_{descriptor}_singleday", detailed = True)
    return None

JSON_NAME = "300car_10region_3000charger_5min_halfcharged_nyc_combo_fullday_d-closest"

with open(f"Args/{JSON_NAME}.json", "r") as f:
    args = json.load(f)

get_table_plot(args, json_name = JSON_NAME)

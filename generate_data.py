import numpy as np
import pandas as pd
import Utils.setup as setup

def generate_data():
    pass

def lst_2_df(lst, headers):
    dct = {}
    for name in headers:
        dct[name] = []
    for i in range(len(lst)):
        for j in range(len(headers)):
            name = headers[j]
            val = lst[i][j]
            dct[name].append(val)
    df = pd.DataFrame.from_dict(dct)
    return df

def write_data(scenario_name, num_layers, time_horizon, map_system = "grid", payoff = {"pickup": 1, "reroute": -0.5, "charge": {2: -0.5}}, region_battery_car = [(0, 2, 1)], region_rate_plug = [(0, 2, 1)], trip_demand = [(0, 0, 1, 1)]):
    assert map_system in ["grid", "hexagon"]
    map = setup.Map(map_system = map_system, num_layers = num_layers)
    regions = map.get_regions()
    
    region_battery_car_df = lst_2_df(region_battery_car, ["region", "battery", "num"])
    region_rate_plug_df = lst_2_df(region_rate_plug, ["region", "rate", "num"])
    trip_demand_df = lst_2_df(trip_demand, ["T", "Origin", "Destination", "Count"])
    
    payoff_headers = ["T", "Type", "Pickup", "Origin", "Destination", "Region", "Rate", "Payoff"]
    payoff_lst = []
    for t in range(time_horizon):
        ## Populate travel payoffs
        for origin in regions:
            for dest in regions:
                steps = map.steps_to_location(origin, dest)
                pickup_tup = (t, "Travel", 1, origin, dest, None, None, payoff["pickup"] * steps)
                reroute_tup = (t, "Travel", 0, origin, dest, None, None, payoff["reroute"] * steps)
                payoff_lst.append(pickup_tup)
                payoff_lst.append(reroute_tup)
        ## Populate charging payoffs
        for tup in region_rate_plug:
            region, rate, _ = tup
            charge_tup = (t, "Travel", None, None, None, region, rate, payoff["charge"][rate] * steps)
            payoff_lst.append(charge_tup)
    payoff_df = lst_2_df(payoff_lst, payoff_headers)
    
    ## Write data to files
    region_battery_car_df.to_csv(f"Data/RegionBatteryCar/region_battery_car_{scenario_name}.tsv", index = False, sep = "\t")
    region_rate_plug_df.to_csv(f"Data/RegionRatePlug/region_rate_plug_{scenario_name}.tsv", index = False, sep = "\t")
    trip_demand_df.to_csv(f"Data/TripDemand/trip_demand_{scenario_name}.tsv", index = False, sep = "\t")
    payoff_df.to_csv(f"Data/Payoff/payoff_{scenario_name}.tsv", index = False, sep = "\t")

write_data(scenario_name = "2car5grid", num_layers = 3, time_horizon = 16, map_system = "grid", payoff = {"pickup": 1, "reroute": -0.5, "charge": {3: -0.5}}, region_battery_car = [(0, 8, 1), (24, 5, 1)], region_rate_plug = [(6, 3, 1), (8, 3, 1), (17, 3, 1)], trip_demand = [(0, 1, 7, 1), (5, 11, 13, 1), (5, 16, 5, 1), (7, 7, 12, 1), (7, 7, 8, 1), (8, 12, 14, 1), (9, 16, 18, 1), (11, 7, 23, 1), (12, 13, 16, 1)])

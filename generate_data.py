import numpy as np
import pandas as pd
import Utils.setup as setup
from tqdm import tqdm

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

def write_data_regular_map(scenario_name, num_layers, time_horizon, map_system = "grid", payoff = {"pickup": 1, "reroute": -0.5, "charge": {2: -0.5}}, region_battery_car = [(0, 2, 1)], region_rate_plug = [(0, 2, 1)], trip_demand = [(0, 0, 1, 1)]):
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

def write_data_traffic_map(scenario_name, num_regions, time_horizon, payoff = {"pickup": 1, "reroute": -0.5, "charge": {2: -0.5}}, region_battery_car = [(0, 2, 1)], region_rate_plug = [(0, 2, 1)], arrival_rate = [(0, [1, 2])], transition_prob = [(0, [[0.5, 0.5], [0.5, 0.5]])], traffic_time = [(0, [[1, 1], [1, 1]])]):
    regions = list(range(num_regions))
    region_battery_car_df = lst_2_df(region_battery_car, ["region", "battery", "num"])
    region_rate_plug_df = lst_2_df(region_rate_plug, ["region", "rate", "num"])
    
    ## Construct trip demand df
    ## TODO: Implement it!!!
    print("Constructing trip demand...")
    idx_arrival = len(arrival_rate) - 1
    idx_trans = len(transition_prob) - 1
    trip_demand = []
    for t in tqdm(range(time_horizon - 1, -1, -1)):
        ts_arrival = arrival_rate[idx_arrival][0]
        ts_trans = transition_prob[idx_trans][0]
        while t < ts_arrival and idx_arrival >= 0:
            idx_arrival -= 1
            ts_arrival = arrival_rate[idx_arrival][0]
        while t < ts_trans and idx_trans >= 0:
            idx_trans -= 1
            ts_trans = transition_prob[idx_trans][0]
        if ts_arrival >= 0 and ts_trans >= 0:
            arrival_rate_vec = arrival_rate[idx_arrival][1]
            transition_prob_mat = transition_prob[idx_trans][1]
            for origin in range(num_regions):
                lam = arrival_rate_vec[origin]
                for dest in range(num_regions):
                    cnt = transition_prob_mat[origin][dest] * lam
                    if cnt > 0:
                        trip_demand.append((t, origin, dest, cnt))
        else:
            break
    trip_demand_df = lst_2_df(trip_demand, ["T", "Origin", "Destination", "Count"])
    
    ## Construct payoff df
    print("Constructing payoff...")
    payoff_headers = ["T", "Type", "Pickup", "Origin", "Destination", "Region", "Rate", "Payoff"]
    payoff_lst = []
    for t in tqdm(range(time_horizon)):
        ## Populate travel payoffs
        for origin in regions:
            for dest in regions:
                pickup_tup = (t, "Travel", 1, origin, dest, None, None, 1)
                reroute_tup = (t, "Travel", 0, origin, dest, None, None, 1)
                payoff_lst.append(pickup_tup)
                payoff_lst.append(reroute_tup)
        ## Populate charging payoffs
        for tup in region_rate_plug:
            region, rate, _ = tup
            charge_tup = (t, "Travel", None, None, None, region, rate, payoff["charge"][rate])
            payoff_lst.append(charge_tup)
    payoff_df = lst_2_df(payoff_lst, payoff_headers)
    
    ## Construct map df
    ## T, Origin, Destination, Distance, TripTime
    ## TODO: Implement it!!!
    print("Constructing map...")
    idx_traffic = len(traffic_time) - 1
    map_sys = []
    for t in tqdm(range(time_horizon - 1, -1, -1)):
        ts_traffic = traffic_time[idx_traffic][0]
        while t < ts_traffic and idx_traffic >= 0:
            idx_traffic -= 1
            ts_traffic = traffic_time[idx_traffic][0]
        if ts_traffic >= 0:
            traffic_time_mat = traffic_time[idx_traffic][1]
            for origin in range(num_regions):
                for dest in range(num_regions):
                    trip_time = traffic_time_mat[origin][dest]
                    #map_sys.append((t, origin, dest, 1, trip_time))
                    map_sys.append((t, origin, dest, trip_time, trip_time))
        else:
            break
    map_df = lst_2_df(map_sys, ["T", "Origin", "Destination", "Distance", "TripTime"])
    
    ## Write data to files
    print("Writing results...")
    region_battery_car_df.to_csv(f"Data/RegionBatteryCar/region_battery_car_{scenario_name}.tsv", index = False, sep = "\t")
    region_rate_plug_df.to_csv(f"Data/RegionRatePlug/region_rate_plug_{scenario_name}.tsv", index = False, sep = "\t")
    trip_demand_df.to_csv(f"Data/TripDemand/trip_demand_{scenario_name}.tsv", index = False, sep = "\t")
    payoff_df.to_csv(f"Data/Payoff/payoff_{scenario_name}.tsv", index = False, sep = "\t")
    map_df.to_csv(f"Data/Map/map_{scenario_name}.tsv", index = False, sep = "\t")

def generate_spatialtemporal_star_to_complete_network(num_regions = 4, xi = 0.5):
    A_star = np.zeros((num_regions, num_regions))
    A_star[0,1:] = 1 / (num_regions - 1)
    A_star[1:,0] = 1
    A_complete = np.ones((num_regions, num_regions)) / (num_regions - 1)
    np.fill_diagonal(A_complete, 0)
    tau_star = np.ones((num_regions, num_regions))
    tau_star[1:,0] = 1 #np.arange(1, num_regions)
    tau_star[0,1:] = 1 #np.arange(1, num_regions)
    np.fill_diagonal(tau_star, 0)
    tau_complete = np.ones((num_regions, num_regions))
    np.fill_diagonal(tau_complete, 0)
    A = xi * A_complete + (1 - xi) * A_star
    tau = xi * tau_complete + (1 - xi) * tau_star
    tau = tau.astype(int)
    lam = np.sum(A, axis = 1)
    p = A / lam[:,np.newaxis]
    return lam, p, tau

#write_data_regular_map(scenario_name = "2car5grid", num_layers = 3, time_horizon = 16, map_system = "grid", payoff = {"pickup": 1, "reroute": -0.5, "charge": {3: -0.5}}, region_battery_car = [(0, 8, 1), (24, 5, 1)], region_rate_plug = [(6, 3, 1), (8, 3, 1), (17, 3, 1)], trip_demand = [(0, 1, 7, 1), (5, 11, 13, 1), (5, 16, 5, 1), (7, 7, 12, 1), (7, 7, 8, 1), (8, 12, 14, 1), (9, 16, 18, 1), (11, 7, 23, 1), (12, 13, 16, 1)])

## 1000 cars 5 regions
#lam_1 = [1.8, 1.8, 1.8, 1.8, 18]
#lam_2 = [12, 8, 8, 8, 2]
#lam_3 = [2, 2, 2, 22, 2]
#p_1 = [[0.6, 0.1, 0, 0.3, 0], [0.1, 0.6, 0, 0.3, 0], [0, 0, 0.7, 0.3, 0], [0.2, 0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.1, 0]]
#p_2 = [[0.1, 0, 0, 0.9, 0], [0, 0.1, 0, 0.9, 0], [0, 0, 0.1, 0.9, 0], [0.05, 0.05, 0.05, 0.8, 0.05], [0, 0, 0, 0.9, 0.1]]
#p_3 = [[0.9, 0.05, 0, 0.05, 0], [0.05, 0.9, 0, 0.05, 0], [0, 0, 0.9, 0.1, 0], [0.3, 0.3, 0.3, 0.05, 0.05], [0, 0, 0, 0.1, 0.9]]
#tau = [[9, 15, 75, 12, 24], [15, 6, 66, 6, 18], [75, 66, 6, 60, 39], [12, 6, 60, 9, 15], [24, 18, 39, 15, 12]]
#write_data_traffic_map("1000car5region", 5, 360, payoff = {"pickup": 1, "reroute": 0, "charge": {2: -0.5}}, region_battery_car = [(0, 2, 200), (1, 2, 200), (2, 2, 200), (3, 2, 200), (4, 2, 200)], region_rate_plug = [(0, 2, 1)], arrival_rate = [(0, lam_1), (120, lam_2), (240, lam_3)], transition_prob = [(0, p_1), (120, p_2), (240, p_3)], traffic_time = [(0, tau)])

## 10 cars 5 regions
#lam_1 = [0.18, 0.18, 0.18, 0.18, 1.8]
#lam_2 = [1.2, 0.8, 0.8, 0.8, 0.2]
#lam_3 = [0.2, 0.2, 0.2, 2.2, 0.2]
#p_1 = [[0.6, 0.1, 0, 0.3, 0], [0.1, 0.6, 0, 0.3, 0], [0, 0, 0.7, 0.3, 0], [0.2, 0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.1, 0]]
#p_2 = [[0.1, 0, 0, 0.9, 0], [0, 0.1, 0, 0.9, 0], [0, 0, 0.1, 0.9, 0], [0.05, 0.05, 0.05, 0.8, 0.05], [0, 0, 0, 0.9, 0.1]]
#p_3 = [[0.9, 0.05, 0, 0.05, 0], [0.05, 0.9, 0, 0.05, 0], [0, 0, 0.9, 0.1, 0], [0.3, 0.3, 0.3, 0.05, 0.05], [0, 0, 0, 0.1, 0.9]]
#tau = [[1, 2, 8, 1, 2], [2, 1, 7, 1, 2], [8, 7, 1, 6, 4], [1, 1, 6, 1, 2], [2, 2, 4, 2, 1]]
#write_data_traffic_map("10car5region", 5, 36, payoff = {"pickup": 1, "reroute": 0, "charge": {2: -0.5}}, region_battery_car = [(0, 2, 2), (1, 2, 2), (2, 2, 2), (3, 2, 2), (4, 2, 2)], region_rate_plug = [(0, 2, 1)], arrival_rate = [(0, lam_1), (12, lam_2), (24, lam_3)], transition_prob = [(0, p_1), (120, p_2), (240, p_3)], traffic_time = [(0, tau)])


### 100 cars 3 regions
#lam_1 = [3, 20, 8]
#p_1 = [[0.6, 0.1, 0.3], [0.1, 0.6, 0.3], [0.1, 0.7, 0.2]]
#tau = [[1, 2, 3], [2, 1, 3], [3, 3, 1]]
#write_data_traffic_map("100car3region", 3, 10, payoff = {"pickup": 1, "reroute": 0, "charge": {2: -0.5}}, region_battery_car = [(0, 0, 30), (1, 0, 40), (2, 0, 30)], region_rate_plug = [(0, 2, 1)], arrival_rate = [(0, lam_1)], transition_prob = [(0, p_1)], traffic_time = [(0, tau)])

## Spatial-Temporal Star-To-Complete Network (ST-STC)
### 12 cars 4 regions
xi = 1
num_regions = 4
num_cars = 12
time_horizon = 48
rate = 24
pack_size = 24
num_chargers = 0 #48
lam, p, tau = generate_spatialtemporal_star_to_complete_network(num_regions = num_regions, xi = xi)
lam = lam * num_cars / np.sum(lam)
name = f"st-stc_{num_cars}car{num_regions}region{num_chargers}chargers_xi={xi}"
write_data_traffic_map(name, num_regions, time_horizon, payoff = {"pickup": 1, "reroute": 0, "charge": {rate: 0}}, region_battery_car = [(x, pack_size - 1, num_cars // num_regions) for x in range(num_regions)], region_rate_plug = [(x, rate, num_chargers // num_regions) for x in range(num_regions)], arrival_rate = [(0, lam)], transition_prob = [(0, p)], traffic_time = [(0, tau)])

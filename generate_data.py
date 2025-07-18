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
                reroute_tup = (t, "Travel", 0, origin, dest, None, None, 0)
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

### 1000 cars 5 regions
#lam_1 = [1.8, 1.8, 1.8, 1.8, 18]
#lam_2 = [12, 8, 8, 8, 2]
#lam_3 = [2, 2, 2, 22, 2]
#p_1 = [[0.6, 0.1, 0, 0.3, 0], [0.1, 0.6, 0, 0.3, 0], [0, 0, 0.7, 0.3, 0], [0.2, 0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.1, 0]]
#p_2 = [[0.1, 0, 0, 0.9, 0], [0, 0.1, 0, 0.9, 0], [0, 0, 0.1, 0.9, 0], [0.05, 0.05, 0.05, 0.8, 0.05], [0, 0, 0, 0.9, 0.1]]
#p_3 = [[0.9, 0.05, 0, 0.05, 0], [0.05, 0.9, 0, 0.05, 0], [0, 0, 0.9, 0.1, 0], [0.3, 0.3, 0.3, 0.05, 0.05], [0, 0, 0, 0.1, 0.9]]
#tau = [[9, 15, 75, 12, 24], [15, 6, 66, 6, 18], [75, 66, 6, 60, 39], [12, 6, 60, 9, 15], [24, 18, 39, 15, 12]]
#write_data_traffic_map("1000car5region", 5, 360, payoff = {"pickup": 1, "reroute": 0, "charge": {2: 0}}, region_battery_car = [(0, 0, 200), (1, 0, 200), (2, 0, 200), (3, 0, 200), (4, 0, 200)], region_rate_plug = [(0, 2, 1)], arrival_rate = [(0, lam_1), (120, lam_2), (240, lam_3)], transition_prob = [(0, p_1), (120, p_2), (240, p_3)], traffic_time = [(0, tau)])

### 10000 cars 5 regions
#lam_1 = [50, 50, 50, 50, 540]
#lam_2 = [360, 240, 240, 240, 60]
#lam_3 = [60, 60, 60, 660, 60]
#p_1 = [[0.6, 0.1, 0, 0.3, 0], [0.1, 0.6, 0, 0.3, 0], [0, 0, 0.7, 0.3, 0], [0.2, 0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.1, 0]]
#p_2 = [[0.1, 0, 0, 0.9, 0], [0, 0.1, 0, 0.9, 0], [0, 0, 0.1, 0.9, 0], [0.05, 0.05, 0.05, 0.8, 0.05], [0, 0, 0, 0.9, 0.1]]
#p_3 = [[0.9, 0.05, 0, 0.05, 0], [0.05, 0.9, 0, 0.05, 0], [0, 0, 0.9, 0.1, 0], [0.3, 0.3, 0.3, 0.05, 0.05], [0, 0, 0, 0.1, 0.9]]
#tau = [[3, 5, 25, 4, 8], [5, 2, 22, 3, 6], [25, 22, 2, 20, 13], [4, 2, 20, 3, 5], [8, 6, 13, 5, 4]]
#write_data_traffic_map("10000car5region", 5, 120, payoff = {"pickup": 1, "reroute": 0, "charge": {2: 0}}, region_battery_car = [(0, 0, 2000), (1, 0, 2000), (2, 0, 2000), (3, 0, 2000), (4, 0, 2000)], region_rate_plug = [(0, 2, 0)], arrival_rate = [(0, lam_1), (40, lam_2), (80, lam_3)], transition_prob = [(0, p_1), (40, p_2), (80, p_3)], traffic_time = [(0, tau)])

# 10 cars 5 regions
#lam_1 = [0.18, 0.18, 0.18, 0.18, 1.8]
#lam_2 = [1.2, 0.8, 0.8, 0.8, 0.2]
#lam_3 = [0.2, 0.2, 0.2, 2.2, 0.2]
#p_1 = [[0.6, 0.1, 0, 0.3, 0], [0.1, 0.6, 0, 0.3, 0], [0, 0, 0.7, 0.3, 0], [0.2, 0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.1, 0]]
#p_2 = [[0.1, 0, 0, 0.9, 0], [0, 0.1, 0, 0.9, 0], [0, 0, 0.1, 0.9, 0], [0.05, 0.05, 0.05, 0.8, 0.05], [0, 0, 0, 0.9, 0.1]]
#p_3 = [[0.9, 0.05, 0, 0.05, 0], [0.05, 0.9, 0, 0.05, 0], [0, 0, 0.9, 0.1, 0], [0.3, 0.3, 0.3, 0.05, 0.05], [0, 0, 0, 0.1, 0.9]]
#tau = [[1, 2, 8, 1, 2], [2, 1, 7, 1, 2], [8, 7, 1, 6, 4], [1, 1, 6, 1, 2], [2, 2, 4, 2, 1]]
#write_data_traffic_map("10car5region", 5, 36, payoff = {"pickup": 1, "reroute": 0, "charge": {2: 0}}, region_battery_car = [(0, 2, 2), (1, 2, 2), (2, 2, 2), (3, 2, 2), (4, 2, 2)], region_rate_plug = [(0, 2, 0)], arrival_rate = [(0, lam_1), (12, lam_2), (24, lam_3)], transition_prob = [(0, p_1), (12, p_2), (24, p_3)], traffic_time = [(0, tau)])


### 100 cars 3 regions
#lam_1 = [3, 20, 8]
#p_1 = [[0.6, 0.1, 0.3], [0.1, 0.6, 0.3], [0.1, 0.7, 0.2]]
#tau = [[1, 2, 3], [2, 1, 3], [3, 3, 1]]
#write_data_traffic_map("100car3region", 3, 10, payoff = {"pickup": 1, "reroute": 0, "charge": {2: -0.5}}, region_battery_car = [(0, 0, 30), (1, 0, 40), (2, 0, 30)], region_rate_plug = [(0, 2, 1)], arrival_rate = [(0, lam_1)], transition_prob = [(0, p_1)], traffic_time = [(0, tau)])

### 1000 cars 3 regions
#lam_1 = [3, 3, 30]
#lam_2 = [20, 15, 15]
#lam_3 = [5, 5, 50]
#p_1 = [[0.6, 0.1, 0.3], [0.1, 0.6, 0.3], [0.1, 0.7, 0.2]]
#tau = [[1, 2, 3], [2, 1, 3], [3, 3, 1]]
#write_data_traffic_map("1000car3region", 3, 36, payoff = {"pickup": 1, "reroute": 0, "charge": {2: 0}}, region_battery_car = [(0, 0, 30), (1, 0, 40), (2, 0, 30)], region_rate_plug = [(0, 2, 0)], arrival_rate = [(0, lam_1), (12, lam_2), (24, lam_3)], transition_prob = [(0, p_1), (12, p_2), (24, p_3)], traffic_time = [(0, tau)])

## 2400 cars 3 regions
#lam_1 = [30, 30, 300]
#lam_2 = [200, 150, 150]
#lam_3 = [50, 50, 500]
#p_1 = [[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]
#p_2 = [[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]
#p_3 = [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]
#tau = [[1, 2, 3], [2, 1, 3], [3, 3, 1]]
#write_data_traffic_map("2400car3region", 3, 36, payoff = {"pickup": 1, "reroute": 0, "charge": {2: 0}}, region_battery_car = [(0, 0, 800), (1, 0, 800), (2, 0, 800)], region_rate_plug = [(0, 2, 0)], arrival_rate = [(0, lam_1), (12, lam_2), (24, lam_3)], transition_prob = [(0, p_1), (12, p_2), (24, p_3)], traffic_time = [(0, tau)])

## 60 cars 3 regions
#lam_1 = [1.5, 1.5, 15]
#lam_2 = [10, 7.5, 7.5]
#lam_3 = [2.5, 2.5, 25]
#p_1 = [[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]
#p_2 = [[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]
#p_3 = [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]
#tau = [[1, 1, 2], [1, 1, 1], [2, 1, 1]]
#write_data_traffic_map("60car3region", 3, 20, payoff = {"pickup": 1, "reroute": 0, "charge": {2: 0}}, region_battery_car = [(0, 0, 20), (1, 0, 20), (2, 0, 20)], region_rate_plug = [(0, 2, 0)], arrival_rate = [(0, lam_1), (7, lam_2), (14, lam_3)], transition_prob = [(0, p_1), (7, p_2), (14, p_3)], traffic_time = [(0, tau)])

### 1 car 2 regions WGC
#p_1 = [[0, 1], [1, 0]]
#p_2 = [[0.5, 0.5], [1, 0]]
#lam_1 = [1, 0]
#lam_2 = [2, 1]
#tau = [[10, 10], [10, 10]]
#write_data_traffic_map("1car2region_wgc", 2, 100, payoff = {"pickup": 1, "reroute": 0, "charge": {2: 0}}, region_battery_car = [(0, 0, 1), (1, 0, 0)], region_rate_plug = [(0, 2, 0)], arrival_rate = [(0, lam_1), (9, lam_2), (10, lam_1), (19, lam_2), (20, lam_1), (29, lam_2), (30, lam_1), (39, lam_2), (40, lam_1), (49, lam_2), (50, lam_1), (59, lam_2), (60, lam_1), (69, lam_2), (70, lam_1), (79, lam_2), (80, lam_1), (89, lam_2), (90, lam_1), (99, lam_2)], transition_prob = [(0, p_1), (9, p_2), (10, p_1), (19, p_2), (20, p_1), (29, p_2), (30, p_1), (39, p_2), (40, p_1), (49, p_2), (50, p_1), (59, p_2), (60, p_1), (69, p_2), (70, p_1), (79, p_2), (80, p_1), (89, p_2), (90, p_1), (99, p_2)], traffic_time = [(0, tau)])

### 1 car 2 regions WGC 2
aggr = 10
T = 100
wgc_map = pd.read_csv("Data/Map/map_1car2region_wgc.tsv", sep = "\t")
wgc_trip_demand = pd.read_csv("Data/TripDemand/trip_demand_1car2region_wgc.tsv", sep = "\t")
wgc_trip_demand["T"] = wgc_trip_demand["T"].apply(lambda x: x // aggr)
wgc_trip_demand = wgc_trip_demand.groupby(["T", "Origin", "Destination"]).sum().reset_index()
wgc_map["T"] = wgc_map["T"].apply(lambda x: x // aggr)
wgc_map["Distance"] = wgc_map["Distance"] // aggr
wgc_map["TripTime"] = wgc_map["TripTime"] // aggr
wgc_map.to_csv(f"Data/Map/map_1car2region_wgc_{aggr}.tsv", sep = "\t", index = False)
wgc_trip_demand.to_csv(f"Data/TripDemand/trip_demand_1car2region_wgc_{aggr}.tsv", sep = "\t", index = False)

## 1 car 2 regions WGC - Simple
#p_1 = [[0, 1], [1, 0]]
#p_2 = [[1, 0], [0, 1]]
#lam_1 = [1, 0]
#tau = [[1, 1], [1, 1]]
#write_data_traffic_map("1car2region_wgc", 2, 10, payoff = {"pickup": 1, "reroute": 0, "charge": {2: 0}}, region_battery_car = [(0, 0, 1), (1, 0, 0)], region_rate_plug = [(0, 2, 0)], arrival_rate = [(0, lam_1)], transition_prob = [(0, p_1), (1, p_2), (2, p_1), (3, p_2), (4, p_1), (5, p_2), (6, p_1), (7, p_2), (8, p_1), (9, p_2)], traffic_time = [(0, tau)])

## Spatial-Temporal Star-To-Complete Network (ST-STC)
### 12 cars 4 regions
#xi = 1
#num_regions = 4
#num_cars = 12
#time_horizon = 48
#rate = 24#36
#pack_size = 24#36
##num_chargers = 0 #48
#for num_chargers in [0, 48]:
#    for xi in [0, 0.5, 1]:
#        lam, p, tau = generate_spatialtemporal_star_to_complete_network(num_regions = num_regions, xi = xi)
#        lam = lam * num_cars / np.sum(lam) * 0.5
#        name = f"st-stc_{num_cars}car{num_regions}region{num_chargers}chargers_xi={xi}"
#        write_data_traffic_map(name, num_regions, time_horizon, payoff = {"pickup": 1, "reroute": 0, "charge": {rate: 0}}, region_battery_car = [(x, pack_size - 1, num_cars // num_regions) for x in range(num_regions)], region_rate_plug = [(x, rate, num_chargers // num_regions) for x in range(num_regions)], arrival_rate = [(0, lam)], transition_prob = [(0, p)], traffic_time = [(0, tau)])

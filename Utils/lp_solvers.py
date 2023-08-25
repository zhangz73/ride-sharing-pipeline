import sys
import gc
import math
import copy
import numpy as np
import pandas as pd
import torch
import cvxpy as cvx
import gurobipy as gp
from gurobipy import GRB
import scipy
from scipy.sparse import csr_matrix, csr_array, dia_matrix, vstack
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import Utils.train as train

class LP_Solver(train.Solver):
    def __init__(self, markov_decision_process = None, num_days = 1, gamma = 1):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process)
        self.reward_df = self.markov_decision_process.reward_df
        self.num_days = num_days
        self.gamma = gamma
    
    def evaluate(self, **kargs):
        pass
    
    def train(self):
        print("Training...")
        print("\tSetting up...")
        model = gp.Model()
        m, n = self.A.shape
        x = model.addMVar(n, lb = 0, vtype = GRB.CONTINUOUS, name = "x")
        #model.addConstrs((gp.quicksum(self.A[i, r] * x[r] for r in range(n)) == self.b[i] for i in range(m)))
        model.addConstr(self.A @ x == self.b)
#        objective = gp.quicksum(self.c[r] * x[r] for r in range(n))
        objective = self.c @ x
        model.setObjective(objective, GRB.MAXIMIZE)
        print("\tOptimizing...")
        model.optimize()
        obj_val = model.ObjVal
        print(obj_val / self.total_revenue)
        print("\tGathering...")
        self.x = np.zeros(n)
        for i in tqdm(range(n), leave = False):
            self.x[i] = x[i].x
        self.describe_x()
        return self.x, obj_val
    
    def train_cvxpy(self):
        print("Training...")
        m, n = self.A.shape
#        A_cvx = cvx.Parameter(shape = (m, n), name = "A")
#        b_cvx = cvx.Parameter(shape = (m,), name = "b")
        c_cvx = cvx.Parameter(shape = (n,), name = "c")
        x = cvx.Variable(shape = (n,), nonneg = True, name = "x")
#        constraints = [A_cvx @ x == b_cvx]
        constraints = [self.A @ x == self.b]
        objective = cvx.Maximize(c_cvx @ x)
        problem = cvx.Problem(objective, constraints)
#        problem.param_dict["A"].value = self.A
#        problem.param_dict["b"].value = self.b
        problem.param_dict["c"].value = self.c
        obj_val = problem.solve()
        self.x = problem.var_dict["x"].value
        print(obj_val / self.total_revenue)
#        print(self.describe_x())
        return self.x, obj_val

class LP_On_AugmentedGraph(LP_Solver):
    def __init__(self, markov_decision_process = None, num_days = 1, gamma = 1, patience_time = 0):
        super().__init__(markov_decision_process = markov_decision_process, num_days = num_days, gamma = gamma)
        self.patience_time = patience_time
        print("Constructing the solver...")
        self.load_data()
        self.construct_problem()
    
    ## Vectors to fetch:
    ##   - Trip reward r_ij^t
    ##   - Charging cost c^t
    ##   - Trip demand \lambda_ij^t
    ##   - Charging facility per region d_i^{\delta}
    ##   - Travel time \tau_ij^t
    ##   - Battery consumption b_ij
    ##   - Initial car battery num per region
    ## Parameters to fetch:
    ##   - Total EVs N
    ##   - Time horizon T
    ##   - Number of battery levels B
    ##   - Number of regions R
    ##   - Number of charging rates \Delta
    def load_data(self):
        print("\tLoading data...")
        ## Get parameters
        self.num_total_cars = self.markov_decision_process.num_total_cars
        self.time_horizon = self.markov_decision_process.time_horizon
        self.num_regions = len(self.markov_decision_process.regions)
        self.num_battery_levels = self.markov_decision_process.num_battery_levels
        self.charging_rates = self.markov_decision_process.charging_rates
        self.num_charging_rates = len(self.charging_rates)
        self.battery_per_step = self.markov_decision_process.battery_per_step
        ## Get vectors
        ### trip rewards T x R^2
        self.trip_rewards = self.markov_decision_process.reward_query.get_wide_reward(self.time_horizon, self.num_regions)
        ### charging cost T x \Delta
        #### Note that charging costs are negative by default!
        self.charging_costs = np.zeros((self.time_horizon, self.num_charging_rates))
        reward_df = self.markov_decision_process.reward_df
        charging_cost_df = reward_df[reward_df["Type"] == "Charge"]
        charging_cost_df = charging_cost_df[["T", "Rate", "Payoff"]].groupby(["T", "Rate"]).mean().reset_index()
        for t in range(self.time_horizon):
            for rate_idx in range(self.num_charging_rates):
                rate = self.charging_rates[rate_idx]
                tmp_df = charging_cost_df[(charging_cost_df["T"] == t) & (charging_cost_df["Rate"] == rate)]
                if tmp_df.shape[0] > 0:
                    cost = tmp_df.iloc[0]["Payoff"]
                else:
                    cost = 0
                self.charging_costs[t, rate_idx] = cost
        ### Trip demand T x R^2
        self.trip_demands = self.markov_decision_process.trip_demands.get_arrival_rates()
        self.total_revenue = np.sum(self.trip_demands * self.trip_rewards)
        ### Charging facility R x \Delta
        self.charging_facility_num = np.zeros((self.num_regions, self.num_charging_rates))
        region_rate_plug_df = self.markov_decision_process.region_rate_plug_df
        for region in range(self.num_regions):
            for rate_idx in range(self.num_charging_rates):
                rate = self.charging_rates[rate_idx]
                tmp_df = region_rate_plug_df[(region_rate_plug_df["region"] == region) & (region_rate_plug_df["rate"] == rate)]
                if tmp_df.shape[0] > 0:
                    cnt = tmp_df.iloc[0]["num"]
                else:
                    cnt = 0
                self.charging_facility_num[region, rate_idx] = cnt
        ### Travel time T x R^2
        self.travel_time = np.zeros((self.time_horizon, self.num_regions ** 2))
        map = self.markov_decision_process.map
        for t in range(self.time_horizon):
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    trip_time = map.time_to_location(origin, dest, t)
                    self.travel_time[t, origin * self.num_regions + dest] = trip_time
        ### Battery consumption T x R^2
        self.battery_consumption = np.zeros(self.num_regions ** 2)
        for origin in range(self.num_regions):
            for dest in range(self.num_regions):
                distance = map.distance(origin, dest)
                self.battery_consumption[origin * self.num_regions + dest] = distance * self.battery_per_step
        ### Initial battery car num per region R x B
        self.init_car_num = np.zeros(self.num_regions * self.num_battery_levels)
        region_battery_car_df = self.markov_decision_process.region_battery_car_df
        for i in range(region_battery_car_df.shape[0]):
            region = region_battery_car_df.iloc[i]["region"]
            battery = region_battery_car_df.iloc[i]["battery"]
            num = region_battery_car_df.iloc[i]["num"]
            self.init_car_num[region * self.num_battery_levels + battery] = num
    
    def construct_problem(self):
        self.construct_x()
        self.construct_obj()
        self.construct_constraints()
    
    ## Variables:
    ##   - Passenger-carry flow T x B x R^2
    ##   - Rerouting flow T x B x R^2
    ##   - Charging flow T x \Delta x B x R
    def construct_x(self):
        travel_flow_len = self.time_horizon * self.num_battery_levels * self.num_regions * self.num_regions
        charging_flow_len = self.time_horizon * self.num_battery_levels * self.num_regions * self.num_charging_rates
        self.rerouting_flow_begin = travel_flow_len
        self.charging_flow_begin = travel_flow_len * 2
        trip_demand_extra_len = self.time_horizon * self.num_regions * self.num_regions
        charging_facility_extra_len = self.time_horizon * self.num_regions * self.num_charging_rates
        self.trip_demand_extra_begin = self.charging_flow_begin + charging_flow_len
        self.charging_facility_extra_begin = self.trip_demand_extra_begin + trip_demand_extra_len
        ## Add dummy variables for number of requests at time t but taken at time s
        slack_request_patience_len = self.time_horizon * (self.patience_time + 1) * self.num_regions * self.num_regions
        self.slack_request_patience_begin = self.charging_facility_extra_begin + charging_facility_extra_len
        self.x_len = travel_flow_len * 2 + charging_flow_len + trip_demand_extra_len + charging_facility_extra_len + slack_request_patience_len
        self.x = np.zeros(self.x_len)
    
    def get_x_entry(self, entry_type, t, b, origin = None, dest = None, region = None, rate_idx = None):
        assert entry_type in ["passenger-carry", "reroute", "charge"]
        if entry_type == "charge":
            assert region is not None and rate_idx is not None
            ans = t * self.num_battery_levels * self.num_regions * self.num_charging_rates + b * self.num_regions * self.num_charging_rates + region * self.num_charging_rates + rate_idx
            return self.charging_flow_begin + ans
        ans = ans = t * self.num_battery_levels * self.num_regions * self.num_regions + b * self.num_regions * self.num_regions + origin * self.num_regions + dest
        if entry_type == "reroute":
            ans += self.rerouting_flow_begin
        return int(ans)
    
    def reset_x_infer(self, strict = True):
        if strict:
            ## Travel: t, b, origin, dest
            ## Charge: t, \delta, b, region
            x_travel = self.x[:self.rerouting_flow_begin] + self.x[self.rerouting_flow_begin:self.charging_flow_begin]
            x_charge = self.x[self.charging_flow_begin:self.trip_demand_extra_begin].copy()
        else:
            ## Travel: t, origin, dest
            ## Charge: t, \delta, region
            travel_len = self.num_battery_levels * self.num_regions * self.num_regions
            charge_len = self.num_battery_levels * self.num_regions
            x_travel = np.array([[self.x[:self.charging_flow_begin][(t * travel_len):((t + 1) * travel_len)][i::(self.num_regions ** 2)] for i in range(self.num_regions ** 2)] for t in range(self.time_horizon * 2)]).sum(axis = 2).flatten()
            x_charge = np.array([[self.x[self.charging_flow_begin:self.trip_demand_extra_begin][(t * charge_len):((t + 1) * charge_len)][i::(self.num_regions)] for i in range(self.num_regions)] for t in range(self.time_horizon * self.num_charging_rates)]).sum(axis = 2).flatten()
        self.x_travel = x_travel.round().astype(int)
        self.x_charge = x_charge.round().astype(int)
    
    def describe_x(self, fname = "lp_debug.txt"):
        eps = 0.1
        with open(fname, "w") as f:
            for entry_type in ["passenger-carry", "reroute"]:
                for t in range(self.time_horizon):
                    for b in range(self.num_battery_levels):
                        for origin in range(self.num_regions):
                            for dest in range(self.num_regions):
                                x_entry = self.get_x_entry(entry_type, t, b, origin = origin, dest = dest)
                                if self.x[x_entry] >= eps and (entry_type == "passenger-carry" or origin != dest):
                                    val = self.x[x_entry]
                                    f.write(f"{entry_type}, t = {t}, b = {b}, origin = {origin}, dest = {dest}, val = {val}\n")
            for t in range(self.time_horizon):
                for b in range(self.num_battery_levels):
                    for region in range(self.num_regions):
                        for rate_idx in range(self.num_charging_rates):
                                x_entry = self.get_x_entry("charge", t, b, region = region, rate_idx = rate_idx)
                                if self.x[x_entry] >= eps:
                                    val = self.x[x_entry]
                                    f.write(f"charge, t = {t}, b = {b}, region = {region}, rate = {self.charging_rates[rate_idx]}, val = {val}\n")
    
    def get_flow_conserv_entry(self, t, b, region):
        return int(t * self.num_battery_levels * self.num_regions + b * self.num_regions + region)
    
    def construct_obj(self):
        self.c = np.zeros(self.x_len)
        for t in range(self.time_horizon):
            for b in range(self.num_battery_levels):
                begin = t * self.num_battery_levels * self.num_regions ** 2 + b * self.num_regions ** 2
                self.c[begin:(begin + self.num_regions ** 2)] = self.trip_rewards[t,:] * self.gamma ** t
        for t in range(self.time_horizon):
            for rate_idx in range(self.num_charging_rates):
                cost = self.charging_costs[t, rate_idx]
                begin = self.charging_flow_begin + t * self.num_charging_rates * self.num_battery_levels * self.num_regions + rate_idx * self.num_battery_levels * self.num_regions
                end = begin + self.num_battery_levels * self.num_regions
                self.c[begin:end] = cost * self.gamma ** t
    
    ## Flows add up to initial car distribution
    ## Flow conservation at each time, battery, and region
    ## Passenger-carrying flows not exceed trip demands (Deprecated!!!)
    ## Passenger-carrying flows equals slack request patience flow:
    ##    f_{ij}^t = s_{ij}^{t, t} + s_{ij}^{t, t - 1} + s_{ij}^{t, t - 2}
    ## Slack request patience flow not exceed trip demands:
    ##    s_{ij}^{t, t} + s_{ij}^{t + 1, t} + s_{ij}^{t + 2, t} <= \lambda_{ij}^t
    ## Charging flows not exceed charging facility nums
    ## Infeasible flows equal to 0 (i.e. trips with insufficient battery)
    ## All flows add up to total cars at each time
    ## All flows being non-negative
    def construct_constraints(self):
        print("\tConstructing constraints...")
        ### Flow conservation
        print("\t\tConstructing flow conservation matrix...")
        flow_conservation_mat, flow_conservation_target = self.construct_flow_conservation_matrix()
        ### Trip demand
        print("\t\tConstructing trip demand matrix...")
#        trip_demand_mat = np.zeros((self.time_horizon * self.num_regions * self.num_regions, self.x_len))
        trip_demand_target = np.zeros(self.time_horizon * self.num_regions * self.num_regions)
        slack_request_patience_target = np.zeros(self.time_horizon * self.num_regions * self.num_regions)
        slack_request_patience_lst = []
        trip_demand_lst = []
        for t in tqdm(range(self.time_horizon), leave = False):
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    slack_request_patience_vec = np.zeros(self.x_len)
                    trip_demand_vec = np.zeros(self.x_len)
                    begin = t * self.num_battery_levels * self.num_regions * self.num_regions + origin * self.num_regions + dest
                    end = begin + self.num_battery_levels * self.num_regions * self.num_regions
                    pos = t * self.num_regions * self.num_regions + origin * self.num_regions + dest
                    trip_demand_vec[begin:end:(self.num_regions ** 2)] = 1
                    slack_begin = self.slack_request_patience_begin + t * self.num_regions * self.num_regions * (self.patience_time + 1) + origin * self.num_regions * (self.patience_time + 1) + dest * (self.patience_time + 1)
                    slack_end = slack_begin + self.patience_time + 1
                    slack_end_2 = slack_begin + (self.patience_time + 1) * self.num_regions * self.num_regions * (self.patience_time + 1)
                    trip_demand_vec[slack_begin:slack_end] = -1
                    
                    slack_request_patience_vec[self.trip_demand_extra_begin + pos] = 1
                    slack_request_patience_vec[slack_begin:slack_end_2:(self.num_regions * self.num_regions * (self.patience_time + 1) + 1)] = 1
                    slack_request_patience_vec = csr_matrix(slack_request_patience_vec)
                    trip_demand_vec = csr_matrix(trip_demand_vec)
                    trip_demand_lst.append(trip_demand_vec)
                    slack_request_patience_lst.append(slack_request_patience_vec)
#                    trip_demand_mat[pos, begin:end:(self.num_regions ** 2)] = 1
#                    trip_demand_mat[pos, self.trip_demand_extra_begin + pos] = 1
                    slack_request_patience_target[pos] = self.trip_demands[t, origin * self.num_regions + dest]
        trip_demand_mat = vstack(trip_demand_lst)
        slack_request_patience_mat = vstack(slack_request_patience_lst)
        ### Charging facility
        print("\t\tConstructing charging facility matrix...")
#        charging_facility_mat = np.zeros((self.time_horizon * self.num_regions * self.num_charging_rates, self.x_len))
        charging_facility_target = np.zeros(self.time_horizon * self.num_regions * self.num_charging_rates)
        charging_facility_lst = []
        for t in tqdm(range(self.time_horizon), leave = False):
            for region in range(self.num_regions):
                for rate_idx in range(self.num_charging_rates):
                    charging_facility_vec = np.zeros(self.x_len)
                    begin = self.charging_flow_begin + t * self.num_charging_rates * self.num_battery_levels * self.num_regions + rate_idx * self.num_battery_levels * self.num_regions + region
                    end = begin + self.num_battery_levels * self.num_regions
                    pos = t * self.num_regions * self.num_charging_rates + region * self.num_charging_rates + rate_idx
                    charging_facility_vec[self.charging_facility_extra_begin + pos] = 1
                    charging_facility_vec[begin:end:(self.num_regions)] = 1
                    charging_facility_vec = csr_matrix(charging_facility_vec)
                    charging_facility_lst.append(charging_facility_vec)
#                    charging_facility_mat[pos, self.charging_facility_extra_begin + pos] = 1
#                    charging_facility_mat[pos, begin:end:(self.num_regions)] = 1
                    charging_facility_target[pos] = self.charging_facility_num[region, rate_idx]
        charging_facility_mat = vstack(charging_facility_lst)
        ### Total flows
        print("\t\tConstructing total flow matrix...")
        total_flow_mat = np.zeros((self.time_horizon, self.x_len))
        for t in tqdm(range(self.time_horizon), leave = False):
            passenger_len = self.num_battery_levels * self.num_regions * self.num_regions
            charge_len = self.num_charging_rates * self.num_battery_levels * self.num_regions
            total_flow_mat[t, (t * passenger_len):((t + 1) * passenger_len)] = 1
            total_flow_mat[t, (self.rerouting_flow_begin + t * passenger_len):(self.rerouting_flow_begin + (t + 1) * passenger_len)] = 1
            total_flow_mat[t, (self.charging_flow_begin + t * charge_len):(self.charging_flow_begin + (t + 1) * charge_len)] = 1
        total_flow_mat = csr_matrix(total_flow_mat)
            
        total_flow_target = np.ones(self.time_horizon) * self.num_total_cars
        ### Concatenate together
        self.A = vstack((flow_conservation_mat, trip_demand_mat, slack_request_patience_mat, charging_facility_mat, total_flow_mat))
        self.b = np.concatenate((flow_conservation_target, trip_demand_target, slack_request_patience_target, charging_facility_target, total_flow_target), axis = None)
#        self.b = csr_matrix(self.b.reshape((len(self.b), 1)))
    
    ## Flow conservation at each (time, battery, region)
    ##   - Passenger-carrying to & from each region
    ##   - Rerouting to & from each region
    ##   - Charging at each rate
    def construct_flow_conservation_matrix(self):
        flow_conservation_target = np.zeros(self.time_horizon * self.num_battery_levels * self.num_regions)
        row_lst = []
        col_lst = []
        val_lst = []
        ## Populate initial car flow
        for region in tqdm(range(self.num_regions), leave = False):
            for battery in range(self.num_battery_levels):
                num = self.init_car_num[region * self.num_battery_levels + battery]
                flow_conservation_target[battery * self.num_regions + region] = num
        ## Populate flow conservation matrix
        for t in tqdm(range(self.time_horizon), leave = False):
            for b in tqdm(range(self.num_battery_levels), leave = False):
                ## Populate traveling flows
                for origin in range(self.num_regions):
                    for dest in range(self.num_regions):
                        trip_time = max(self.travel_time[t, origin * self.num_regions + dest], 1)
                        battery_cost = self.battery_consumption[origin * self.num_regions + dest]
                        passenger_pos = self.get_x_entry("passenger-carry", t, b, origin = origin, dest = dest)
                        reroute_pos = self.get_x_entry("reroute", t, b, origin = origin, dest = dest)
                        start_row = self.get_flow_conserv_entry(t, b, origin)
                        row_lst += [start_row, start_row]
                        col_lst += [passenger_pos, reroute_pos]
                        val_lst += [1, 1]
                        if b >= battery_cost:
                            end_time = t + trip_time
                            if end_time >= self.time_horizon and self.num_days > 1:
                                end_row = self.get_flow_conserv_entry(end_time - self.time_horizon, b - battery_cost, dest)
                            elif end_time < self.time_horizon:
                                end_row = self.get_flow_conserv_entry(end_time, b - battery_cost, dest)
                            else:
                                end_row = None
                            if origin != dest:
                                if end_row is not None:
                                    row_lst += [end_row, end_row]
                                    col_lst += [passenger_pos, reroute_pos]
                                    val_lst += [-1, -1]
                            else:
                                end_time_reroute = t + 1
                                if end_time_reroute >= self.time_horizon and self.num_days > 1:
                                    end_row_reroute = self.get_flow_conserv_entry(end_time_reroute - self.time_horizon, b, dest)
                                elif end_time_reroute < self.time_horizon:
                                    end_row_reroute = self.get_flow_conserv_entry(end_time_reroute, b, dest)
                                else:
                                    end_time_reroute = None
                                if end_row is not None:
                                    row_lst += [end_row, end_row_reroute]
                                    col_lst += [passenger_pos, reroute_pos]
                                    val_lst += [-1, -1]
                ## Populate charging flows
                for region in range(self.num_regions):
                    for rate_idx in range(self.num_charging_rates):
                        start_charge_pos = self.get_x_entry("charge", t, b, region = region, rate_idx = rate_idx)
                        rate = self.charging_rates[rate_idx]
                        end_time = t + 1
                        end_battery = min(b + rate, self.num_battery_levels - 1)
                        charge_pos = self.get_x_entry("charge", t, b, region = region, rate_idx = rate_idx)
                        start_row = self.get_flow_conserv_entry(t, b, region)
                        row_lst += [start_row]
                        col_lst += [charge_pos]
                        val_lst += [1]
                        if end_time >= self.time_horizon and self.num_days > 1:
                            end_row = self.get_flow_conserv_entry(end_time - self.time_horizon, end_battery, region)
                        elif end_time < self.time_horizon:
                            end_row = self.get_flow_conserv_entry(end_time, end_battery, region)
                        else:
                            end_row = None
                        if end_row is not None:
                            row_lst += [end_row]
                            col_lst += [charge_pos]
                            val_lst += [-1]
        ## Construct flow_conservation_mat row by row
        flow_conservation_mat = csr_matrix((val_lst, (row_lst, col_lst)), shape = (self.time_horizon * self.num_battery_levels * self.num_regions, self.x_len))
        return flow_conservation_mat, flow_conservation_target
    
    ## Deprecated!!
    def get_relevant_x_prev(self, t, region, battery):
        passenger_carry_idx_begin = t * self.num_battery_levels * self.num_regions * self.num_regions + battery * self.num_regions * self.num_regions + region * self.num_regions
        passenger_carry_idx_end = passenger_carry_idx_begin + self.num_regions
        reroute_idx_begin = passenger_carry_idx_begin + self.rerouting_flow_begin
        reroute_idx_end = reroute_idx_begin + self.num_regions
        charge_idx_begin = self.charging_flow_begin + t * self.num_charging_rates * self.num_battery_levels * self.num_regions + battery * self.num_regions + region
        charge_idx_end = charge_idx_begin + self.num_charging_rates * self.num_battery_levels * self.num_regions
        travel_x_ids = list(range(passenger_carry_idx_begin, passenger_carry_idx_end)) + list(range(reroute_idx_begin, reroute_idx_end))
        charge_x_ids = list(range(charge_idx_begin, charge_idx_end, self.num_battery_levels * self.num_regions))
        return travel_x_ids, charge_x_ids
    
    def get_relevant_x(self, t, region, battery, strict = True):
        if strict:
            travel_idx_begin = t * self.num_battery_levels * self.num_regions * self.num_regions + battery * self.num_regions * self.num_regions + region * self.num_regions
            travel_idx_end = travel_idx_begin + self.num_regions
            charge_idx_begin = t * self.num_charging_rates * self.num_battery_levels * self.num_regions + battery * self.num_regions + region
            charge_idx_end = charge_idx_begin + self.num_charging_rates * self.num_battery_levels * self.num_regions
            travel_x_ids = list(range(travel_idx_begin, travel_idx_end))
            charge_x_ids = list(range(charge_idx_begin, charge_idx_end, self.num_battery_levels * self.num_regions))
        else:
            ## Travel: t, origin, dest
            ## Charge: t, \delta, region
            travel_idx_begin = t * self.num_regions * self.num_regions + region * self.num_regions
            travel_idx_end = travel_idx_begin + self.num_regions
            charge_idx_begin = t * self.num_charging_rates * self.num_regions + region
            charge_idx_end = charge_idx_begin + self.num_charging_rates * self.num_regions
            travel_x_ids = list(range(travel_idx_begin, travel_idx_end))
            charge_x_ids = list(range(charge_idx_begin, charge_idx_end, self.num_regions))
        return travel_x_ids, charge_x_ids
    
    def evaluate(self, return_action = True, seed = None, day_num = 0, strict = True):
        if seed is not None:
            torch.manual_seed(seed)
        self.markov_decision_process.reset_states(new_episode = day_num == 0)
        self.trip_demands = self.markov_decision_process.trip_arrivals
        self.train()
        init_payoff = float(self.markov_decision_process.get_payoff_curr_ts(deliver = True))
        action_lst_ret = []
        payoff_lst = []
        discount_lst = []
        atomic_payoff_lst = []
        self.reset_x_infer(strict = strict)
        for t in range(self.time_horizon):
            available_car_ids = self.markov_decision_process.get_available_car_ids(True)
            num_available_cars = len(available_car_ids)
            for car_idx in range(num_available_cars):
                car_id = available_car_ids[car_idx]
                dest, eta, battery = self.markov_decision_process.get_car_info(car_id)
                action_assigned = False
                if eta == 0:
                    travel_x_ids, charge_x_ids = self.get_relevant_x(t, dest, battery, strict = strict)
                    for i in range(len(charge_x_ids)):
                        x_id = charge_x_ids[i]
                        if self.x_charge[x_id] > 0:
                            action_assigned = True
                            self.x_charge[x_id] -= 1
                            action_id = self.markov_decision_process.query_action(("charge", dest, self.charging_rates[i]))
                            break
                    if not action_assigned:
                        for i in range(len(travel_x_ids)):
                            x_id = travel_x_ids[i]
                            if self.x_travel[x_id] > 0:
                                action_assigned = True
                                self.x_travel[x_id] -= 1
                                action_id = self.markov_decision_process.query_action(("travel", dest, i % self.num_regions))
                                break
                if not action_assigned:
                    action_id = self.markov_decision_process.query_action(("nothing"))
                action = self.all_actions[action_id]
                self.markov_decision_process.transit_within_timestamp(action, car_id = car_id)
                curr_state_counts_full = self.markov_decision_process.get_state_counts(deliver = True)
                action_lst_ret.append((curr_state_counts_full, action, t, car_id))
            curr_state_counts_full = self.markov_decision_process.get_state_counts(deliver = True)
            action_lst_ret.append((curr_state_counts_full, None, t, None))
            self.markov_decision_process.transit_across_timestamp()
            curr_payoff = self.markov_decision_process.get_payoff_curr_ts(deliver = True)
            payoff_lst.append(curr_payoff)
            discount_lst.append(self.gamma ** t)
        discount_lst = torch.tensor(discount_lst)
        atomic_payoff_lst = torch.tensor([init_payoff] + payoff_lst)
        atomic_payoff_lst = atomic_payoff_lst[1:] - atomic_payoff_lst[:-1]
        discounted_payoff = torch.sum(atomic_payoff_lst * discount_lst)
        return None, None, payoff_lst, action_lst_ret, discounted_payoff

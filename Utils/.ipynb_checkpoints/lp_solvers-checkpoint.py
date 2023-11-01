import sys
import gc
import math
import copy
import numpy as np
import pandas as pd
import torch
#import cvxpy as cvx
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
    def __init__(self, markov_decision_process = None, num_days = 1, gamma = 1, retrain = True, n_cpu = 1):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process)
        self.reward_df = self.markov_decision_process.reward_df
        self.num_days = num_days
        self.gamma = gamma
        self.retrain = retrain
        self.n_cpu = n_cpu
    
    def evaluate(self, **kargs):
        pass
    
    def train(self):
        if self.retrain:
            print("Training...")
            print("\tSetting up...")
            model = gp.Model()
            m, n = self.A.shape
    #        model.setParam("MIPFocus", 3)
            model.setParam("Method", 2)
            model.setParam("Crossover", 0)
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
            np.save("lp_x.npy", self.x)
        else:
            self.x = np.load("lp_x.npy")
            obj_val = -1
        return obj_val / self.total_revenue #self.x, obj_val
    
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
    def __init__(self, markov_decision_process = None, num_days = 1, gamma = 1, patience_time = 0, retrain = True, n_cpu = 1, **kargs):
        super().__init__(markov_decision_process = markov_decision_process, num_days = num_days, gamma = gamma, retrain = retrain, n_cpu = n_cpu)
        self.patience_time = markov_decision_process.connection_patience + markov_decision_process.pickup_patience
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
#        self.trip_rewards = np.ones(self.trip_rewards.shape)
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
                self.charging_costs[t, rate_idx] = cost * self.markov_decision_process.charging_cost_inflation
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
    
    def reset_timestamp(self, fractional_cars = True, full_knowledge = False):
        self.trip_demands = self.markov_decision_process.trip_arrivals.numpy()
        self.total_revenue = self.markov_decision_process.get_total_market_revenue() #np.sum(self.trip_demands * self.trip_rewards) #
        if fractional_cars or full_knowledge:
            self.construct_problem()
    
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
        slack_request_patience_len = self.time_horizon * self.num_regions * self.num_regions * (self.patience_time + 1)
        self.slack_request_patience_begin = self.charging_facility_extra_begin + charging_facility_extra_len
        self.x_len = travel_flow_len * 2 + charging_flow_len + trip_demand_extra_len + charging_facility_extra_len + slack_request_patience_len
#        self.x = np.zeros(self.x_len)
    
    def get_x_entry(self, entry_type, t, b, origin = None, dest = None, region = None, rate_idx = None):
        assert entry_type in ["passenger-carry", "reroute", "charge"]
        if entry_type == "charge":
            assert region is not None and rate_idx is not None
            ans = t * self.num_battery_levels * self.num_regions * self.num_charging_rates + b * self.num_regions * self.num_charging_rates + region * self.num_charging_rates + rate_idx
            return self.charging_flow_begin + ans
        ans = t * self.num_battery_levels * self.num_regions * self.num_regions + b * self.num_regions * self.num_regions + origin * self.num_regions + dest
        if entry_type == "reroute":
            ans += self.rerouting_flow_begin
        return int(ans)
    
    def reset_x_infer(self, strict = True, x_copy = None):
        if x_copy is None:
            x_copy = self.x
        if strict:
            ## Travel: t, b, origin, dest
            ## Charge: t, \delta, b, region
            x_travel = x_copy[:self.rerouting_flow_begin] + x_copy[self.rerouting_flow_begin:self.charging_flow_begin]
            x_charge = x_copy[self.charging_flow_begin:self.trip_demand_extra_begin].copy()
        else:
            ## Travel: t, origin, dest
            ## Charge: t, \delta, region
            travel_len = self.num_battery_levels * self.num_regions * self.num_regions
            charge_len = self.num_battery_levels * self.num_regions
            x_travel = np.array([[x_copy[:self.charging_flow_begin][(t * travel_len):((t + 1) * travel_len)][i::(self.num_regions ** 2)] for i in range(self.num_regions ** 2)] for t in range(self.time_horizon * 2)]).sum(axis = 2).flatten()
            x_charge = np.array([[x_copy[self.charging_flow_begin:self.trip_demand_extra_begin][(t * charge_len):((t + 1) * charge_len)][i::(self.num_regions)] for i in range(self.num_regions)] for t in range(self.time_horizon * self.num_charging_rates)]).sum(axis = 2).flatten()
        self.x_travel = x_travel #x_travel.round().astype(int)
        self.x_charge = x_charge #x_charge.round().astype(int)
    
    def describe_x(self, fname = "lp_debug.txt"):
        eps = 0.1
        with open(fname, "w") as f:
            for entry_type in ["passenger-carry", "reroute"]:
                for t in range(self.time_horizon):
                    for b in range(self.num_battery_levels):
                        for origin in range(self.num_regions):
                            for dest in range(self.num_regions):
                                x_entry = self.get_x_entry(entry_type, t, b, origin = origin, dest = dest)
                                #if self.x[x_entry] >= eps and (entry_type == "passenger-carry" or origin != dest):
                                if self.x[x_entry] >= eps:
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
                for b in range(self.num_battery_levels):
                    unit_cost = self.charging_costs[t, rate_idx]
                    rate = self.charging_rates[rate_idx]
                    next_battery = self.markov_decision_process.get_next_battery(rate, b)
                    cost = (next_battery - b) / rate * unit_cost
                    begin = self.charging_flow_begin + t * self.num_charging_rates * self.num_battery_levels * self.num_regions + rate_idx * self.num_battery_levels * self.num_regions + b * self.num_regions
                    end = begin + self.num_regions
                    self.c[begin:end] = cost * self.gamma ** t

    def construct_trip_demand_matrix_single(self, t_lo, t_hi):
        trip_demand_target = np.zeros(self.time_horizon * self.num_regions * self.num_regions)
        slack_request_patience_target = np.zeros(self.time_horizon * self.num_regions * self.num_regions)
        slack_request_patience_lst = []
        trip_demand_lst = []
        for t in tqdm(range(t_lo, t_hi), leave = False):
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    slack_request_patience_vec = np.zeros(self.x_len)
                    trip_demand_vec = np.zeros(self.x_len)
                    begin = t * self.num_battery_levels * self.num_regions * self.num_regions + origin * self.num_regions + dest
                    end = begin + self.num_battery_levels * self.num_regions * self.num_regions
                    pos = t * self.num_regions * self.num_regions + origin * self.num_regions + dest
                    trip_demand_vec[begin:end:(self.num_regions ** 2)] = 1
                    slack_begin = self.slack_request_patience_begin + t * self.num_regions * self.num_regions * (self.patience_time + 1) + origin * self.num_regions * (self.patience_time + 1) + dest * (self.patience_time + 1)
                    slack_terminal = self.slack_request_patience_begin + (self.time_horizon) * self.num_regions * self.num_regions * (self.patience_time + 1)
                    if self.num_days == 1:
                        end_1_t = min(self.patience_time, t) + 1
                        end_2_t = min(self.patience_time + 1, self.time_horizon - t)
                        slack_end = slack_begin + end_1_t
                        slack_end_2 = slack_begin + end_2_t * self.num_regions * self.num_regions * (self.patience_time + 1)
                        slack_lst_1 = np.arange(slack_begin, slack_end)
                        slack_lst_2 = np.arange(slack_begin, slack_end_2, (self.num_regions * self.num_regions * (self.patience_time + 1) + 1))
                    else:
                        end_1_t = self.patience_time + 1 #min(self.patience_time, t) + 1
                        end_2_t = self.patience_time + 1
                        slack_end = slack_begin + end_1_t
                        slack_end_2 = slack_begin + end_2_t * self.num_regions * self.num_regions * (self.patience_time + 1)
                        slack_lst_1 = np.arange(slack_begin, slack_end)
                        slack_lst_2 = np.arange(slack_begin, slack_end_2, (self.num_regions * self.num_regions * (self.patience_time + 1) + 1))
                        for i in range(len(slack_lst_2)):
                            if slack_lst_2[i] > slack_terminal:
                                slack_lst_2[i] -= self.time_horizon * self.num_regions * self.num_regions * (self.patience_time + 1)
                    trip_demand_vec[slack_lst_1] = -1
                    
                    slack_request_patience_vec[self.trip_demand_extra_begin + pos] = 1
                    slack_request_patience_vec[slack_lst_2] = 1
                    slack_request_patience_vec = csr_matrix(slack_request_patience_vec)
                    trip_demand_vec = csr_matrix(trip_demand_vec)
                    trip_demand_lst.append(trip_demand_vec)
                    slack_request_patience_lst.append(slack_request_patience_vec)
#                    trip_demand_mat[pos, begin:end:(self.num_regions ** 2)] = 1
#                    trip_demand_mat[pos, self.trip_demand_extra_begin + pos] = 1
                    slack_request_patience_target[pos] = self.trip_demands[t, origin * self.num_regions + dest]
        return trip_demand_target, trip_demand_lst, slack_request_patience_target, slack_request_patience_lst
    
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
        ### Flow conservation and Battery Feasibility
        print("\t\tConstructing flow conservation matrix...")
        flow_conservation_mat, flow_conservation_target, battery_feasible_mat, battery_feasible_target = self.construct_flow_conservation_battery_feasibility_matrix()
        ### Trip demand
        print("\t\tConstructing trip demand matrix...")
#        trip_demand_mat = np.zeros((self.time_horizon * self.num_regions * self.num_regions, self.x_len))
        batch_size = int(math.ceil(self.time_horizon / self.n_cpu))
        results = Parallel(n_jobs = self.n_cpu)(delayed(self.construct_trip_demand_matrix_single)(
            i * batch_size, min((i + 1) * batch_size, self.time_horizon)
        ) for i in range(self.n_cpu))
        trip_demand_target = np.zeros(self.time_horizon * self.num_regions * self.num_regions)
        slack_request_patience_target = np.zeros(self.time_horizon * self.num_regions * self.num_regions)
        slack_request_patience_lst = []
        trip_demand_lst = []
        for res in results:
            trip_demand_target_single, trip_demand_lst_single, slack_request_patience_target_single, slack_request_patience_lst_single = res
            trip_demand_target += trip_demand_target_single
            trip_demand_lst += trip_demand_lst_single
            slack_request_patience_target += slack_request_patience_target_single
            slack_request_patience_lst += slack_request_patience_lst_single
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
#            total_flow_mat[t, (t * passenger_len):((t + 1) * passenger_len)] = 1
#            total_flow_mat[t, (self.rerouting_flow_begin + t * passenger_len):(self.rerouting_flow_begin + (t + 1) * passenger_len)] = 1
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    trip_time = int(max(self.travel_time[t, origin * self.num_regions + dest], 1))
                    trip_time2 = 1
                    if self.num_days == 1:
                        t_lst = np.arange(t, min(t + trip_time, self.time_horizon))
                        t_lst2 = np.arange(t, min(t + trip_time2, self.time_horizon))
                    else:
                        t_lst = [x if x < self.time_horizon else x - self.time_horizon for x in np.arange(t, t + trip_time)]
                        t_lst2 = [x if x < self.time_horizon else x - self.time_horizon for x in np.arange(t, t + trip_time2)]
                    for b in range(self.num_battery_levels):
                        passenger_begin = t * self.num_battery_levels * self.num_regions * self.num_regions + b * self.num_regions * self.num_regions + origin * self.num_regions + dest
                        reroute_begin = self.rerouting_flow_begin + passenger_begin
                        total_flow_mat[t_lst, passenger_begin] = 1
                        if origin != dest:
                            total_flow_mat[t_lst, reroute_begin] = 1
                        else:
                            total_flow_mat[t_lst2, reroute_begin] = 1
            total_flow_mat[t, (self.charging_flow_begin + t * charge_len):(self.charging_flow_begin + (t + 1) * charge_len)] = 1
        total_flow_mat = csr_matrix(total_flow_mat)
            
        total_flow_target = np.ones(self.time_horizon) * self.num_total_cars
        ### Concatenate together
        self.A = vstack((flow_conservation_mat, trip_demand_mat, slack_request_patience_mat, charging_facility_mat, total_flow_mat, battery_feasible_mat))
        self.b = np.concatenate((flow_conservation_target, trip_demand_target, slack_request_patience_target, charging_facility_target, total_flow_target, battery_feasible_target), axis = None)
#        self.b = csr_matrix(self.b.reshape((len(self.b), 1)))
    
    ## Flow conservation at each (time, battery, region)
    ##   - Passenger-carrying to & from each region
    ##   - Rerouting to & from each region
    ##   - Charging at each rate
    ##
    def construct_flow_conservation_battery_feasibility_matrix(self):
        flow_conservation_target = np.zeros(self.time_horizon * self.num_battery_levels * self.num_regions)
        row_lst = []
        col_lst = []
        val_lst = []
        battery_feasible_col_lst = []
        ## Populate initial car flow
        if self.num_days == 1:
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
                        start_row_filled = False
                        if b >= battery_cost:
                            row_lst += [start_row, start_row]
                            col_lst += [passenger_pos, reroute_pos]
                            val_lst += [1, 1]
                            start_row_filled = True
                            end_time = t + trip_time
                            if end_time >= self.time_horizon and self.num_days > 1:
                                end_row = self.get_flow_conserv_entry(end_time - self.time_horizon, b - battery_cost, dest)
                            elif end_time < self.time_horizon:
                                end_row = self.get_flow_conserv_entry(end_time, b - battery_cost, dest)
                            else:
                                end_row = None
                            if end_row is not None:
                                if origin != dest:
                                    row_lst += [end_row, end_row]
                                    col_lst += [passenger_pos, reroute_pos]
                                    val_lst += [-1, -1]
                                else:
                                    row_lst += [end_row]
                                    col_lst += [passenger_pos]
                                    val_lst += [-1]
                        else:
                            battery_feasible_col_lst.append(passenger_pos)
                            if origin != dest:
                                battery_feasible_col_lst.append(reroute_pos)
                        if origin == dest:
                            if not start_row_filled:
                                row_lst += [start_row]
                                col_lst += [reroute_pos]
                                val_lst += [1]
                            end_time_reroute = t + 1
                            if end_time_reroute >= self.time_horizon and self.num_days > 1:
                                end_row_reroute = self.get_flow_conserv_entry(end_time_reroute - self.time_horizon, b, dest)
                            elif end_time_reroute < self.time_horizon:
                                end_row_reroute = self.get_flow_conserv_entry(end_time_reroute, b, dest)
                            else:
                                end_row_reroute = None
                            if end_row_reroute is not None:
                                row_lst += [end_row_reroute]
                                col_lst += [reroute_pos]
                                val_lst += [-1]
                ## Populate charging flows
                for region in range(self.num_regions):
                    for rate_idx in range(self.num_charging_rates):
                        start_charge_pos = self.get_x_entry("charge", t, b, region = region, rate_idx = rate_idx)
                        rate = self.charging_rates[rate_idx]
                        end_time = t + 1
                        ## TODO: Adapt it to incorporate charging curves
                        end_battery = self.markov_decision_process.get_next_battery(rate, b) #min(b + rate, self.num_battery_levels - 1)
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
        ## Construct battery feasibility matrix
        battery_feasible_constraints_cnt = len(battery_feasible_col_lst)
        battery_feasible_val_lst = np.ones(battery_feasible_constraints_cnt)
        battery_feasible_row_lst = np.arange(battery_feasible_constraints_cnt)
        battery_feasible_mat = csr_matrix((battery_feasible_val_lst, (battery_feasible_row_lst, battery_feasible_col_lst)), shape = (battery_feasible_constraints_cnt, self.x_len))
        battery_feasible_target = np.zeros(battery_feasible_constraints_cnt)
        return flow_conservation_mat, flow_conservation_target, battery_feasible_mat, battery_feasible_target
    
    ## Deprecated!!
    def get_relevant_x_prev(self, t, region, battery):
        passenger_carry_idx_begin = t * self.num_battery_levels * self.num_regions * self.num_regions + battery * self.num_regions * self.num_regions + region * self.num_regions
        passenger_carry_idx_end = passenger_carry_idx_begin + self.num_regions
        reroute_idx_begin = passenger_carry_idx_begin + self.rerouting_flow_begin
        reroute_idx_end = reroute_idx_begin + self.num_regions
        charge_idx_begin = self.charging_flow_begin + t * self.num_charging_rates * self.num_battery_levels * self.num_regions + battery * self.num_regions + region
        charge_idx_end = charge_idx_begin + self.num_charging_rates * self.num_battery_levels * self.num_regions
        passenger_carry_x_ids = list(range(passenger_carry_idx_begin, passenger_carry_idx_end))
        reroute_x_ids = list(range(reroute_idx_begin, reroute_idx_end))
        charge_x_ids = list(range(charge_idx_begin, charge_idx_end, self.num_battery_levels * self.num_regions))
        return passenger_carry_x_ids, reroute_x_ids, charge_x_ids
    
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
    
    def get_fleet_status(self):
        status_mat = np.zeros((4, self.time_horizon))
        ## Travel
        for t in range(self.time_horizon):
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    ## Passenger-Carry
                    begin = t * self.num_battery_levels * self.num_regions * self.num_regions + origin * self.num_regions + dest
                    end = begin + self.num_battery_levels * self.num_regions * self.num_regions
                    trip_time = int(max(self.travel_time[t, origin * self.num_regions + dest], 1))
                    if self.num_days > 1:
                        idx_lst = np.arange(t, t + trip_time) % self.time_horizon
                    else:
                        idx_lst = np.arange(t, min(t + trip_time, self.time_horizon))
                    status_mat[0, idx_lst] += np.sum(self.x[begin:end:(self.num_regions ** 2)])
                    ## Reroute
                    begin = self.rerouting_flow_begin + begin
                    end = self.rerouting_flow_begin + end
                    if origin != dest:
                        status_mat[1, idx_lst] += np.sum(self.x[begin:end:(self.num_regions ** 2)])
                    ## Idle
                    else:
                        status_mat[3, t] += np.sum(self.x[begin:end:(self.num_regions ** 2)])
        ## Charge
        for t in range(self.time_horizon):
            for region in range(self.num_regions):
                for rate_idx in range(self.num_charging_rates):
                    begin = self.charging_flow_begin + t * self.num_battery_levels * self.num_regions * self.num_charging_rates + region * self.num_charging_rates + rate_idx
                    end = begin + self.num_battery_levels * self.num_regions * self.num_charging_rates
                    status_mat[2, t] += np.sum(self.x[begin:end:(self.num_regions * self.num_charging_rates)])
        ## Normalize by number of cars
        status_mat /= self.num_total_cars
        return status_mat
    
    def plot_stacked(self, x_arr, y_arr_lst, label_lst, xlabel, ylabel, title, figname):
        assert len(y_arr_lst) == len(label_lst)
        curr_lo = 0
        curr_hi = 0
        for i in range(len(label_lst)):
            curr_hi += y_arr_lst[i]
            label = label_lst[i]
            plt.fill_between(x_arr, curr_lo, curr_hi, label = label)
            curr_lo += y_arr_lst[i]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"TablePlots/{figname}.png")
        plt.clf()
        plt.close()
    
    def plot_fleet_status(self, suffix):
        status_mat = self.get_fleet_status()
        self.plot_stacked(np.arange(self.time_horizon), [status_mat[0,:], status_mat[1,:], status_mat[2,:], status_mat[3,:]], label_lst = ["% Passenger-Carrying Cars", "% Rerouting Cars", "% Charging Cars", "% Idling Cars"], xlabel = "Time Steps", ylabel = "% Cars", title = "", figname = f"lp_car_status_{suffix}")
    
    ## Assign demands to s_{ij}^{t + l, t} in increasing order of l
    ## Update f according to \sum_b f_{ij}^{t, b} = \sum_l s_{ij}^{t, t - l}
    ##  Assign plethora of f (if any) to the corresponding e, starting from lowest possible battery (it doesn't matter though)
    def cap_flow_with_demands(self):
        x_copy = self.x.copy()
        ## Update s
        for t in range(self.time_horizon):
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    trip_demand = self.trip_demands[t, origin * self.num_regions + dest]
                    for l in range(self.patience_time, -1, -1):
                        if t + l < self.time_horizon:
                            s_idx = self.slack_request_patience_begin + (t + l) * self.num_regions * self.num_regions * (self.patience_time + 1) + origin * self.num_regions * (self.patience_time + 1) + dest * (self.patience_time + 1) + l
                            num = min(trip_demand, x_copy[s_idx])
                            x_copy[s_idx] = num
                            trip_demand -= num
        ## Update f and e
        for t in range(self.time_horizon):
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    f_begin = t * self.num_battery_levels * self.num_regions * self.num_regions + origin * self.num_regions + dest
                    f_end = f_begin + self.num_battery_levels * self.num_regions * self.num_regions
                    fe_gap = self.num_regions * self.num_regions
                    f_total = np.sum(x_copy[f_begin:f_end:fe_gap])
                    s_begin = self.slack_request_patience_begin + t * self.num_regions * self.num_regions * (self.patience_time + 1) + origin * self.num_regions * (self.patience_time + 1) + dest * (self.patience_time + 1)
                    s_end = s_begin + self.patience_time + 1
                    s_total = np.sum(x_copy[s_begin:s_end])
                    fs_gap = f_total - s_total
                    for b in range(self.num_battery_levels):
                        f_idx = f_begin + b * self.num_regions * self.num_regions
                        e_idx = self.rerouting_flow_begin + f_idx
                        num = min(fs_gap, x_copy[f_idx])
                        x_copy[f_idx] -= num
                        x_copy[e_idx] += num
                        fs_gap -= num
        return x_copy
    
    def evaluate(self, return_action = True, seed = None, day_num = 0, strict = False, full_knowledge = False, fractional_cars = False, random_eval_round = 0):
        if seed is not None:
            torch.manual_seed(seed)
        self.markov_decision_process.reset_states(new_episode = day_num == 0, seed = seed)
        
        if random_eval_round == 0:
            self.reset_timestamp(fractional_cars, full_knowledge)
        
        if full_knowledge and random_eval_round == 0:
            obj_val_normalized = self.train()

        x_copy = self.cap_flow_with_demands()
        if fractional_cars:
            obj_val_normalized = np.sum(self.c * x_copy) / self.total_revenue
            return None, None, torch.tensor([0, obj_val_normalized]), None, torch.tensor(obj_val_normalized), None
        
        init_payoff = float(self.markov_decision_process.get_payoff_curr_ts(deliver = True))
        action_lst_ret = []
        payoff_lst = []
        discount_lst = []
        atomic_payoff_lst = []
        self.reset_x_infer(strict = strict, x_copy = x_copy)
        for t in range(self.time_horizon):
            available_car_ids = self.markov_decision_process.get_available_car_ids(True)
            num_available_cars = len(available_car_ids)
            for car_idx in range(num_available_cars):
                car_id = available_car_ids[car_idx]
                dest, eta, battery = self.markov_decision_process.get_car_info(car_id)
                action_assigned = False
                travel_x_ids, charge_x_ids = self.get_relevant_x(t, dest, battery, strict = strict)
                if eta == 0:
                    for i in range(len(charge_x_ids)):
                        x_id = charge_x_ids[i]
                        if self.x_charge[x_id] > 0:
                            action_prob = min(self.x_charge[x_id], 1)
#                            rv = np.random.binomial(n = 1, p = action_prob)
                            rv = int(round(action_prob))
                            if rv == 1:
                                self.x_charge[x_id] -= 1
                                action_id = self.markov_decision_process.query_action(("charge", dest, self.charging_rates[i]))
                                action_success = self.markov_decision_process.transit_within_timestamp(action, car_id = car_id)
                                if action_success:
                                    action_assigned = True
                                    break
                if not action_assigned:
                    travel_x_ids, charge_x_ids = self.get_relevant_x(t + eta, dest, battery, strict = strict)
                    for i in range(len(travel_x_ids)):
                        x_id = travel_x_ids[i]
                        if self.x_travel[x_id] > 0:
                            action_prob = min(self.x_travel[x_id], 1)
                            rv = np.random.binomial(n = 1, p = action_prob)
                            if rv == 1:
                                self.x_travel[x_id] -= 1
                                action_id = self.markov_decision_process.query_action(("travel", dest, i % self.num_regions))
                                action = self.all_actions[action_id]
                                action_success = self.markov_decision_process.transit_within_timestamp(action, car_id = car_id)
                                if action_success:
                                    action_assigned = True
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
        passenger_carrying_cars = self.markov_decision_process.passenger_carrying_cars
        return None, None, payoff_lst, action_lst_ret, discounted_payoff, passenger_carrying_cars

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
    def __init__(self, markov_decision_process = None, num_days = 1, gamma = 1, retrain = True, n_cpu = 1, add_integer_var = False, verbose = True):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process)
        self.reward_df = self.markov_decision_process.reward_df
        self.num_days = num_days
        self.max_tracked_eta = self.markov_decision_process.pickup_patience + self.markov_decision_process.max_travel_time
        self.state_to_id = self.markov_decision_process.state_to_id
        self.gamma = gamma
        self.retrain = retrain
        self.n_cpu = n_cpu
        self.add_integer_var = add_integer_var
        self.verbose = verbose
        if not self.verbose:
            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0)
                env.start()
    
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
            model.setParam("NodeMethod", 2)
            model.setParam("Crossover", 0)
            x = model.addMVar(n, lb = 0, vtype = GRB.CONTINUOUS, name = "x")
            if self.add_integer_var:
                n2 = self.B.shape[1]
                y = model.addMVar(n2, lb = 0, vtype = GRB.CONTINUOUS, name = "y")
                model.addConstr(self.A @ x + self.B @ y == self.b)
                model.addConstr(y.sum() <= 20)
                model.addConstr(y <= 5)
                model.addConstr(y >= 1)
            else:
                model.addConstr(self.A @ x == self.b)
    #        objective = gp.quicksum(self.c[r] * x[r] for r in range(n))
            objective = self.c @ x
            if self.add_integer_var:
                objective += self.c2 @ y
            model.setObjective(objective, GRB.MAXIMIZE)
            if self.verbose:
                print("\tOptimizing...")
            model.optimize()
            obj_val = model.ObjVal
            if self.verbose:
                print(obj_val / self.total_revenue, obj_val)
                print("\tGathering...")
            self.x = np.zeros(n)
            for i in tqdm(range(n), leave = False):
                self.x[i] = x[i].x
            self.describe_x()
#                with open("lp_b.txt", "a") as f:
#                    f.write(f"{self.b}\n")
            np.save("lp_x.npy", self.x)
            if self.add_integer_var:
                self.y = np.zeros(n2)
                for i in range(n2):
                    self.y[i] = y[i].x
                self.describe_y()
                np.save("lp_y.npy", self.y)
        else:
            self.x = np.load("lp_x.npy")
            obj_val = -1
        return obj_val #obj_val / self.total_revenue #self.x, obj_val
    
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
    def __init__(self, markov_decision_process = None, num_days = 1, gamma = 1, patience_time = 0, retrain = True, n_cpu = 1, charging_capacity_as_var = False, **kargs):
        super().__init__(markov_decision_process = markov_decision_process, num_days = num_days, gamma = gamma, retrain = retrain, n_cpu = n_cpu, add_integer_var = charging_capacity_as_var)
        self.patience_time = markov_decision_process.connection_patience + markov_decision_process.pickup_patience
        self.connection_patience = markov_decision_process.connection_patience
        self.pickup_patience = markov_decision_process.pickup_patience
        self.start_ts = 0
        self.charging_capacity_as_var = charging_capacity_as_var
        self.adjusted_time_horizon = self.time_horizon
        self.state_reduction = True
        self.all_actions_reduced = self.markov_decision_process.get_all_actions(state_reduction = True)
        self.construct_via_state_counts = False
        print("Constructing the solver...")
        self.load_data()
        self.construct_problem()
    
    def set_start_time(self, ts):
        self.start_ts = ts
        self.adjusted_time_horizon = self.time_horizon - self.start_ts
    
    def set_lp_option(self, state_counts = None):
        if state_counts is None:
            self.construct_via_state_counts = False
            self.state_counts = None
        else:
            self.construct_via_state_counts = True
            self.state_counts = state_counts
    
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
        self.charging_time = self.markov_decision_process.charging_time
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
        travel_flow_len = self.adjusted_time_horizon * self.num_battery_levels * self.num_regions * self.num_regions * (self.pickup_patience + 1)
        reroute_flow_len = self.adjusted_time_horizon * self.num_battery_levels * self.num_regions * self.num_regions
        charging_flow_len = self.adjusted_time_horizon * self.num_battery_levels * self.num_regions * self.num_charging_rates
        self.rerouting_flow_begin = travel_flow_len
        self.charging_flow_begin = self.rerouting_flow_begin + reroute_flow_len
        pass_len = self.adjusted_time_horizon * self.num_battery_levels * self.num_regions * self.pickup_patience
        self.pass_begin = self.charging_flow_begin + charging_flow_len
        trip_demand_extra_len = self.adjusted_time_horizon * self.num_regions * self.num_regions
        charging_facility_extra_len = self.adjusted_time_horizon * self.num_regions * self.num_charging_rates
        self.trip_demand_extra_begin = self.pass_begin + pass_len
        self.charging_facility_extra_begin = self.trip_demand_extra_begin + trip_demand_extra_len
        ## Add dummy variables for number of requests at time t but taken at time s
        slack_request_patience_len = self.adjusted_time_horizon * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1)
        self.slack_request_patience_begin = self.charging_facility_extra_begin + charging_facility_extra_len
        self.x_len = travel_flow_len + reroute_flow_len + charging_flow_len + pass_len + trip_demand_extra_len + charging_facility_extra_len + slack_request_patience_len
        self.y_len = self.num_regions * self.num_charging_rates
#        self.x = np.zeros(self.x_len)
    
    def get_x_entry(self, entry_type, t, b, origin = None, dest = None, region = None, rate_idx = None, lp = 0, lc = 0):
        assert entry_type in ["passenger-carry", "reroute", "charge", "pass", "slack"]
        assert lp >= 0 and lp <= self.pickup_patience
        if entry_type in ["reroute", "charge"]:
            assert lp == 0
        if entry_type == "charge":
            assert region is not None and rate_idx is not None
            ans = t * self.num_battery_levels * self.num_regions * self.num_charging_rates + b * self.num_regions * self.num_charging_rates + region * self.num_charging_rates + rate_idx
            return int(self.charging_flow_begin + ans)
        elif entry_type == "pass":
            ans = t * self.num_battery_levels * self.num_regions * self.pickup_patience + b * self.num_regions * self.pickup_patience + region * self.pickup_patience + lp - 1
            return int(self.pass_begin + ans)
        elif entry_type == "passenger-carry":
            ans = t * self.num_battery_levels * self.num_regions * self.num_regions * (self.pickup_patience + 1) + b * self.num_regions * self.num_regions * (self.pickup_patience + 1) + origin * self.num_regions * (self.pickup_patience + 1) + dest * (self.pickup_patience + 1) + lp
            return int(ans)
        elif entry_type == "reroute":
            ans = t * self.num_battery_levels * self.num_regions * self.num_regions + b * self.num_regions * self.num_regions + origin * self.num_regions + dest
            ans += self.rerouting_flow_begin
            return int(ans)
        else: ## if entry_type == "slack":
            ans = self.slack_request_patience_begin + t * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1) + origin * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1) + dest * (self.pickup_patience + 1) * (self.connection_patience + 1) + lp * (self.connection_patience + 1) + lc
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
            x_travel = np.array([[x_copy[self.rerouting_flow_begin:self.charging_flow_begin][(t * travel_len):((t + 1) * travel_len)][i::(self.num_regions ** 2)] for i in range(self.num_regions ** 2)] for t in range(self.adjusted_time_horizon)]).sum(axis = 2)
#            for t in range(self.adjusted_time_horizon):
#                for lp in range(self.pickup_patience + 1):
#                    if self.num_days == 1:
#                        t_new = min(t + lp, self.adjusted_time_horizon)
#                    else:
#                        t_new = (t + lp) % self.adjusted_time_horizon
#                    for lc in range(self.connection_patience + 1):
#                        slack_begin = self.slack_request_patience_begin + t * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1) + lp * (self.connection_patience + 1) + lc
#                        slack_end = self.slack_request_patience_begin + (t + 1) * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1)
#                        slack_step = (self.pickup_patience + 1) * (self.connection_patience + 1)
#                        x_travel[(t_new * self.num_regions * self.num_regions):((t_new + 1) * self.num_regions * self.num_regions)] += x_copy[slack_begin:slack_end:slack_step]
            for lp in range(self.pickup_patience + 1):
                x_travel += np.array([[x_copy[:self.rerouting_flow_begin][(t * travel_len):((t + 1) * travel_len)][(i * (self.pickup_patience + 1) + lp)::((self.num_regions ** 2) * (self.pickup_patience + 1))] for i in range(self.num_regions ** 2)] for t in range(self.adjusted_time_horizon)]).sum(axis = 2)
            # x_travel = x_travel[:self.adjusted_time_horizon,:] + x_travel[self.adjusted_time_horizon:,:]
            x_travel = x_travel.flatten()
            x_charge = np.array([[x_copy[self.charging_flow_begin:self.trip_demand_extra_begin][(t * charge_len):((t + 1) * charge_len)][i::(self.num_regions)] for i in range(self.num_regions)] for t in range(self.adjusted_time_horizon * self.num_charging_rates)]).sum(axis = 2).flatten()
        self.x_travel = x_travel #x_travel.round().astype(int)
        self.x_charge = x_charge

    def reset_x_infer_detailed(self, strict = True, x_copy = None):
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
            travel_len = self.adjusted_time_horizon * self.num_regions * self.num_regions
            travel_len_total = (self.pickup_patience + 1) * travel_len
            travel_len_single = self.num_battery_levels * self.num_regions * self.num_regions
            charge_len = self.num_battery_levels * self.num_regions
            x_travel = np.zeros(travel_len_total)
            x_travel[:travel_len] = np.array([[x_copy[self.rerouting_flow_begin:self.charging_flow_begin][(t * travel_len_single):((t + 1) * travel_len_single)][i::(self.num_regions ** 2)] for i in range(self.num_regions ** 2)] for t in range(self.adjusted_time_horizon)]).sum(axis = 2).flatten()
            for eta in range(self.pickup_patience + 1):
                for lc in range(self.connection_patience + 1):
                    x_travel[(eta * travel_len):((eta + 1) * travel_len)] += x_copy[self.slack_request_patience_begin:][(eta * (self.connection_patience + 1) + lc)::((self.pickup_patience + 1) * (self.connection_patience + 1))]
            x_charge = np.array([[x_copy[self.charging_flow_begin:self.trip_demand_extra_begin][(t * charge_len):((t + 1) * charge_len)][i::(self.num_regions)] for i in range(self.num_regions)] for t in range(self.adjusted_time_horizon * self.num_charging_rates)]).sum(axis = 2).flatten()
        self.x_travel = x_travel #x_travel.round().astype(int)
        self.x_charge = x_charge #x_charge.round().astype(int)
    
    def describe_x(self, fname = "lp_debug.txt"):
        eps = 0.01
        with open(fname, "w") as f:
            for entry_type in ["passenger-carry", "reroute"]:
                for t in range(self.adjusted_time_horizon):
                    for b in range(self.num_battery_levels):
                        for origin in range(self.num_regions):
                            for dest in range(self.num_regions):
                                if entry_type == "reroute":
                                    x_entry = self.get_x_entry(entry_type, t, b, origin = origin, dest = dest)
                                    #if self.x[x_entry] >= eps and (entry_type == "passenger-carry" or origin != dest):
                                    if self.x[x_entry] >= eps:
                                        val = self.x[x_entry]
                                        f.write(f"{entry_type}, t = {t}, b = {b}, origin = {origin}, dest = {dest}, val = {val}\n")
                                else:
                                    for lp in range(self.pickup_patience + 1):
                                        x_entry = self.get_x_entry(entry_type, t, b, origin = origin, dest = dest, lp = lp)
                                        if self.x[x_entry] >= eps:
                                            val = self.x[x_entry]
                                            f.write(f"{entry_type}, t = {t}, b = {b}, origin = {origin}, dest = {dest}, eta = {lp}, val = {val}\n")
            for t in range(self.adjusted_time_horizon):
                for b in range(self.num_battery_levels):
                    for region in range(self.num_regions):
                        for rate_idx in range(self.num_charging_rates):
                                x_entry = self.get_x_entry("charge", t, b, region = region, rate_idx = rate_idx)
                                if self.x[x_entry] >= eps:
                                    val = self.x[x_entry]
                                    f.write(f"charge, t = {t}, b = {b}, region = {region}, rate = {self.charging_rates[rate_idx]}, val = {val}\n")
            f.write("============\n")
            for t in range(self.adjusted_time_horizon):
                for origin in range(self.num_regions):
                    for dest in range(self.num_regions):
                        for lp in range(self.pickup_patience + 1):
                            for lc in range(self.connection_patience + 1):
                                x_entry = self.get_x_entry("slack", t, b = None, origin = origin, dest = dest, lp = lp, lc = lc)
                                if self.x[x_entry] >= eps:
                                    val = self.x[x_entry]
                                    f.write(f"slack, t = {t}, origin = {origin}, dest = {dest}, lp = {lp}, lc = {lc}, val = {val}\n")
    
    def describe_y(self, fname = "lp_debug.txt"):
        with open(fname, "a") as f:
            for region in range(self.num_regions):
                for rate_idx in range(self.num_charging_rates):
                    pos = region * self.num_charging_rates + rate_idx
                    if self.y[pos] > 0:
                        val = self.y[pos]
                        f.write(f"Charging Station, region = {region}, rate = {self.charging_rates[rate_idx]}, val = {val}\n")
    
    def get_flow_conserv_entry(self, t, b, region, lp = 0):
        assert lp >= 0 and lp <= self.pickup_patience
        return int(t * self.num_battery_levels * self.num_regions * (self.pickup_patience + 1) + b * self.num_regions * (self.pickup_patience + 1) + region * (self.pickup_patience + 1) + lp)
    
    def construct_obj(self):
        self.c = np.zeros(self.x_len)
        for t in range(self.adjusted_time_horizon):
            for b in range(self.num_battery_levels):
                ## self.adjusted_time_horizon * self.num_battery_levels * self.num_regions * self.num_regions * (self.pickup_patience + 1)
                for l in range(self.pickup_patience + 1):
                    begin = t * self.num_battery_levels * self.num_regions ** 2 * (self.pickup_patience + 1) + b * self.num_regions ** 2 * (self.pickup_patience + 1) + l
                    self.c[begin:(begin + self.num_regions ** 2 * (self.pickup_patience + 1)):(self.pickup_patience + 1)] = self.trip_rewards[t + self.start_ts,:] * self.gamma ** t
        for t in range(self.adjusted_time_horizon):
            for rate_idx in range(self.num_charging_rates):
                for b in range(self.num_battery_levels):
                    unit_cost = self.charging_costs[t + self.start_ts, rate_idx]
                    rate = self.charging_rates[rate_idx]
                    next_battery = self.markov_decision_process.get_next_battery(rate, b)
                    cost = (next_battery - b) / rate * unit_cost
                    begin = self.charging_flow_begin + t * self.num_charging_rates * self.num_battery_levels * self.num_regions + rate_idx * self.num_battery_levels * self.num_regions + b * self.num_regions
                    end = begin + self.num_regions
                    self.c[begin:end] = cost * self.gamma ** t
        self.c2 = np.zeros(self.y_len)
        # self.c2 += -27 #-13.36 #-0.1

    def construct_trip_demand_matrix_single(self, t_lo, t_hi):
        trip_demand_target = np.zeros(self.adjusted_time_horizon * self.num_regions * self.num_regions * (self.pickup_patience + 1))
        slack_request_patience_target = np.zeros(self.adjusted_time_horizon * self.num_regions * self.num_regions)
        slack_request_patience_lst = []
        trip_demand_lst = []
        for t in tqdm(range(t_lo, t_hi), leave = False):
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    slack_request_patience_vec = np.zeros(self.x_len)
                    pos = t * self.num_regions * self.num_regions + origin * self.num_regions + dest
                    for lp in range(self.pickup_patience + 1):
                        trip_demand_vec = np.zeros(self.x_len)
                        begin = t * self.num_battery_levels * self.num_regions * self.num_regions * (self.pickup_patience + 1) + origin * self.num_regions * (self.pickup_patience + 1) + dest * (self.pickup_patience + 1) + lp
                        end = begin + self.num_battery_levels * self.num_regions * self.num_regions * (self.pickup_patience + 1)
                        trip_demand_vec[begin:end:(self.num_regions ** 2 * (self.pickup_patience + 1))] = 1
                        ## Slack: self.adjusted_time_horizon * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1)
                        slack_begin = self.slack_request_patience_begin + t * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1) + origin * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1) + dest * (self.pickup_patience + 1) * (self.connection_patience + 1) + lp * (self.connection_patience + 1)
                        slack_terminal = self.slack_request_patience_begin + (self.adjusted_time_horizon) * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1)
                        if self.num_days == 1:
                            end_1_t = self.connection_patience + 1 #min(self.connection_patience, t) + 1
                            end_2_t = min(self.connection_patience + 1, self.adjusted_time_horizon - t)
                            slack_end = slack_begin + end_1_t
                            slack_end_2 = slack_begin + end_2_t * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1)
                            slack_lst_1 = np.arange(slack_begin, slack_end)
                            slack_lst_2 = np.arange(slack_begin, slack_end_2, (self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1) + 1))
                        else:
                            end_1_t = self.connection_patience + 1 #self.connection_patience + 1 #self.patience_time + 1 #min(self.patience_time, t) + 1
                            end_2_t = self.connection_patience + 1 #self.patience_time + 1
                            slack_end = slack_begin + end_1_t
                            slack_end_2 = slack_begin + end_2_t * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1)
                            slack_lst_1 = np.arange(slack_begin, slack_end)
                            slack_lst_2 = np.arange(slack_begin, slack_end_2, (self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1) + 1))
                            for i in range(len(slack_lst_2)):
                                if slack_lst_2[i] > slack_terminal:
                                    slack_lst_2[i] -= self.adjusted_time_horizon * self.num_regions * self.num_regions * (self.pickup_patience + 1) * (self.connection_patience + 1)
                        trip_demand_vec[slack_lst_1] = -1
                        slack_request_patience_vec[slack_lst_2] = 1
                        trip_demand_vec = csr_matrix(trip_demand_vec)
                        trip_demand_lst.append(trip_demand_vec)
                    slack_request_patience_vec[self.trip_demand_extra_begin + pos] = 1
                    slack_request_patience_vec = csr_matrix(slack_request_patience_vec)
                    slack_request_patience_lst.append(slack_request_patience_vec)
#                    trip_demand_mat[pos, begin:end:(self.num_regions ** 2)] = 1
#                    trip_demand_mat[pos, self.trip_demand_extra_begin + pos] = 1
                    if self.construct_via_state_counts and t == 0:
                        for stag_time in range(self.connection_patience + 1):
                            trip_idx = self.state_to_id["trip"][(origin, dest, stag_time)]
                            trip_num = int(self.state_counts[trip_idx])
                            slack_request_patience_target[pos] += trip_num
                    else:
                        slack_request_patience_target[pos] = self.trip_demands[t + self.start_ts, origin * self.num_regions + dest]
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
        if self.construct_via_state_counts:
            self.flow_conservation_target_sc, total_flow_target_sc = self.construct_constraints_from_state_counts(self.state_counts)
        ### Flow conservation and Battery Feasibility
        print("\t\tConstructing flow conservation matrix...")
        flow_conservation_mat, flow_conservation_target, battery_feasible_mat, battery_feasible_target = self.construct_flow_conservation_battery_feasibility_matrix()
        ### Trip demand
        print("\t\tConstructing trip demand matrix...")
#        trip_demand_mat = np.zeros((self.time_horizon * self.num_regions * self.num_regions, self.x_len))
        batch_size = int(math.ceil(self.adjusted_time_horizon / self.n_cpu))
        results = Parallel(n_jobs = self.n_cpu)(delayed(self.construct_trip_demand_matrix_single)(
            i * batch_size, min((i + 1) * batch_size, self.adjusted_time_horizon)
        ) for i in range(self.n_cpu))
        trip_demand_target = np.zeros(self.adjusted_time_horizon * self.num_regions * self.num_regions * (self.pickup_patience + 1))
        slack_request_patience_target = np.zeros(self.adjusted_time_horizon * self.num_regions * self.num_regions)
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
        charging_facility_mat = np.zeros((self.time_horizon * self.num_regions * self.num_charging_rates, self.x_len))
        charging_facility_target = np.zeros(self.adjusted_time_horizon * self.num_regions * self.num_charging_rates)
        charging_facility_lst = []
        charging_facility_lst2 = []
        for t in tqdm(range(self.adjusted_time_horizon), leave = False):
            if self.num_days == 1:
                t_lst = np.arange(t, min(t + self.charging_time, self.adjusted_time_horizon))
            else:
                t_lst = [x if x < self.time_horizon else x - self.time_horizon for x in np.arange(t, t + self.charging_time)]
            t_lst = np.array(t_lst)
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
                    pos_lst = t_lst * self.num_regions * self.num_charging_rates + region * self.num_charging_rates + rate_idx
                    charging_facility_mat[pos_lst, begin:end:(self.num_regions)] = 1
                    charging_facility_mat[pos, self.charging_facility_extra_begin + pos] = 1
#                    charging_facility_mat[pos, begin:end:(self.num_regions)] = 1
                    if self.charging_capacity_as_var:
                        charging_facility_vec2 = np.zeros(self.y_len)
                        pos2 = region * self.num_charging_rates + rate_idx
                        charging_facility_vec2[pos2] = -1
                        charging_facility_vec2 = csr_matrix(charging_facility_vec2)
                        charging_facility_lst2.append(charging_facility_vec2)
                    else:
                        charging_facility_target[pos] = self.charging_facility_num[region, rate_idx]
#        charging_facility_mat = vstack(charging_facility_lst)
        charging_facility_mat = csr_matrix(charging_facility_mat)
        if self.charging_capacity_as_var:
            charging_facility_mat2 = vstack(charging_facility_lst2)
        ### Total flows
        print("\t\tConstructing total flow matrix...")
        total_flow_mat = np.zeros((self.adjusted_time_horizon, self.x_len))
        for t in tqdm(range(self.adjusted_time_horizon), leave = False):
            passenger_len = self.num_battery_levels * self.num_regions * self.num_regions
            charge_len = self.num_charging_rates * self.num_battery_levels * self.num_regions
            pass_len = self.num_battery_levels * self.num_regions * self.pickup_patience
#            total_flow_mat[t, (t * passenger_len):((t + 1) * passenger_len)] = 1
#            total_flow_mat[t, (self.rerouting_flow_begin + t * passenger_len):(self.rerouting_flow_begin + (t + 1) * passenger_len)] = 1
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    trip_time = int(max(self.travel_time[t + self.start_ts, origin * self.num_regions + dest], 1))
                    trip_time2 = 1
                    end_time_1 = t + max(trip_time + self.pickup_patience, 1)
                    if self.num_days == 1:
                        t_lst = np.arange(t, min(end_time_1, self.adjusted_time_horizon))
                        t_lst2 = np.arange(t, min(t + trip_time2, self.adjusted_time_horizon))
                    else:
                        t_lst = [x if x < self.time_horizon else x - self.time_horizon for x in np.arange(t, end_time_1)]
                        t_lst2 = [x if x < self.time_horizon else x - self.time_horizon for x in np.arange(t, t + trip_time2)]
                    for b in range(self.num_battery_levels):
                        passenger_begin = t * self.num_battery_levels * self.num_regions * self.num_regions * (self.pickup_patience + 1) + b * self.num_regions * self.num_regions * (self.pickup_patience + 1) + origin * self.num_regions * (self.pickup_patience + 1) + dest * (self.pickup_patience + 1)
                        reroute_begin = self.rerouting_flow_begin + t * self.num_battery_levels * self.num_regions * self.num_regions + b * self.num_regions * self.num_regions + origin * self.num_regions + dest
                        total_flow_mat[t_lst, passenger_begin] = 1
                        if origin != dest:
                            total_flow_mat[t_lst, reroute_begin] = 1
                        else:
                            total_flow_mat[t_lst2, reroute_begin] = 1
                    for lp in range(1, self.pickup_patience + 1):
                        trip_time = int(max(self.travel_time[(t + self.start_ts + lp) % self.time_horizon, origin * self.num_regions + dest], 1))
                        if self.num_days == 1:
                            t_lst = np.arange(min(t + lp, self.adjusted_time_horizon), min(t + lp + trip_time, self.adjusted_time_horizon))
                        else:
                            t_lst = [x if x < self.time_horizon else x - self.time_horizon for x in np.arange(t + lp, t + lp + trip_time)]
                        for b in range(self.num_battery_levels):
                            passenger_begin = t * self.num_battery_levels * self.num_regions * self.num_regions * (self.pickup_patience + 1) + b * self.num_regions * self.num_regions * (self.pickup_patience + 1) + origin * self.num_regions * (self.pickup_patience + 1) + dest * (self.pickup_patience + 1) + lp
                            total_flow_mat[t_lst, passenger_begin] = 1
            if self.num_days == 1:
                t_lst = np.arange(t, min(t + self.charging_time, self.adjusted_time_horizon))
            else:
                t_lst = [x if x < self.time_horizon else x - self.time_horizon for x in np.arange(t, t + self.charging_time)]
            total_flow_mat[t_lst, (self.charging_flow_begin + t * charge_len):(self.charging_flow_begin + (t + 1) * charge_len)] = 1
            total_flow_mat[t, (self.pass_begin + t * pass_len):(self.pass_begin + (t+1) * pass_len)] = 1
        total_flow_mat = csr_matrix(total_flow_mat)
            
        total_flow_target = np.ones(self.adjusted_time_horizon) * self.num_total_cars
        if self.construct_via_state_counts:
            total_flow_target = total_flow_target_sc
        ### Concatenate together
        self.A = vstack((flow_conservation_mat, trip_demand_mat, slack_request_patience_mat, charging_facility_mat, total_flow_mat, battery_feasible_mat))
        if self.charging_capacity_as_var:
            leading_zero_B = csr_matrix((flow_conservation_mat.shape[0] + trip_demand_mat.shape[0] + slack_request_patience_mat.shape[0], self.y_len))
            trailing_zero_B = csr_matrix((total_flow_mat.shape[0] + battery_feasible_mat.shape[0], self.y_len))
            self.B = vstack((leading_zero_B, charging_facility_mat2, trailing_zero_B))
        self.b = np.concatenate((flow_conservation_target, trip_demand_target, slack_request_patience_target, charging_facility_target, total_flow_target, battery_feasible_target), axis = None)
#        self.b = csr_matrix(self.b.reshape((len(self.b), 1)))
    
    def construct_constraints_from_state_counts(self, state_counts):
        flow_conservation_target = np.zeros(self.adjusted_time_horizon * self.num_battery_levels * self.num_regions)
        total_flow_mat = np.zeros((self.adjusted_time_horizon, self.x_len))
        total_flow_target = np.ones(self.adjusted_time_horizon) * self.num_total_cars
        ## Idling cars + cars with eta > 0
        ### Traveling cars
        ### ONLY AT BEGINNING OF EACH DECISION EPOCH
        for region in range(self.num_regions):
            for battery in range(self.num_battery_levels):
                for eta in range(self.max_tracked_eta + 1):
                    car_idx = self.state_to_id["car"][("general", region, eta, battery)]
                    car_num = int(state_counts[car_idx])
                    car_idx2 = self.state_to_id["car"][("assigned", region, eta, battery)]
                    car_num2 = int(state_counts[car_idx2])
                    passenger_begin = battery * self.num_regions * self.num_regions + 0 * self.num_regions + region
                    reroute_begin = self.rerouting_flow_begin + passenger_begin
#                    total_flow_mat[:min(eta, self.adjusted_time_horizon), passenger_begin] = 1
                    total_flow_target[:min(eta, self.adjusted_time_horizon)] -= (car_num + car_num2)
                    if eta < self.adjusted_time_horizon:
                        flow_idx = eta * self.num_battery_levels * self.num_regions + battery * self.num_regions + region
                        flow_conservation_target[flow_idx] += car_num + car_num2
        ## Charging cars
        for region in range(self.num_regions):
            for battery in range(self.num_battery_levels):
                for rate_idx in range(self.num_charging_rates):
                    rate = self.charging_rates[rate_idx]
                    car_idx = self.state_to_id["car"][("charged", region, battery, rate)]
                    car_num = int(state_counts[car_idx])
                    total_flow_target[0] -= car_num
                    if self.adjusted_time_horizon > 1:
                        flow_idx = self.num_battery_levels * self.num_regions + battery * self.num_regions + region
                        flow_conservation_target[flow_idx] += car_num
        return flow_conservation_target, total_flow_target
    
    ## Flow conservation at each (time, battery, region)
    ##   - Passenger-carrying to & from each region
    ##   - Rerouting to & from each region
    ##   - Charging at each rate
    ##
    def construct_flow_conservation_battery_feasibility_matrix(self):
        flow_conservation_target = np.zeros(self.adjusted_time_horizon * self.num_battery_levels * self.num_regions * (self.pickup_patience + 1))
        row_lst = []
        col_lst = []
        val_lst = []
        battery_feasible_col_lst = []
        ## Populate initial car flow
        if self.num_days == 1 and not self.construct_via_state_counts:
            for region in tqdm(range(self.num_regions), leave = False):
                for battery in range(self.num_battery_levels):
                    num = self.init_car_num[region * self.num_battery_levels + battery]
                    flow_conservation_target[battery * self.num_regions + region] = num
        elif self.construct_via_state_counts:
            flow_conservation_target = self.flow_conservation_target_sc
        ## Populate flow conservation matrix
        for t in tqdm(range(self.adjusted_time_horizon), leave = False):
            for b in tqdm(range(self.num_battery_levels), leave = False):
                ## Populate traveling flows
                for origin in range(self.num_regions):
                    for dest in range(self.num_regions):
                        ## Populate trip fulfilling flows
                        for lp in range(self.pickup_patience + 1):
                            trip_time = max(self.travel_time[(t + self.start_ts + lp) % self.time_horizon, origin * self.num_regions + dest], 1)
                            battery_cost = self.battery_consumption[origin * self.num_regions + dest]
                            passenger_pos = self.get_x_entry("passenger-carry", t, b, origin = origin, dest = dest, lp = lp)
                            start_row = self.get_flow_conserv_entry(t, b, origin, lp = lp)
                            start_row_filled = False
                            if b >= battery_cost:
                                row_lst += [start_row]
                                col_lst += [passenger_pos]
                                val_lst += [1]
                                start_row_filled = True
                                end_time = t + max(lp + trip_time - self.pickup_patience, 1)
                                if end_time >= self.time_horizon and self.num_days > 1:
                                    end_row = self.get_flow_conserv_entry(end_time - self.time_horizon, b - battery_cost, dest, lp = min(self.pickup_patience, lp + trip_time - 1))
                                elif end_time < self.adjusted_time_horizon:
                                    end_row = self.get_flow_conserv_entry(end_time, b - battery_cost, dest, lp = min(self.pickup_patience, lp + trip_time - 1))
                                else:
                                    end_row = None
                                if end_row is not None:
                                    row_lst += [end_row]
                                    col_lst += [passenger_pos]
                                    val_lst += [-1]
                            else:
                                battery_feasible_col_lst.append(passenger_pos)
                        ## Populate rerouting flows
                        trip_time = max(self.travel_time[min(t + self.start_ts, self.time_horizon - 1), origin * self.num_regions + dest], 1)
                        battery_cost = self.battery_consumption[origin * self.num_regions + dest]
                        reroute_pos = self.get_x_entry("reroute", t, b, origin = origin, dest = dest, lp = 0)
                        start_row = self.get_flow_conserv_entry(t, b, origin, lp = 0)
                        start_row_filled = False
                        if b >= battery_cost:
                            row_lst += [start_row]
                            col_lst += [reroute_pos]
                            val_lst += [1]
                            start_row_filled = True
                            end_time = t + max(trip_time - self.pickup_patience, 1)
                            if end_time >= self.time_horizon and self.num_days > 1:
                                end_row = self.get_flow_conserv_entry(end_time - self.time_horizon, b - battery_cost, dest, lp = min(self.pickup_patience, trip_time - 1))
                            elif end_time < self.adjusted_time_horizon:
                                end_row = self.get_flow_conserv_entry(end_time, b - battery_cost, dest, lp = min(self.pickup_patience, trip_time - 1))
                            else:
                                end_row = None
                            if end_row is not None:
                                if origin != dest:
                                    row_lst += [end_row]
                                    col_lst += [reroute_pos]
                                    val_lst += [-1]
                        else:
                            if origin != dest:
                                battery_feasible_col_lst.append(reroute_pos)
                        if origin == dest:
                            if not start_row_filled:
                                row_lst += [start_row]
                                col_lst += [reroute_pos]
                                val_lst += [1]
                            end_time_reroute = t + 1
                            if end_time_reroute >= self.time_horizon and self.num_days > 1:
                                end_row_reroute = self.get_flow_conserv_entry(end_time_reroute - self.time_horizon, b, dest, lp = 0)
                            elif end_time_reroute < self.adjusted_time_horizon:
                                end_row_reroute = self.get_flow_conserv_entry(end_time_reroute, b, dest, lp = 0)
                            else:
                                end_row_reroute = None
                            if end_row_reroute is not None:
                                row_lst += [end_row_reroute]
                                col_lst += [reroute_pos]
                                val_lst += [-1]
                for region in range(self.num_regions):
                    ## Populate charging flows
                    for rate_idx in range(self.num_charging_rates):
                        start_charge_pos = self.get_x_entry("charge", t, b, region = region, rate_idx = rate_idx)
                        rate = self.charging_rates[rate_idx]
                        end_time = t + self.charging_time
                        ## TODO: Adapt it to incorporate charging curves
                        end_battery = self.markov_decision_process.get_next_battery(rate, b) #min(b + rate, self.num_battery_levels - 1)
                        charge_pos = self.get_x_entry("charge", t, b, region = region, rate_idx = rate_idx)
                        start_row = self.get_flow_conserv_entry(t, b, region, lp = 0)
                        row_lst += [start_row]
                        col_lst += [charge_pos]
                        val_lst += [1]
                        if end_time >= self.time_horizon and self.num_days > 1:
                            end_row = self.get_flow_conserv_entry(end_time - self.time_horizon, end_battery, region, lp = min(self.charging_time - 1, self.pickup_patience))
                        elif end_time < self.adjusted_time_horizon:
                            end_row = self.get_flow_conserv_entry(end_time, end_battery, region, lp = min(self.charging_time - 1, self.pickup_patience))
                        else:
                            end_row = None
                        if end_row is not None:
                            row_lst += [end_row]
                            col_lst += [charge_pos]
                            val_lst += [-1]
                    ## Populate pass flows
                    for lp in range(1, self.pickup_patience + 1):
                        pass_pos = self.get_x_entry("pass", t, b, region = region, lp = lp)
                        start_row = self.get_flow_conserv_entry(t, b, region, lp = lp)
                        row_lst += [start_row]
                        col_lst += [pass_pos]
                        val_lst += [1]
                        end_time = t + 1
                        if end_time >= self.time_horizon and self.num_days > 1:
                            end_row = self.get_flow_conserv_entry(end_time - self.time_horizon, b, region, lp = lp - 1)
                        elif end_time < self.adjusted_time_horizon:
                            end_row = self.get_flow_conserv_entry(end_time, b, region, lp = lp - 1)
                        else:
                            end_row = None
                        if end_row is not None:
                            row_lst += [end_row]
                            col_lst += [pass_pos]
                            val_lst += [-1]
        ## Construct flow_conservation_mat row by row
        flow_conservation_mat = csr_matrix((val_lst, (row_lst, col_lst)), shape = (self.adjusted_time_horizon * self.num_battery_levels * self.num_regions * (self.pickup_patience + 1), self.x_len))
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
    
    def get_relevant_x(self, t, region, battery, eta, strict = True):
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
            if t < self.adjusted_time_horizon:
                travel_idx_begin = t * self.num_regions * self.num_regions + region * self.num_regions
                travel_idx_end = travel_idx_begin + self.num_regions
                eta_offset = eta * self.adjusted_time_horizon * self.num_regions * self.num_regions
                travel_idx_begin, travel_idx_end = travel_idx_begin + eta_offset, travel_idx_end + eta_offset
                charge_idx_begin = t * self.num_charging_rates * self.num_regions + region
                charge_idx_end = charge_idx_begin + self.num_charging_rates * self.num_regions
                travel_x_ids = list(range(travel_idx_begin, travel_idx_end))
                charge_x_ids = list(range(charge_idx_begin, charge_idx_end, self.num_regions))
            else:
                travel_x_ids, charge_x_ids = [], []
        return travel_x_ids, charge_x_ids
    
    def get_fleet_status(self):
        status_mat = np.zeros((4, self.time_horizon))
        ## Travel
        for t in range(self.time_horizon):
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    trip_time = int(max(self.travel_time[t, origin * self.num_regions + dest], 1))
                    ## Passenger-Carry
                    for lp in range(self.pickup_patience + 1):
                        begin = t * self.num_battery_levels * self.num_regions * self.num_regions * (self.pickup_patience + 1) + origin * self.num_regions * (self.pickup_patience + 1) + dest * (self.pickup_patience + 1) + lp
                        end = begin + self.num_battery_levels * self.num_regions * self.num_regions * (self.pickup_patience + 1)
                        if self.num_days > 1:
                            idx_lst = np.arange(t + lp, t + trip_time + lp) % self.time_horizon
                        else:
                            idx_lst = np.arange(min(t + lp, self.time_horizon), min(t + trip_time + lp, self.time_horizon))
                        status_mat[0, idx_lst] += np.sum(self.x[begin:end:((self.num_regions ** 2)*(self.pickup_patience + 1))])
                    ## Reroute
                    begin = t * self.num_battery_levels * self.num_regions * self.num_regions + origin * self.num_regions + dest
                    end = begin + self.num_battery_levels * self.num_regions * self.num_regions
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
                    if self.num_days > 1:
                        idx_lst = np.arange(t, t + self.charging_time) % self.time_horizon
                    else:
                        idx_lst = np.arange(t, min(t + self.charging_time, self.time_horizon))
                    status_mat[2, idx_lst] += np.sum(self.x[begin:end:(self.num_regions * self.num_charging_rates)])
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
    def cap_flow_with_demands(self, cap_demand = True):
        x_copy = self.x.copy()
        ## Update s
        for t in range(self.adjusted_time_horizon):
            for origin in range(self.num_regions):
                for dest in range(self.num_regions):
                    if cap_demand:
                        trip_demand = self.trip_demands[t, origin * self.num_regions + dest]
                    else:
                        trip_demand = np.inf
                    for l in range(self.patience_time, -1, -1):
                        if t + l < self.time_horizon:
                            s_idx = self.slack_request_patience_begin + (t + l) * self.num_regions * self.num_regions * (self.patience_time + 1) + origin * self.num_regions * (self.patience_time + 1) + dest * (self.patience_time + 1) + l
                            num = min(trip_demand, x_copy[s_idx])
                            x_copy[s_idx] = num
                            trip_demand -= num
        ## Update f and e
        for t in range(self.adjusted_time_horizon):
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
    
    def evaluate(self, return_action = True, return_data = False, seed = None, day_num = 0, strict = False, full_knowledge = False, fractional_cars = False, random_eval_round = 0, markov_decision_process = None):
        if seed is not None:
            torch.manual_seed(seed)
        if markov_decision_process is None:
            markov_decision_process = self.markov_decision_process
        markov_decision_process.reset_states(new_episode = day_num == 0, seed = seed)
        
        if random_eval_round == 0:
            self.reset_timestamp(fractional_cars, full_knowledge)
        
        if full_knowledge and random_eval_round == 0:
            obj_val_normalized = self.train()

        if fractional_cars:
            x_copy = self.cap_flow_with_demands(cap_demand = fractional_cars)
            obj_val_normalized = np.sum(self.c * x_copy) / self.total_revenue
            return None, None, torch.tensor([0, obj_val_normalized]), None, torch.tensor(obj_val_normalized), None
        x_copy = self.x.copy()
        
        init_payoff = float(markov_decision_process.get_payoff_curr_ts(deliver = True))
        action_lst_ret = []
        payoff_lst = []
        discount_lst = []
        atomic_payoff_lst = []
        state_action_advantage_lst = []
        curr_state_lst = []
        next_state_lst = []
        full_state_lst = []
        full_next_state_lst = []
        self.reset_x_infer(strict = strict, x_copy = None)
        for t in range(self.time_horizon):
            available_car_ids = markov_decision_process.get_available_car_ids(True)
            num_available_cars = len(available_car_ids)
            for car_idx in range(num_available_cars):
                car_id = available_car_ids[car_idx]
                dest, eta, battery = markov_decision_process.get_car_info(car_id)
                action_assigned = False
                travel_x_ids, charge_x_ids = self.get_relevant_x(t, dest, battery, eta, strict = strict)
                curr_state_counts = markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx])#.to(device = self.device)
#                if return_action:
                curr_state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                curr_payoff = markov_decision_process.get_payoff_curr_ts().clone()
                trip_requests_num_lst = markov_decision_process.get_num_trip_requests_region_lst(dest)
                sorted_region_lst = []
                for i in range(self.num_regions):
                    if trip_requests_num_lst[i] > 0:
                        sorted_region_lst = [i] + sorted_region_lst
                    else:
                        sorted_region_lst = sorted_region_lst + [i]
                if eta == 0:
                    for i in range(len(charge_x_ids)):
                        x_id = charge_x_ids[i]
                        if self.x_charge[x_id] > 0:
                            action_prob = min(self.x_charge[x_id], 1)
                            rv = np.random.binomial(n = 1, p = action_prob)
#                            rv = int(round(action_prob))
                            if rv == 1:
                                action_id = markov_decision_process.query_reduced_action(("charge", self.charging_rates[i]))
                                action = self.all_actions_reduced[action_id]
                                action_success = markov_decision_process.transit_within_timestamp(action, reduced = True, car_id = car_id)
                                if action_success:
                                    action_assigned = True
                                    self.x_charge[x_id] -= 1
                                    break
                if not action_assigned:
                    travel_x_ids, charge_x_ids = self.get_relevant_x(t + eta, dest, battery, 0, strict = strict)
                    if len(travel_x_ids) > 0:
                        for i in sorted_region_lst:
                            x_id = travel_x_ids[i]
                            if self.x_travel[x_id] > 0:
                                action_prob = min(self.x_travel[x_id], 1)
                                rv = np.random.binomial(n = 1, p = action_prob)
    #                            rv = int(round(action_prob))
                                if rv == 1:
                                    action_id = markov_decision_process.query_reduced_action(("travel", i % self.num_regions))
                                    action = self.all_actions_reduced[action_id]
                                    action_success = markov_decision_process.transit_within_timestamp(action, reduced = True, car_id = car_id)
                                    if action_success:
                                        action_assigned = True
                                        self.x_travel[x_id] -= 1
                                        break
#                if not action_assigned:
#                    travel_x_ids, charge_x_ids = self.get_relevant_x((t + eta) % self.adjusted_time_horizon, dest, battery, 0, strict = strict)
#                    travel_options = np.zeros(self.num_regions + 1)
#                    for i in range(len(travel_x_ids)):
#                        x_id = travel_x_ids[i]
#                        if self.x_travel[x_id] > 0:
#                            travel_options[i] += self.x_travel[x_id]
#                    while not action_assigned and np.sum(travel_options) > 0:
#                        if np.sum(travel_options[:-1]) < 1:
#                            travel_options[-1] = 1 - np.sum(travel_options[:-1])
#                        travel_probs = travel_options / np.sum(travel_options)
##                            action_prob = min(self.x_travel[x_id], 1)
##                            rv = np.random.binomial(n = 1, p = action_prob)
##                            rv = int(round(action_prob))
#                        action_num = np.random.choice(self.num_regions + 1, size = 1, p = travel_probs)[0]
#                        if action_num == self.num_regions:
#                            break
#                        x_id = travel_x_ids[action_num]
##                            if rv == 1:
##                                action_id = markov_decision_process.query_reduced_action(("travel", i % self.num_regions))
#                        action_id = markov_decision_process.query_reduced_action(("travel", action_num))
#                        action = self.all_actions_reduced[action_id]
#                        action_success = markov_decision_process.transit_within_timestamp(action, reduced = True, car_id = car_id)
#                        if action_success:
#                            action_assigned = True
#                            self.x_travel[x_id] -= 1
#                            travel_options[action_num] -= 1
#                            break
#                        else:
#                            travel_options[action_num] = 0
                if not action_assigned:
                    action_id = markov_decision_process.query_reduced_action(("nothing"))
                    action = self.all_actions_reduced[action_id]
                    markov_decision_process.transit_within_timestamp(action, reduced = True, car_id = car_id)
#                if car_idx == num_available_cars - 1:
#                    markov_decision_process.transit_across_timestamp()
                next_state_counts = markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx])#.to(device = self.device)
                full_next_state_counts = markov_decision_process.get_state_counts(deliver = True)
                payoff = markov_decision_process.get_payoff_curr_ts().clone()
                if return_data:
                    if car_idx == num_available_cars - 1:
                        next_t = t + 1
                    else:
                        next_t = t
                    state_action_advantage_lst.append((None, action_id, None, t, curr_payoff, next_t, payoff - curr_payoff, day_num))
                    curr_state_lst.append(curr_state_counts)
                    next_state_lst.append(next_state_counts)
                    full_state_lst.append(curr_state_counts_full)
                    full_next_state_lst.append(full_next_state_counts)
#                curr_state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                action_lst_ret.append((curr_state_counts_full, action, t, car_id))
            curr_state_counts_full = markov_decision_process.get_state_counts(deliver = True)
            action_lst_ret.append((curr_state_counts_full, None, t, None))
#            if num_available_cars == 0:
            markov_decision_process.transit_across_timestamp()
            curr_payoff = markov_decision_process.get_payoff_curr_ts(deliver = True)
            payoff_lst.append(curr_payoff)
            discount_lst.append(self.gamma ** t)
        discount_lst = torch.tensor(discount_lst)
        atomic_payoff_lst = torch.tensor([init_payoff] + payoff_lst)
        atomic_payoff_lst = atomic_payoff_lst[1:] - atomic_payoff_lst[:-1]
        discounted_payoff = torch.sum(atomic_payoff_lst * discount_lst)
        if return_data:
            final_payoff = float(markov_decision_process.get_payoff_curr_ts(deliver = True))
            return curr_state_lst, next_state_lst, full_state_lst, full_next_state_lst, state_action_advantage_lst, final_payoff, discounted_payoff, None, None
        passenger_carrying_cars = self.markov_decision_process.passenger_carrying_cars
        rerouting_cars = self.markov_decision_process.rerouting_cars
        idling_cars = self.markov_decision_process.idling_cars
        charging_cars = self.markov_decision_process.charging_cars
        payoff_lst = torch.tensor([init_payoff] + payoff_lst)
        return None, None, payoff_lst, action_lst_ret, discounted_payoff, passenger_carrying_cars, rerouting_cars, idling_cars, charging_cars

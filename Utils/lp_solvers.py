import sys
import gc
import math
import copy
import numpy as np
import pandas as pd
import torch
import cvxpy as cvx
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import Utils.train as train

class LP_Solver(train.Solver):
    def __init__(self, markov_decision_process = None, num_days = 1, gamma = 1):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process)
        self.trip_demand = self.markov_decision_process.trip_demand
        self.reward_df = self.markov_decision_process.reward_df
        self.num_days = num_days
        self.gamma = gamma
    
    def evaluate(self, **kargs):
        pass
    
    def train(self, A, b, c):
        m, n = A.shape
        A_cvx = cvx.Parameter(shape = (m, n), name = "A")
        b_cvx = cvx.Parameter(shape = (m,), name = "b")
        c_cvx = cvx.Parameter(shape = (n,), name = "c")
        x = cvx.Variable(shape = (n,), nonneg = True, name = "x")
        constraints = [A_cvx @ x == b_cvx]
        objective = cvx.Maximize(c_cvx @ x)
        problem = cvx.Problem(objective, constraints)
        problem.param_dict["A"].value = A
        problem.param_dict["b"].value = b
        problem.param_dict["c"].value = c
        obj_val = problem.solve()
        x = problem.var_dict["x"].value
        return x, obj_val

class LP_On_AugmentedGraph(LP_Solver):
    def __init__(self, markov_decision_process = None, num_days = 1, gamma = 1):
        super().__init__(markov_decision_process = markov_decision_process, num_days = num_days, gamma = gamma)
        self.load_data()
        self.construct_problem()
    
    ## Vectors to fetch:
    ##   - Trip reward r_ij^t
    ##   - Charging cost c^t
    ##   - Trip demand \lambda_ij^t
    ##   - Charging facility per region d_i^{\delta}
    ##   - Travel time \tau_ij^t
    ##   - Battery consumption b_ij
    ## Parameters to fetch:
    ##   - Total EVs N
    ##   - Time horizon T
    ##   - Number of battery levels B
    ##   - Number of regions R
    ##   - Number of charging rates \Delta
    def load_data(self):
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
    
    def construct_problem(self):
        self.construct_x()
        self.construct_obj()
        self.construct_constraints()
    
    def construct_x(self):
        pass
    
    def construct_obj(self):
        pass
    
    ## Flows add up to initial car distribution
    ## Flow conservation at each time, battery, and region
    ## Passenger-carrying flows not exceed trip demands
    ## Charging flows not exceed charging facility nums
    ## Infeasible flows equal to 0 (i.e. trips with insufficient battery)
    ## All flows add up to 1 at each time
    ## All flows being non-negative
    def construct_constraints(self):
        pass
    
    def evaluate(self):
        pass

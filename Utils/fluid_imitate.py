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
import Utils.lp_solvers as lp_solvers

class IL_Solver(train.Solver):
    def __init__(self, markov_decision_process = None, num_days = 1, gamma = 1, retrain = True, n_cpu = 1):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process)
        self.reward_df = self.markov_decision_process.reward_df
        self.num_days = num_days
        self.gamma = gamma
        self.retrain = retrain
        self.n_cpu = n_cpu
        self.lp_solver = lp_solvers.LP_On_AugmentedGraph(markov_decision_process = self.markov_decision_process, num_days = self.num_days, gamma = self.gamma, retrain = self.retrain, n_cpu = self.n_cpu)
    
    ## Return a distribution of car actions
    def query_fluid_resolve(self, state_counts, curr_ts):
        ## Re-train the fluid LP given state_counts and curr_ts
        self.lp_solver.set_start_time(curr_ts)
        self.lp_solver.set_lp_option(state_counts = state_counts)
        self.lp_solver.construct_problem()
        self.lp_solver.train()
        lp_x = self.lp_solver.x
        ## Extract the policy from the first time-step of the model
        pass
    
    ## Get an atomic action for each category (region, battery, eta) of cars
    ## State: global state + car category
    def collect_data_single(self, n_traj = 1):
        data_dict = {}
        for t in range(self.time_horizon):
            data_dict[t] = {"state_counts": [], "atomic_actions": []}
        markov_decision_process = self.markov_decision_process
        for traj in tqdm(range(n_traj)):
            ## Simulate trajectory
            ## Query fluid resolve to get joint actions
            markov_decision_process.reset_states(new_episode = True, seed = None)
            for t in range(self.time_horizon):
                state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                fluid_policy_dct = self.query_fluid_resolve(state_counts_full, t)
                available_car_ids = markov_decision_process.get_available_car_ids(self.state_reduction)
                num_available_cars = len(available_car_ids)
                for car_idx in range(num_available_cars):
                    curr_state_counts = markov_decision_process.get_state_counts(state_reduction = True, car_id = available_car_ids[car_idx])
                    car = markov_decision_process.state_dict[available_car_ids[car_idx]]
                    car_dest, car_eta, car_battery = car.get_dest(), car.get_time_to_dest(), car.get_battery()
                    ## Extract atomic action from the dct
            pass

    ## Train a neural network classifier for atomic actions given a state and car category
    def train(self):
        pass

    def evaluate(self, return_action = True, seed = None, day_num = 0):
        pass

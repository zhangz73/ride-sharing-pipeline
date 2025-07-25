import sys
import gc
import math
import copy
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import Utils.neural as neural

## This module computes different types of losses and performance metrics
class MetricFactory:
    def __init__(self):
        pass
    
    def get_surrogate_loss(self, total_payoff_lst):
        pass

    def get_total_payoff(self, total_payoff_lst):
        return torch.mean(total_payoff_lst)
    
    def get_total_payoff_loss(self, total_payoff_lst):
        return -torch.mean(total_payoff_lst)

## This module implements different types of solvers to the MDP problem
## Solvers:
##      1. Deep Reinforcement Learning
##      2. Dynamic Programming
##      3. Greedy
## Functionalities:
##      1. Construct solvers
##      2. Train solvers
##      3. Generate actions given states
class Solver:
    def __init__(self, type = "sequential", markov_decision_process = None, state_reduction = False, num_days = 1, useful_days = 1, gamma = 1):
        assert type in ["sequential", "group"]
        self.type = type
        self.num_days = num_days
        self.useful_days = useful_days
        self.gamma = gamma
        self.markov_decision_process = markov_decision_process
        self.state_reduction = state_reduction
        ## Save some commonly used variables from MDP
        self.time_horizon = self.markov_decision_process.time_horizon
#        self.reward_query = self.markov_decision_process.reward_query
        self.all_actions = self.markov_decision_process.get_all_actions(state_reduction = self.state_reduction)
        self.action_lst = [self.all_actions[k] for k in self.all_actions.keys()]
#        self.payoff_map = self.markov_decision_process.payoff_map
#        self.payoff_schedule_dct = self.markov_decision_process.payoff_schedule_dct
    
    def train(self, **kargs):
        return None

    ## Sequential Solvers: return an atomic action
    ## Group Solvers: return a list of actions and a list of corresponding car_type ids
    def predict(self, state_counts):
        return None

class PPO_Solver(Solver):
    def __init__(self, markov_decision_process = None, value_model_name = "discretized_feedforward", value_hidden_dim_lst = [10, 10], value_activation_lst = ["relu", "relu"], value_batch_norm = False, value_lr = 1e-2, value_epoch = 1, value_batch = 100, value_decay = 0.1, value_scheduler_step = 10000, value_solver = "Adam", value_retrain = False, policy_model_name = "discretized_feedforward", policy_hidden_dim_lst = [10, 10], policy_activation_lst = ["relu", "relu"], policy_batch_norm = False, policy_lr = 1e-2, policy_epoch = 1, policy_batch = 100, policy_decay = 0.1, policy_scheduler_step = 10000, policy_solver = "Adam", policy_retrain = False, descriptor = "PPO", dir = ".", device = "cpu", ts_per_network = 1, embedding_dim = 1, num_itr = 100, num_episodes = 100, car_batch = None, normalize_input = False, network_horizon_repeat = 1, num_days = 1, useful_days = 1, gamma = 1, eval_days = 1, use_region = False, ckpt_freq = 100, benchmarking_policy = "uniform", eps = 0.2, eps_sched = 1000, eps_eta = 0.5, policy_syncing_freq = 1, value_syncing_freq = 1, n_cpu = 1, n_threads = 4, lazy_removal = False, state_reduction = False, remove_trip = False):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process, state_reduction = state_reduction)
        ## Store some commonly used variables
        self.value_input_dim = self.markov_decision_process.get_state_len(state_reduction = state_reduction, model = "value", remove_trip = remove_trip)
        self.policy_input_dim = self.markov_decision_process.get_state_len(state_reduction = state_reduction, model = "policy", use_region = use_region, remove_trip = remove_trip)
        self.use_region = use_region
        self.value_output_dim = 1
        self.policy_output_dim = len(self.action_lst)
        self.network_horizon_repeat = network_horizon_repeat
        self.ts_per_network = ts_per_network
        self.discretized_len = int(math.ceil(self.time_horizon / self.ts_per_network)) * self.network_horizon_repeat
        self.num_embeddings = ts_per_network
        self.embedding_dim = embedding_dim
        self.use_embedding = self.ts_per_network > 1
        self.num_itr = num_itr
        self.num_episodes = num_episodes
        self.num_days = num_days
        self.useful_days = useful_days
        self.gamma = gamma
        self.use_avg_value = False #(self.num_days > 1) and (self.gamma == 1)
        self.eval_days = eval_days
        self.num_cars = self.markov_decision_process.num_total_cars
        self.car_batch = car_batch
        if self.car_batch is None:
            self.car_batch = markov_decision_process.num_total_cars - 1
        self.normalize_input = normalize_input
        self.ckpt_freq = ckpt_freq
        self.value_epoch = value_epoch
        self.value_retrain = value_retrain
        self.policy_retrain = policy_retrain
        self.policy_epoch = policy_epoch
        self.value_batch = value_batch #min(value_batch, num_episodes)
        self.policy_batch = policy_batch #min(policy_batch, num_episodes)
        self.benchmarking_policy = benchmarking_policy
        self.eps = eps
        self.eps_sched = eps_sched
        self.eps_eta = eps_eta
        self.policy_syncing_freq = policy_syncing_freq
        self.value_syncing_freq = value_syncing_freq
        self.device = device
#        self.payoff_map = self.payoff_map.to(device = self.device)
        self.one_minus_eps = torch.tensor(1 - eps).to(device = self.device)
        self.one_plus_eps = torch.tensor(1 + eps).to(device = self.device)
        self.n_cpu = n_cpu
        self.lazy_removal = lazy_removal
        self.state_reduction = state_reduction
        self.remove_trip = remove_trip
        ## Construct models
        self.value_model_factory = neural.ModelFactory(value_model_name, self.value_input_dim, value_hidden_dim_lst, value_activation_lst, self.value_output_dim, value_batch_norm, value_lr, value_decay, value_scheduler_step, value_solver, value_retrain, self.discretized_len, descriptor + "_value", dir, device, prob = False, use_embedding = self.use_embedding, num_embeddings = self.num_embeddings, embedding_dim = self.embedding_dim, ts_per_network = self.ts_per_network)
        self.policy_model_factory = neural.ModelFactory(policy_model_name, self.policy_input_dim, policy_hidden_dim_lst, policy_activation_lst, self.policy_output_dim, policy_batch_norm, policy_lr, policy_decay, policy_scheduler_step, policy_solver, policy_retrain, self.discretized_len, descriptor + "_policy", dir, device, prob = True, use_embedding = self.use_embedding, num_embeddings = self.num_embeddings, embedding_dim = self.embedding_dim, ts_per_network = self.ts_per_network)
        self.value_model = self.get_value_model()
        self.policy_model = self.get_policy_model()
        self.benchmark_policy_model = copy.deepcopy(self.policy_model)
        self.benchmark_value_model = copy.deepcopy(self.value_model)
        self.value_optimizer, self.value_scheduler = self.value_model_factory.prepare_optimizer()
        self.policy_optimizer, self.policy_scheduler = self.policy_model_factory.prepare_optimizer()
        self.markov_decision_process_pg = copy.deepcopy(markov_decision_process)
        self.markov_decision_process_lst = []
        for i in range(1): #self.n_cpu
            cp = copy.deepcopy(markov_decision_process)
            self.markov_decision_process_lst.append(cp)
        self.value_scale = self.value_model_factory.get_value_scale()
        self.input_scale = self.policy_model_factory.get_input_scale()
    
    def get_value_model(self):
        return self.value_model_factory.get_model()
    
    def get_policy_model(self):
        return self.policy_model_factory.get_model()
    
    def get_offset(self, day_num):
        return min(day_num, self.network_horizon_repeat - 1)
    
    def scale_input(self, state_counts, ts, type = "value"):
        assert type in ["value", "policy"]
        if ts not in self.input_scale:
            input_scale_mean, input_scale_std = torch.zeros(self.policy_input_dim), torch.ones(self.policy_input_dim)
        else:
            input_scale_mean, input_scale_std = self.input_scale[ts]["mu"], self.input_scale[ts]["std"]
        if type == "value":
            input_scale_mean, input_scale_std = input_scale_mean[:self.value_input_dim], input_scale_std[:self.value_input_dim]
        return (state_counts - input_scale_mean) / input_scale_std
    
    def get_advantage(self, curr_state_counts, next_state_counts, action_id, ts, next_ts, payoff, day_num = 0):
#        payoff = self.payoff_map[ts, action_id]
        offset = self.get_offset(day_num) * self.time_horizon
        if len(curr_state_counts.shape) > 1:
            curr, next = curr_state_counts[:,:self.value_input_dim], next_state_counts[:,:self.value_input_dim]
#            curr, next = curr_state_counts, next_state_counts
        else:
            curr, next = curr_state_counts[:self.value_input_dim], next_state_counts[:self.value_input_dim]
#            curr, next = curr_state_counts, next_state_counts
        if self.normalize_input:
            curr, next = self.scale_input(curr, ts, "value"), self.scale_input(next, next_ts, "value")
        with torch.no_grad():
            curr_value = self.value_model((ts + offset, curr)).reshape((-1,))
        mu, sd = self.value_scale[ts + offset]
        curr_value = curr_value * sd + mu
        if next_ts < self.time_horizon - 1:
            with torch.no_grad():
                next_value = self.value_model((next_ts + offset, next)).reshape((-1,))
            mu2, sd2 = self.value_scale[next_ts + offset]
            next_value = next_value * sd2 + mu2
        else:
            #next_value = 0
            if self.num_days > 1:
                next_value = self.value_model((0 + offset, next)).reshape((-1,))
                mu2, sd2 = self.value_scale[0 + offset]
                next_value = next_value * sd2 + mu2
            else:
                next_value = 0
        if self.use_avg_value:
            cum_ts = self.time_horizon * self.num_days - (day_num * self.time_horizon + ts)
            next_cum_ts = self.time_horizon * self.num_days - (day_num * self.time_horizon + next_ts)
            return (payoff + next_value * next_cum_ts - curr_value * cum_ts) / cum_ts / sd #((next_value - curr_value) * (self.time_horizon * self.num_days) + payoff) / self.time_horizon / self.num_days / sd
        return (payoff + next_value * self.gamma - curr_value) / sd
    
    def get_ratio(self, state_counts, action_id, ts, clipped = False, eps = 0.2, car_id = None, day_num = 0):
        offset = self.get_offset(day_num) * self.time_horizon
        prob_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = False, car_id = car_id, day_num = day_num)
        action_id = action_id.reshape((len(action_id), 1))
        prob = prob_output.gather(1, action_id) #prob_output[action_id]
        prob_benchmark_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = False, use_benchmark = True, car_id = car_id, day_num = day_num)
#         prob_benchmark_output = self.policy_benchmark_predict(state_counts, ts, prob = True, remove_infeasible = False)
        prob_benchmark = prob_benchmark_output.gather(1, action_id) #prob_benchmark_output[action_id]
        prob = prob.reshape((-1,))
        prob_benchmark = prob_benchmark.reshape((-1,))
        ratio = prob / prob_benchmark
        if clipped:
            clipped_ratio = torch.min(torch.max(ratio, torch.tensor(1 - eps)), torch.tensor(1 + eps))
        else:
            clipped_ratio = ratio
#        if prob > 0:
#            ratio = prob / prob_benchmark
#            if clipped:
#                clipped_ratio = torch.min(torch.max(ratio, torch.tensor(1 - eps)), torch.tensor(1 + eps))
#            else:
#                clipped_ratio = ratio
#        else:
#            ratio, clipped_ratio = 0
        return ratio, clipped_ratio
    
    ## Deprecated
    def get_ratio_single(self, state_counts, action_id, ts, clipped = False, eps = 0.2, car_id = None, state_count_check = None):
        prob_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = True, car_id = car_id, state_count_check = state_count_check)
        prob = prob_output[action_id]
        prob_benchmark_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = True, use_benchmark = True, car_id = car_id, state_count_check = state_count_check)
#         prob_benchmark_output = self.policy_benchmark_predict(state_counts, ts, prob = True, remove_infeasible = False)
        prob_benchmark = prob_benchmark_output[action_id]
        if prob > 0:
            ratio = prob / prob_benchmark
            if clipped:
                clipped_ratio = torch.min(torch.max(ratio, self.one_minus_eps), self.one_plus_eps)
            else:
                clipped_ratio = ratio
        else:
            ratio, clipped_ratio = 0, 0
        return ratio, clipped_ratio
    
    ## Deprecated
    def get_value_loss_single(self, state_action_advantage_lst_episodes):
        total_value_loss = 0
        val_num = 0
        for day in range(self.value_batch):
            state_num = len(state_action_advantage_lst_episodes[day])
            payoff = 0
            if state_num > 0:
                final_payoff = state_action_advantage_lst_episodes[day][state_num - 1][4].clone()
            for i in range(state_num - 1, -1, -1):
                tup = state_action_advantage_lst_episodes[day][i]
                curr_state_counts, action_id, _, t, curr_payoff, _ = tup
                payoff = final_payoff - curr_payoff
                value_model_output = self.value_model((t, curr_state_counts))
                total_value_loss += (value_model_output - payoff) ** 2 / self.num_cars / self.time_horizon
                val_num += 1
        total_value_loss /= val_num #self.num_episodes
        return total_value_loss
    
    def get_value_loss(self, value_dct, update_value_scale = False):
        total_value_loss = 0
        val_num = 0
        for t in range(self.time_horizon):
            if "payoff" in value_dct[t]:
                val_num += len(value_dct[t]["payoff"])
        for t in range(self.time_horizon):
            if "payoff" in value_dct[t]:
                payoff_lst = value_dct[t]["payoff"].to(device = self.device)
                if update_value_scale:
                    if not self.use_avg_value:
                        mu, sd = 0, 1 #torch.mean(payoff_lst), torch.std(payoff_lst) + 1e-3 #0, torch.std(payoff_lst) + 1e-3 #torch.mean(payoff_lst), torch.std(payoff_lst) + 1 #(self.time_horizon - t)
                    else:
                        mu, sd = 0, 1
                    self.value_scale[t] = (mu, sd)
                mu, sd = self.value_scale[t]
                payoff_lst = (payoff_lst - mu) / sd
                if len(value_dct[t]["state_counts"]) > 0:
                    state_counts_lst = value_dct[t]["state_counts"]
                    if update_value_scale:
                        input_norm_mean = torch.mean(state_counts_lst, dim = 0)
                        input_norm_std = torch.std(state_counts_lst, dim = 0)
                        input_scale_mean = input_norm_mean
                        input_scale_std = torch.max(input_norm_std, torch.tensor(1e-3))
                        self.input_scale[t] = {"mu": input_scale_mean, "std": input_scale_std}
                    state_counts_lst = state_counts_lst[:,:self.value_input_dim]
    #                state_counts_lst = torch.vstack(value_dct[t]["state_counts"]).to_dense()
                    if self.normalize_input:
                        state_counts_lst = self.scale_input(state_counts_lst, t, "value")
                    value_model_output = self.value_model((t, state_counts_lst)).reshape((-1,))
                    total_value_loss += torch.sum((value_model_output - payoff_lst) ** 2) / val_num #/ self.num_cars / self.time_horizon
#        total_value_loss /= val_num #self.num_episodes
        value_dct = None
        if update_value_scale:
            self.value_model_factory.set_value_scale(self.value_scale)
            self.policy_model_factory.set_input_scale(self.input_scale)
        return total_value_loss
    
    def get_data_single(self, num_episodes, episode_start = 0, worker_num = 0):
        state_action_advantage_lst_episodes = []
        total_payoff = 0
        single_day_payoffs = np.zeros(self.num_days)
        single_day_payoffs_raw = np.zeros((num_episodes, self.num_days))
        single_day_total_revenue = np.zeros((num_episodes, self.num_days))
        data_traj = {}
        for episode in tqdm(range(num_episodes)):
            tmp = []
            payoff_prev = 0
            data_traj[episode_start + episode] = {}
            curr_state_lst = []
            next_state_lst = []
            for t in range(self.time_horizon * self.network_horizon_repeat):
                data_traj[episode_start + episode][t] = {"state_counts": deque([]), "next_state_counts": deque([]), "payoff": deque([]), "atomic_payoff": deque([]), "action_id": deque([]), "ts": deque([]), "next_ts": deque([]), "day_num": deque([]), "term_val": deque([])}
            for day in range(self.num_days):
                curr_state_lst_single, next_state_lst_single, state_action_advantage_lst, payoff_val, discounted_payoff, payoff_raw, total_revenue = self.evaluate(train = True, return_data = True, debug = False, debug_dir = None, lazy_removal = self.lazy_removal, markov_decision_process = self.markov_decision_process_lst[0], day_num = day)
                with torch.no_grad():
                    curr_term_val = 0#self.value_model((0, next_state_lst[-1][:self.value_input_dim]))
                tmp += state_action_advantage_lst
                total_payoff += discounted_payoff * self.gamma ** (self.time_horizon * day) #discounted_payoff / self.num_days #payoff_val / self.num_days
#                total_payoff += payoff_val * self.gamma ** day
                single_day_payoffs[day] += payoff_val - payoff_prev
                payoff_prev = payoff_val
                single_day_payoffs_raw[episode, day] = payoff_raw
                single_day_total_revenue[episode, day] = total_revenue
                curr_state_lst += curr_state_lst_single
                next_state_lst += next_state_lst_single
            ## Indent starts
            state_num = len(tmp)
            ## Collect trajectory data
            payoff = 0
            if state_num > 0:
                final_payoff = tmp[-1][4].clone()
            curr_t = self.time_horizon
            curr_day = self.num_days
            cum_ts = 0
            cum_steps = 0
            total_ts = self.num_days * self.time_horizon
            record_num = 0
            for i in range(state_num - 1, -1, -1):
                tup = tmp[i]
                curr_state_counts = curr_state_lst[i]
                next_state_counts = next_state_lst[i]
                _, action_id, _, t, curr_payoff, next_t, atomic_payoff, day_num, rev_atomic_step = tup
                offset = self.get_offset(day_num) * self.time_horizon
                if self.use_avg_value:
                    next_cum_ts = total_ts - (day_num * self.time_horizon + t)
                    next_cum_steps = (next_cum_ts - 1) * self.num_cars + rev_atomic_step
                    payoff = (atomic_payoff + payoff * cum_steps) / next_cum_steps
                    cum_steps = next_cum_steps
                    cum_ts = next_cum_ts
                else:
                    if t != curr_t: #day_num != curr_day: #
                        payoff = atomic_payoff + self.gamma * payoff
                        curr_t = t
                    else:
                        payoff = atomic_payoff + payoff
                if day_num < self.useful_days and curr_state_counts is not None:
                    lens = len(curr_state_counts)
                    if next_state_counts is None and i < state_num - 1:
                        next_state_counts = tmp[i + 1][0]
                        assert next_state_counts is not None
                    data_traj[episode_start + episode][t + offset]["next_state_counts"].appendleft(next_state_counts.reshape((1, lens)))
                    data_traj[episode_start + episode][t + offset]["payoff"].appendleft(payoff)
                    data_traj[episode_start + episode][t + offset]["state_counts"].appendleft(curr_state_counts.reshape((1, lens)))
                    data_traj[episode_start + episode][t + offset]["action_id"].appendleft(action_id)
                    data_traj[episode_start + episode][t + offset]["ts"].appendleft(t)
                    data_traj[episode_start + episode][t + offset]["next_ts"].appendleft(next_t)
                    data_traj[episode_start + episode][t + offset]["day_num"].appendleft(day_num)
                    data_traj[episode_start + episode][t + offset]["atomic_payoff"].appendleft(atomic_payoff)
                    data_traj[episode_start + episode][t + offset]["term_val"].appendleft(curr_term_val)
                    record_num += 1
            ## Indent ends
            for t in range(self.time_horizon):
                if len(data_traj[episode_start + episode][t]["state_counts"]) > 0:
                    data_traj[episode_start + episode][t]["state_counts"] = torch.cat(list(data_traj[episode_start + episode][t]["state_counts"]), dim = 0)
                    data_traj[episode_start + episode][t]["next_state_counts"] = torch.cat(list(data_traj[episode_start + episode][t]["next_state_counts"]), dim = 0)
                    data_traj[episode_start + episode][t]["payoff"] = torch.tensor(data_traj[episode_start + episode][t]["payoff"])
                    data_traj[episode_start + episode][t]["atomic_payoff"] = torch.tensor(data_traj[episode_start + episode][t]["atomic_payoff"])
                    data_traj[episode_start + episode][t]["action_id"] = torch.tensor(data_traj[episode_start + episode][t]["action_id"])
                    data_traj[episode_start + episode][t]["ts"] = torch.tensor(data_traj[episode_start + episode][t]["ts"])
                    data_traj[episode_start + episode][t]["next_ts"] = torch.tensor(data_traj[episode_start + episode][t]["next_ts"])
                    data_traj[episode_start + episode][t]["day_num"] = torch.tensor(data_traj[episode_start + episode][t]["day_num"])
                    data_traj[episode_start + episode][t]["term_val"] = torch.tensor(data_traj[episode_start + episode][t]["term_val"])
                else:
                    data_traj[episode_start + episode][t]["state_counts"] = []
            ############
#            state_action_advantage_lst_episodes.append(tmp)
        norm_factor = torch.sum(self.gamma ** (self.time_horizon * torch.arange(self.num_days)))
#        norm_factor = torch.sum(self.gamma ** torch.arange(self.num_days))
        total_payoff /= norm_factor
        data_traj_combo = {}
        for t in range(self.time_horizon * self.network_horizon_repeat):
            data_traj_combo[t] = {"state_counts": [], "next_state_counts": [], "payoff": [], "atomic_payoff": [], "action_id": [], "ts": [], "next_ts": [], "day_num": [], "term_val": []}
            for episode in range(num_episodes):
                if len(data_traj[episode_start + episode][t]["state_counts"]) > 0:
                    for key in data_traj_combo[t]:
                        data_traj_combo[t][key] += [data_traj[episode_start + episode][t][key]]
            for key in data_traj_combo[t]:
                if len(data_traj_combo[t]["state_counts"]) > 0:
                    if key in ["state_counts", "next_state_counts"]:
                        data_traj_combo[t][key] = torch.cat(data_traj_combo[t][key], dim = 0)
                    else:
                        data_traj_combo[t][key] = torch.cat(data_traj_combo[t][key])
        return data_traj_combo, (num_episodes, total_payoff), single_day_payoffs, single_day_payoffs_raw, single_day_total_revenue
    
    def train(self, return_payoff = False, debug = False, debug_dir = "debugging_log.txt", label = ""):
        value_loss_arr = []
        policy_loss_arr = []
        payoff_arr = []
        self.value_model.train()
        self.policy_model.train()
        ## Policy Iteration
        dct_outer = {}
        report_factory = ReportFactory(markov_decision_process = self.markov_decision_process)
        with open(f"payoff_log_{label}.txt", "w") as f:
            f.write("Payoff Logs:\n")
        with open(f"payoff_log_multi_{label}.csv", "w") as f:
            f.write(f"{','.join(['Day_' + str(x) for x in range(self.num_days)])}\n")
        eps = self.eps
        with Parallel(n_jobs = self.n_cpu, max_nbytes = None) as parallel:
            for itr in range(self.num_itr + 0):
                print(f"Iteration #{itr+1}/{self.num_itr}:")
                ## Obtain simulated data from each episode
                data_traj_all = {}
                for t in range(self.time_horizon * self.network_horizon_repeat):
                    data_traj_all[t] = {"state_counts": [], "next_state_counts": [], "payoff": [], "atomic_payoff": [], "action_id": [], "ts": [], "next_ts": [], "day_num": [], "term_val": []}
                print("\tGathering data...")
                if self.n_cpu == 1:
                    data_traj_all, tup, single_day_payoffs, single_day_payoffs_raw, single_day_total_revenue = self.get_data_single(self.num_episodes)
                    payoff_val = float(tup[1] / tup[0])
                    single_day_payoffs = single_day_payoffs / tup[0]
                else:
                    batch_size = int(math.ceil(self.num_episodes / self.n_cpu))
                    results = parallel(delayed(
                        self.get_data_single
                    )(min((i + 1) * batch_size, self.num_episodes) - i * batch_size, i * batch_size, i) for i in range(self.n_cpu))
                    print("Gathering results...")
                    payoff_val = 0
                    single_day_payoffs = np.zeros(self.num_days)
                    single_day_payoffs_raw_lst = []
                    single_day_total_revenue_lst = []
                    for res in results:
#                        state_action_advantage_lst_episodes += res[0]
                        for t in range(self.time_horizon * self.network_horizon_repeat):
                            for key in data_traj_all[t]:
                                data_traj_all[t][key] += [res[0][t][key]]
                        tup = res[1]
                        payoff_val += tup[1]
                        single_day_payoffs += res[2]
                        single_day_payoffs_raw_lst.append(res[3])
                        single_day_total_revenue_lst.append(res[4])
                    for t in range(self.time_horizon * self.network_horizon_repeat):
                        if len(data_traj_all[t]["state_counts"]) > 0:
                            for key in data_traj_all[t]:
                                if key in ["state_counts", "next_state_counts"]:
                                    data_traj_all[t][key] = torch.cat(data_traj_all[t][key], dim = 0)
                                else:
                                    data_traj_all[t][key] = torch.cat(data_traj_all[t][key])
                    payoff_val /= self.num_episodes
                    single_day_payoffs /= self.num_episodes
                    single_day_payoffs_raw = np.vstack(single_day_payoffs_raw_lst)
                    single_day_total_revenue = np.vstack(single_day_total_revenue_lst)
                    results = None
                payoff_arr.append(payoff_val)
                with open(f"payoff_log_{label}.txt", "a") as f:
                    f.write(f"{float(payoff_val)}\n")
                with open(f"payoff_log_multi_{label}.csv", "a") as f:
                    f.write(f"{','.join([str(x) for x in list(single_day_payoffs)])}\n")
                ## Log detailed payoff and revenue per iteration
                np.save(f"Logs/single_day_payoff_raw_{label}_itr={itr+1}.npy", single_day_payoffs_raw)
                np.save(f"Logs/single_day_total_revenue_{label}_itr={itr+1}.npy", single_day_total_revenue)
                if itr == self.num_itr:
                    break
                g_est = payoff_val / self.time_horizon
                g_est_step = g_est / self.num_cars
                    
                ## Update models
                ## Update value models
                print("\tUpdating value models...")
                value_curr_arr = []
                for value_itr in tqdm(range(self.value_epoch)):
                    self.value_optimizer.zero_grad(set_to_none=True)
                    value_dct = {}
                    for t in range(self.time_horizon):
                        value_dct[t] = {}
                        if len(data_traj_all[t]["state_counts"]) > 0:
                            batch_idx = torch.from_numpy(np.random.choice(len(data_traj_all[t]["state_counts"]), size = min(self.value_batch, len(data_traj_all[t]["state_counts"])), replace = False))
                            value_dct[t]["state_counts"] = data_traj_all[t]["state_counts"][batch_idx,:]
                            term_vals = data_traj_all[t]["term_val"][batch_idx]
                            value_dct[t]["payoff"] = data_traj_all[t]["payoff"][batch_idx] - g_est_step #- (self.time_horizon - t) * g_est + term_vals
                    total_value_loss = self.get_value_loss(value_dct, update_value_scale = value_itr == 0 and self.value_retrain)
                    value_curr_arr.append(float(total_value_loss.data))
                    total_value_loss.backward()
                    self.value_optimizer.step()
                    self.value_scheduler.step()
                plt.plot(value_curr_arr)
                plt.title(f"Itr = {itr + 1}\nFinal Loss = {value_curr_arr[-1]}")
                plt.savefig(f"DebugPlots/value_itr={itr+1}.png")
                plt.clf()
                plt.close()
    #            if debug:
    #                with open(debug_dir, "a") as f:
    #                    f.write(f"\tFinal Value Loss = {float(total_value_loss.data)}\n")
                
                ## Update policy models
                print("\tUpdating policy models...")
                policy_curr_arr = []
                for policy_itr in tqdm(range(self.policy_epoch)):
                    policy_dct = {}
                    for t in range(self.time_horizon):
                        for offset in [0, 1]:
                            tup = (t, t + offset, 0)
                            policy_dct[tup] = {"curr_state_counts": [], "next_state_counts": [], "action_id": [], "atomic_payoff": []}
                            if len(data_traj_all[t]["state_counts"]) > 0:
                                relevant_idx = (data_traj_all[t]["next_ts"] == t + offset).nonzero(as_tuple = True)[0].numpy()
                            else:
                                relevant_idx = []
                            if len(relevant_idx) > 0:
                                batch_idx = torch.from_numpy(np.random.choice(relevant_idx, size = min(self.policy_batch, len(relevant_idx)), replace = False))
                                policy_dct[tup]["curr_state_counts"] = data_traj_all[t]["state_counts"][batch_idx,:]
                                policy_dct[tup]["next_state_counts"] = data_traj_all[t]["next_state_counts"][batch_idx,:]
                                policy_dct[tup]["action_id"] = data_traj_all[t]["action_id"][batch_idx]
                                policy_dct[tup]["atomic_payoff"] = data_traj_all[t]["atomic_payoff"][batch_idx]
                    total_policy_loss = 0
                    total_policy_num = 0
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    for day_num in range(1):
                        for t in range(self.time_horizon):
                            for offset in [0, 1]:
                                next_t = t + offset
                                if len(policy_dct[(t, next_t, day_num)]["curr_state_counts"]) > 0:
                                    curr_state_counts_lst = policy_dct[(t, next_t, day_num)]["curr_state_counts"]
                                    next_state_counts_lst = policy_dct[(t, next_t, day_num)]["next_state_counts"]
                                    action_id_lst = policy_dct[(t, next_t, day_num)]["action_id"].to(device = self.device)
                                    atomic_payoff_lst = policy_dct[(t, next_t, day_num)]["atomic_payoff"].to(device = self.device)
                                    advantage = self.get_advantage(curr_state_counts_lst, next_state_counts_lst, action_id_lst, t, next_t, atomic_payoff_lst, day_num = day_num)
#                                    if next_t > t:
#                                        advantage += g_est
                                    advantage -= g_est_step
                                    ## TODO: Fix it!!!
                                    ratio, ratio_clipped = self.get_ratio(curr_state_counts_lst, action_id_lst, t, clipped = True, eps = eps, day_num = day_num)
                                    loss_curr = -torch.min(ratio * advantage, ratio_clipped * advantage)
                                    total_policy_loss += torch.sum(loss_curr) #/ len(batch_idx)
                                    total_policy_num += len(loss_curr)
                    total_policy_loss /= total_policy_num
                    policy_curr_arr.append(float(total_policy_loss.data))
                    total_policy_loss.backward()
                    self.policy_optimizer.step()
                    self.policy_scheduler.step()
                    policy_dct = None
                plt.plot(policy_curr_arr)
                plt.title(f"Itr = {itr + 1}\nFinal Loss = {policy_curr_arr[-1]}")
                plt.savefig(f"DebugPlots/policy_itr={itr+1}.png")
                plt.clf()
                plt.close()
                if itr % self.value_syncing_freq == 0:
                    self.benchmark_value_model = copy.deepcopy(self.value_model)
                if itr % self.policy_syncing_freq == 0:
                    self.benchmark_policy_model = copy.deepcopy(self.policy_model)
                ## Save loss data
                value_loss_arr.append(float(total_value_loss.data))
                policy_loss_arr.append(float(total_policy_loss.data))
                
                ## Update eps according to scheduler
                if itr > 0 and itr % self.eps_sched == 0:
                    eps = max(eps * self.eps_eta, 0.01)
                ## Checkpoint
                if itr > 0 and itr % self.ckpt_freq == 0:
                    self.value_model_factory.update_model(self.value_model, update_ts = False)
                    self.policy_model_factory.update_model(self.policy_model, update_ts = False)
                    descriptor = f"_itr={itr}"
                    self.value_model_factory.save_to_file(descriptor, include_ts = True)
                    self.policy_model_factory.save_to_file(descriptor, include_ts = True)
                    num_trials = 10
                    df_table_all, payoff = self.evaluate_batch(num_trials, self.eval_days, seed_lst = None, n_cpu = self.n_cpu, parallel = parallel)
                    df_table_all = df_table_all.groupby(["t"]).mean().reset_index()
                    report_factory.visualize_table(df_table_all, f"{label}_itr={itr}", title = f"Total Payoff: {payoff:.2f}", detailed = True)
        self.benchmark_policy_model = copy.deepcopy(self.policy_model)
        self.benchmark_value_model = copy.deepcopy(self.value_model)
        ## Save final model
        self.value_model_factory.update_model(self.value_model, update_ts = False)
        self.policy_model_factory.update_model(self.benchmark_policy_model, update_ts = False)
        self.value_model_factory.save_to_file(include_ts = True)
        self.policy_model_factory.save_to_file(include_ts = True)
        return value_loss_arr, policy_loss_arr, payoff_arr

    def policy_describe(self, action_id_prob):
        ret = ""
        for action_id in range(len(action_id_prob)):
            prob = action_id_prob[action_id]
            action_msg = self.all_actions[action_id].describe()
            msg = f"{action_msg}: {prob}"
            ret += f"\t\t{msg}\n"
        return ret
    
    def policy_predict(self, state_counts, ts, prob = True, remove_infeasible = True, use_benchmark = False, lazy_removal = True, car_id = None, state_count_check = None, day_num = 0):
        offset = self.get_offset(day_num)
        if self.normalize_input:
            state_counts_input = self.scale_input(state_counts, ts, "policy")
        else:
            state_counts_input = state_counts
        if not use_benchmark:
            output = self.policy_model((ts + offset, state_counts_input))
        else:
            with torch.no_grad():
                output = self.benchmark_policy_model((ts + offset, state_counts_input))
        if remove_infeasible:
            if len(state_counts.shape) == 1:
                ret = self.remove_infeasible_actions(state_counts.cpu(), ts, output, car_id = car_id, state_count_check = state_count_check)
            else:
                ret_lst = []
                for i in range(state_counts.shape[0]):
                    ret = self.remove_infeasible_actions(state_counts[i,:].cpu(), ts, output[i,:], lazy_removal = lazy_removal, car_id = car_id, state_count_check = state_count_check)
                    ret_lst.append(ret.reshape((1, len(ret))))
                ret = torch.cat(ret_lst, dim = 0)
            ret = ret.to(device = self.device)
            output = output * ret
        if torch.sum(output) == 0:
            return None
        output = output / torch.sum(output)
        if prob:
            return output
        return torch.argmax(output, dim = 1).unsqueeze(-1)
    
    def remove_infeasible_actions(self, state_counts, ts, output, lazy_removal = True, car_id = None, state_count_check = None):
        ## Eliminate infeasible actions
        ret = torch.ones(len(output))
        if self.state_reduction:
            mask = self.markov_decision_process.state_counts_to_potential_feasible_actions(self.state_reduction, state_counts = state_count_check, car_id = car_id)
        else:
            mask = self.markov_decision_process.state_counts_to_potential_feasible_actions(self.state_reduction, state_counts = state_count_check)
        ret = ret * mask
        if not lazy_removal and self.state_reduction:
            potential_feasible_action_ids = torch.where(ret > 0)[0]
            for action_id in potential_feasible_action_ids:
                action_id = int(action_id)
                if not self.action_is_feasible(state_counts, ts, action_id, car_id = car_id, state_count_check = state_count_check):
                    ret[action_id] = 0
        return ret
    
    def action_is_feasible(self, state_counts, ts, action_id, car_id = None, state_count_check = None):
        return self.markov_decision_process.action_is_potentially_feasible(action_id, reduced = self.state_reduction, car_id = car_id, state_counts = state_count_check)
#        self.markov_decision_process_pg.set_states(state_count_check, ts)
#        action = self.all_actions[action_id]
#        return self.markov_decision_process_pg.transit_within_timestamp(action, reduced = self.state_reduction, car_id = car_id)
    
    ## Deprecated
    def policy_benchmark_predict(self, state_counts, ts, prob = True, remove_infeasible = True):
        if self.benchmarking_policy == "uniform":
            policy_output = torch.ones(self.policy_output_dim)
        else:
            ## TODO: Implement it!!!
            policy_output = torch.ones(self.policy_output_dim)
        if remove_infeasible:
            ret = self.remove_infeasible_actions(state_counts, ts, policy_output)
            policy_output = policy_output * ret
        if torch.sum(policy_output) == 0:
            return policy_output
        policy_output = policy_output / torch.sum(policy_output)
        if prob:
            return None
        return torch.argmax(policy_output)
    
    def evaluate_batch(self, num_trials, eval_days, seed_lst = None, n_cpu = 1, parallel = None):
        df_table_all = None
        batch_size = int(math.ceil(num_trials / n_cpu))
        if seed_lst is None:
            seed_lst = [None] * num_trials
        if parallel is None:
            parallel = Parallel(n_jobs = n_cpu)
        res = parallel(delayed(self.evaluate_batch_single)(
            min(num_trials, (i + 1) * batch_size) - i * batch_size, eval_days, seed_lst[(i * batch_size):min(num_trials, (i + 1) * batch_size)]
        ) for i in range(n_cpu))
        payoff = 0
        for df_table, payoff_single in res:
            if df_table_all is None:
                df_table_all = df_table
            else:
                df_table_all = pd.concat([df_table_all, df_table], axis = 0)
            payoff += payoff_single
        payoff /= num_trials
        return df_table_all, payoff
    
    def evaluate_batch_single(self, num_trials, eval_days, seed_lst = None):
        df_table_all = None
        report_factory = ReportFactory(markov_decision_process = self.markov_decision_process)
        norm_factor = eval_days #torch.sum(self.gamma ** (self.time_horizon * torch.arange(self.eval_days)))
        payoff = 0
        for i in tqdm(range(num_trials)):
            for day in range(self.eval_days):
                if seed_lst is not None:
                    seed = seed_lst[i]
                else:
                    seed = None
                _, _, payoff_lst, action_lst, discounted_payoff, passenger_carrying_cars, rerouting_cars, idling_cars, charging_cars = self.evaluate(return_action = True, seed = seed, day_num = day)
                if len(payoff_lst) > 0:
                    payoff += float(payoff_lst[-1].data - payoff_lst[0].data) / norm_factor
                df_table = report_factory.get_table(self.markov_decision_process, action_lst, passenger_carrying_cars, rerouting_cars, idling_cars, charging_cars, detailed = True)
                df_table["trial"] = i
                df_table["t"] += self.time_horizon * day
                if df_table_all is None:
                    df_table_all = df_table
                else:
                    df_table_all = pd.concat([df_table_all, df_table], axis = 0)
        return df_table_all, payoff
    
    def evaluate(self, seed = None, train = False, return_data = False, return_action = False, debug = False, debug_dir = "debugging_log.txt", lazy_removal = False, markov_decision_process = None, day_num = 0, log_policy = False, policy_log_dir = "PolicyLogs/policy_log.csv"):
        if True: #not train:
            self.value_model.eval()
            self.benchmark_policy_model.eval()
        #if markov_decision_process is None:
        markov_decision_process = self.markov_decision_process
        return_data = return_data and train
        if seed is not None:
            torch.manual_seed(seed)
        markov_decision_process.reset_states(new_episode = day_num == 0, seed = seed)
        payoff_lst = []
        atomic_payoff_lst = []
        discount_lst = []
        model_value_lst = []
        action_lst = []
        policy_loss = 0
        state_action_advantage_lst = []
        curr_state_lst = []
        next_state_lst = []
        total_cars = 0
        total_trips = 0
        curr_payoff = float(markov_decision_process.get_payoff_curr_ts(deliver = True))
        payoff_lst.append(curr_payoff)
        total_revenue = markov_decision_process.get_total_market_revenue()
        payoff_begin_raw = float(markov_decision_process.get_payoff_curr_ts(deliver = False))
        if log_policy:
            with open(policy_log_dir, "w") as f:
                f.write("t,car_dest,car_eta,car_battery,action_type,action_info\n")
            trip_requests_realized = markov_decision_process.trip_arrivals.numpy()
            df_trips = pd.DataFrame(trip_requests_realized)
            df_trips.to_csv("PolicyLogs/trip_requests_realized.csv", index = False)
        for t in tqdm(range(self.time_horizon), leave = False):
            ## Add up ||v_model - v_hat||^2
            ## Add up ratio * advantage
            available_car_ids = markov_decision_process.get_available_car_ids(self.state_reduction)
            num_available_cars = len(available_car_ids)
            total_cars += num_available_cars
#            if not train:
#                print("t = ", t)
#                print(self.markov_decision_process.describe_state_counts())
            transit_applied = False
            curr_car_batch = min(self.car_batch, num_available_cars - 1)
            if curr_car_batch > 0:
                selected_idx_for_state_data = set(np.random.choice(num_available_cars - 1, size = curr_car_batch, replace = False))
            else:
                selected_idx_for_state_data = set([])
            rev_atomic_step = self.num_cars
            for car_idx in tqdm(range(num_available_cars), leave = False):
                ## Perform state transitions
                curr_state_counts = markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx], remove_trip = self.remove_trip)#.to(device = self.device)
#                if return_action:
                curr_state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                curr_state_counts = curr_state_counts.view((1, len(curr_state_counts)))
                action_id_prob = self.policy_predict(curr_state_counts, t, prob = True, use_benchmark = True, lazy_removal = lazy_removal, car_id = available_car_ids[car_idx], state_count_check = curr_state_counts_full, day_num = day_num)
                action_id_prob = action_id_prob.flatten()
                curr_state_counts = curr_state_counts.flatten()
                if action_id_prob is not None:
                    action_id_prob = action_id_prob.cpu().detach().numpy()
                    if debug:
                        with open(debug_dir, "a") as f:
                            msg = self.policy_describe(action_id_prob)
                            payoff = markov_decision_process.get_payoff_curr_ts().clone()
                            self.value_model.eval()
#                            inferred_value = float(self.value_model((t, (curr_state_counts - self.input_scale_mean_value) / self.input_scale_std_value)).data)
                            f.write(f"\tt = {t}, car_id = {car_idx}, payoff = {payoff}, inferred state value = {inferred_value}:\n")
                            f.write(self.markov_decision_process.describe_state_counts(curr_state_counts))
                            f.write(msg)
                    if True:#train:
                        is_feasible = False
                        while not is_feasible:
                            action_id = np.random.choice(len(action_id_prob), p = action_id_prob)
                            is_feasible = self.action_is_feasible(curr_state_counts, t, int(action_id), car_id = available_car_ids[car_idx], state_count_check = curr_state_counts_full)
                            if not is_feasible:
                                print(self.all_actions[action_id].describe(), action_id)
                                print(self.all_actions[action_id].get_type())
                                print(self.markov_decision_process.describe_state_counts())
                                assert False
                    else:
                        action_id = np.argmax(action_id_prob)
                    action = self.all_actions[int(action_id)]
                    if log_policy:
                        car = markov_decision_process.state_dict[available_car_ids[car_idx]]
                        car_dest, car_eta, car_battery = car.get_dest(), car.get_time_to_dest(), car.get_battery()
                        action_type = action.get_type()
                        if action_type in ["pickup", "rerouting", "idling"]:
                            action_info = action.get_dest()
                        elif action_type == "charged":
                            action_info = action.get_rate()
                        else:
                            action_info = "n/a"
                        with open(policy_log_dir, "a") as f:
                            f.write(f"{t},{car_dest},{car_eta},{car_battery},{action_type},{action_info}\n")
                    if action.get_type() in ["pickup", "rerouting"]:
                        total_trips += 1
                    if return_action:
                        action_lst.append((curr_state_counts_full, action, t, car_idx))
                    curr_payoff = markov_decision_process.get_payoff_curr_ts().clone().to(device = self.device)
                    res = markov_decision_process.transit_within_timestamp(action, self.state_reduction, available_car_ids[car_idx])
                    next_t = t
                    if car_idx == num_available_cars - 1:
                        transit_applied = True
                        if return_action:
                            curr_state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                            action_lst.append((curr_state_counts_full, None, t, None))
                        markov_decision_process.transit_across_timestamp()
                        next_t += 1
                    if True: #t == self.time_horizon - 1 and car_idx == num_available_cars - 1:
                        next_state_counts = markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx], remove_trip = self.remove_trip)#.to(device = self.device)
                    else:
                        next_state_counts = None
                    payoff = markov_decision_process.get_payoff_curr_ts().clone()
                    if return_data: # and t < self.time_horizon - 1:
                        if day_num >= self.useful_days:
                            curr_state_counts, next_state_counts = None, None
                        if car_idx < num_available_cars - 1 and car_idx not in selected_idx_for_state_data:
                            curr_state_counts = None
                        if curr_state_counts is None:
                            next_state_counts is None
#                        else:
#                            if (car_idx - 1) in selected_idx_for_state_data:
#                                tup = state_action_advantage_lst[-1]
#                                tup_new = (tup[0], tup[1], None, tup[3], tup[4], tup[5], tup[6], tup[7])
#                                state_action_advantage_lst[-1] = tup_new
                        state_action_advantage_lst.append((None, action_id, None, t, curr_payoff, next_t, payoff - curr_payoff, day_num, rev_atomic_step))
                        curr_state_lst.append(curr_state_counts)
                        next_state_lst.append(next_state_counts)
                    atomic_payoff_lst.append(payoff - curr_payoff)
                    discount_lst.append(self.gamma ** t)
                    ## Compute loss
                    if not return_data:
#                        if car_idx < num_available_cars - 1:
#                            next_t = t
#                        else:
#                            next_t = t + 1
#                        curr, next = curr_state_counts[:self.value_input_dim], next_state_counts[:self.value_input_dim]
#                        advantage = self.get_advantage(curr, next, action_id, t, next_t, payoff - curr_payoff)
#                        ratio, ratio_clipped = self.get_ratio_single(curr_state_counts, action_id, t, clipped = False, car_id = available_car_ids[car_idx], state_count_check = curr_state_counts_full)
#                        loss_curr = -torch.min(ratio * advantage, ratio_clipped * advantage)
#                        policy_loss += loss_curr[0]
#                        ## Compute values
                        curr_payoff = float(markov_decision_process.get_payoff_curr_ts(deliver = True))
#                        curr_value = self.value_model((t, curr))
                        payoff_lst.append(curr_payoff)
#                        if t == 0:
#                            if action.get_type() in ["pickup", "rerouting"]:
#                                print(action.get_origin(), action.get_dest(), curr_payoff)
#                        model_value_lst.append(curr_value)
                rev_atomic_step -= 1
            if not transit_applied:
                if return_action:
                    curr_state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                    action_lst.append((curr_state_counts_full, None, t, None))
                markov_decision_process.transit_across_timestamp()
        if not return_data:
            payoff_lst = torch.tensor(payoff_lst)
#            payoff_lst[-1] = 0
#            empirical_value_lst = torch.cumsum(payoff_lst.flip(0), 0).flip(0)
#            payoff_lst[-1] = payoff_lst[-2]
#            empirical_value_lst = payoff_lst
#            value_loss = 0
#            assert len(empirical_value_lst) == len(model_value_lst)
#            for t in range(len(model_value_lst) - 1, -1, -1):
#                value_loss += torch.sum((model_value_lst[t] - empirical_value_lst[t]) ** 2)
#            model_value_lst = torch.tensor(model_value_lst)
#            value_loss = torch.sum((model_value_lst - empirical_value_lst) ** 2)
        discount_lst = torch.tensor(discount_lst)
        atomic_payoff_lst = torch.tensor(atomic_payoff_lst)
        discounted_payoff = torch.sum(discount_lst * atomic_payoff_lst) / markov_decision_process.get_total_market_revenue()
        payoff_end_raw = float(markov_decision_process.get_payoff_curr_ts(deliver = False))
        payoff_raw = payoff_end_raw - payoff_begin_raw
        if return_data:
            final_payoff = float(markov_decision_process.get_payoff_curr_ts(deliver = True))
            return curr_state_lst, next_state_lst, state_action_advantage_lst, final_payoff, discounted_payoff, payoff_raw, total_revenue
        passenger_carrying_cars = markov_decision_process.passenger_carrying_cars
        rerouting_cars = markov_decision_process.rerouting_cars
        charging_cars = markov_decision_process.charging_cars
        idling_cars = markov_decision_process.idling_cars
#        print("total cars", total_cars)
#        print("total trips", total_trips)
#        print("payoff", float(payoff_lst[-1].data))
        #return value_loss.cpu(), policy_loss.cpu(), payoff_lst, action_lst
        return None, None, payoff_lst, action_lst, discounted_payoff, passenger_carrying_cars, rerouting_cars, idling_cars, charging_cars

class D_Closest_Car_Solver(Solver):
    def __init__(self, markov_decision_process = None, d = 1, num_days = 1, useful_days = 1, gamma = 1):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process)
        self.d = d
        self.num_days = num_days
        self.useful_days = useful_days
        self.gamma = gamma
    
    def evaluate(self, return_action = True, seed = None, day_num = 0, log_policy = False, policy_log_dir = "PolicyLogs/policy_log.csv"):
        if seed is not None:
            torch.manual_seed(seed)
        self.markov_decision_process.reset_states(new_episode = day_num == 0, seed = seed)
        init_payoff = float(self.markov_decision_process.get_payoff_curr_ts(deliver = True))
        action_lst_ret = []
        payoff_lst = []
        discount_lst = []
        atomic_payoff_lst = []
        for t in range(self.time_horizon):
            any_action_applied = False
            ## Assign all trip requests with d-closest cars
            active_trip_ods = self.markov_decision_process.get_all_active_trip_ods()
            for active_trip_od in active_trip_ods:
                origin, dest, action = active_trip_od
                selected_car_id = self.markov_decision_process.get_max_battery_car_id_among_d_closest(region = origin, d = self.d)
                if selected_car_id is not None:
                    self.markov_decision_process.transit_within_timestamp(action, car_id = selected_car_id)
                    any_action_applied = True
                    curr_state_counts_full = self.markov_decision_process.get_state_counts(deliver = True)
                    action_lst_ret.append((curr_state_counts_full, action, t, selected_car_id))
#            ## Charge all cars at low battery levels
#            low_battery_car_ids = self.markov_decision_process.get_all_low_battery_car_ids()
            ## Charge all idling cars
            idling_car_ids = self.markov_decision_process.get_all_idling_car_ids()
            for tup in idling_car_ids: #low_battery_car_ids:
                car_id, region = tup
                action_lst = self.markov_decision_process.get_charging_actions(region)
                for action in action_lst:
                    action_applied = self.markov_decision_process.transit_within_timestamp(action, car_id = car_id)
                    if action_applied:
                        curr_state_counts_full = self.markov_decision_process.get_state_counts(deliver = True)
                        any_action_applied = True
                        action_lst_ret.append((curr_state_counts_full, action, t, car_id))
                        break
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
        rerouting_cars = self.markov_decision_process.rerouting_cars
        idling_cars = self.markov_decision_process.idling_cars
        charging_cars = self.markov_decision_process.charging_cars
        return None, None, payoff_lst, action_lst_ret, discounted_payoff, passenger_carrying_cars, rerouting_cars, idling_cars, charging_cars

## This module constructs a corresponding solver given parameters
class SolverFactory:
    def __init__(self, type = "rl", markov_decision_process = None):
        assert type in ["rl", "dp", "greedy"]
        self.type = type
        self.markov_decision_process = markov_decision_process
        self.solver = None
        self.construct_solver()
    
    def construct_solver(self):
        if self.type == "rl":
            self.solver = RL_Solver(self.markov_decision_process)
        elif self.type == "dp":
            self.solver = DP_Solver(self.markov_decision_process)
        elif self.type == "greedy":
            self.solver = Greedy_Solver(self.markov_decision_process)
    
    def get_solver(self):
        return self.solver

## This module generate plots and tables
class ReportFactory:
    def __init__(self, vis_battery_level = 10, markov_decision_process = None):
        self.vis_battery_level = vis_battery_level
        self.markov_decision_process = markov_decision_process
        self.time_horizon = self.markov_decision_process.time_horizon
        self.total_revenue_benchmark = self.markov_decision_process.total_revenue_benchmark
    
    def get_plot(self):
        pass
    
    def plot_single(self, y_arr, xlabel, ylabel, title, figname, x_arr = None, dir = "Plots"):
        if x_arr is None:
            plt.plot(y_arr)
        else:
            plt.plot(x_arr, y_arr)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(f"{dir}/{figname}.png")
        plt.clf()
        plt.close()
    
    def plot_double(self, y_arr, y_arr2, xlabel, ylabel, ylabel2, title, figname, x_arr = None, dir = "Plots"):
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        if x_arr is None:
            ax.plot(y_arr)
            ax2.plot(y_arr2)
        else:
            ax.plot(x_arr, y_arr)
            ax2.plot(x_arr, y_arr2)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        plt.xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel2)
        plt.title(title)
#        plt.tight_layout()
        plt.savefig(f"{dir}/{figname}.png")
        plt.clf()
        plt.close()
    
    def plot_bar(self, x_arr, y_arr, xlabel, ylabel, title, figname, dir = "TablePlots"):
        plt.bar(x_arr, y_arr)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(f"{dir}/{figname}.png")
        plt.clf()
        plt.close()
    
    def plot_multi(self, y_arr_lst, label_lst, xlabel, ylabel, title, figname, dir = "Plots"):
        for y_arr, label in zip(y_arr_lst, label_lst):
            plt.plot(y_arr, label = label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(f"{dir}/{figname}.png")
        plt.clf()
        plt.close()

    def get_training_loss_plot(self, loss_arr, loss_name, figname, loss = True):
        final_loss = float(loss_arr[-1])
        if loss:
            type = "Loss"
        else:
            type = "Payoff"
        if self.total_revenue_benchmark is None or "loss" in loss_name.lower():
            self.plot_single(loss_arr, xlabel = "Policy Iterations", ylabel = type, title = loss_name + f"\nFinal {type} = {final_loss:.2f}", figname = figname, dir = "Plots")
        else:
            loss_arr2 = [x * self.total_revenue_benchmark for x in loss_arr]
            self.plot_double(loss_arr, loss_arr2, xlabel = "Policy Iterations", ylabel = type, ylabel2 = "Revenue", title = loss_name + f"\nFinal {type} = {final_loss:.2f}", figname = figname, dir = "Plots")
    
    def plot_stacked(self, x_arr, y_arr_lst, label_lst, xlabel, ylabel, title, figname):
        assert len(y_arr_lst) == len(label_lst)
        curr_lo = 0
        curr_hi = 0
        x_arr = np.array(x_arr)
        fig, ax = plt.subplots(figsize = (12, 4))
        interval = int(24 * 60 / self.time_horizon)
        day_start = x_arr[0] // self.time_horizon
        start = datetime(1900, 1, 1 + day_start, 0, 0, 0)
        step = timedelta(minutes = interval)
        date_arr = [start]
        for _ in range(len(x_arr) - 1):
            date_arr.append(date_arr[-1] + step)
        for i in range(len(label_lst)):
            curr_hi += y_arr_lst[i]
            label = label_lst[i]
            plt.fill_between(date_arr, curr_lo, curr_hi, label = label)
            curr_lo += y_arr_lst[i]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gcf().axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
        fig.autofmt_xdate()
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"TablePlots/{figname}.png")
        plt.clf()
        plt.close()
    
    def visualize_table(self, df_table, suffix, title = "", detailed = False):
        ## Visualize new requests
        self.plot_single(df_table["num_new_requests"], xlabel = "Time Horizon", ylabel = "# New Trip Requests", title = title, figname = f"new_requests_{suffix}", x_arr = df_table["t"], dir = "TablePlots")
        ## Visualize fulfilled requests
        self.plot_single(df_table["frac_requests_fulfilled"], xlabel = "Time Horizon", ylabel = "% Fulfilled Trip Requests", title = title, figname = f"fulfilled_requests_{suffix}", x_arr = df_table["t"], dir = "TablePlots")
        ## Visualize car status
        self.plot_stacked(df_table["t"], [df_table["frac_passenger-carrying_cars"], df_table["frac_rerouting_cars"], df_table["frac_charging_cars"], df_table["frac_idling_cars"]], label_lst = ["% Passenger-Carrying Cars", "% Rerouting Cars", "% Charging Cars", "% Idling Cars"], xlabel = "Time Horizon", ylabel = "% Cars", title = title, figname = f"car_status_{suffix}")
        ## Visualize trip status
        self.plot_stacked(df_table["t"], [df_table["num_fulfilled_requests"], df_table["num_queued_requests"], df_table["num_abandoned_requests"]], label_lst = ["# Fulfilled Requests", "# Queued Requests", "# Abandoned Requests"], xlabel = "Time Horizon", ylabel = "# Requests", title = title, figname = f"trip_status_{suffix}")
        ## Visualize car battery status
        self.plot_stacked(df_table["t"], [df_table[f"frac_battery_cars_{i}"] for i in range(self.vis_battery_level)], label_lst = [f"{i/self.vis_battery_level*100}%-{((i+1)/self.vis_battery_level)*100}% Battery Cars" for i in range(self.vis_battery_level)], xlabel = "Time Horizon", ylabel = "% Cars", title = title, figname = f"battery_status_{suffix}")
#        self.plot_stacked(df_table["t"], [df_table["frac_high_battery_cars"], df_table["frac_med_battery_cars"], df_table["frac_low_battery_cars"]], label_lst = ["% High Battery Cars", "% Med Battery Cars", "% Low Battery Cars"], xlabel = "Time Steps", ylabel = "% Cars", title = title, figname = f"battery_status_{suffix}")
        ## Visualize region supply & demand balancedness
        if detailed:
            regions = [int(x.split("_")[-1]) for x in df_table.columns if x.startswith("num_cars")]
            num_regions = np.max(regions) + 1
            for region in range(num_regions):
                num_trip_requests_region = df_table[f"num_trip_requests_{region}"]
                num_cars_region = df_table[f"num_cars_region_{region}"]
                self.plot_multi([num_trip_requests_region, num_cars_region], ["# Trip Requests", "# Available Cars"], "Time Horizon", "Supply & Demand", "", f"supply_demand_{region}_{suffix}", dir = "TablePlots")
        ## Visualize charging car distribution across regions
        if detailed:
            regions = [int(x.split("_")[-1]) for x in df_table.columns if x.startswith("num_charging_cars")]
            num_regions = np.max(regions) + 1
            y_lst = []
            label_lst = []
            for region in range(num_regions):
                num_charging_cars_region = df_table[f"num_charging_cars_region_{region}"]
                y_lst.append(num_charging_cars_region)
                label_lst.append(f"# Charging Cars @ Region {region}")
            self.plot_multi(y_lst, label_lst, "Time Horizon", "# Charging Cars", "", f"region_charging_distribution_{suffix}", dir = "TablePlots")
            y_arr = np.array([np.sum(x) for x in y_lst])
            y_arr = y_arr / np.sum(y_arr)
            self.plot_bar([str(x) for x in range(num_regions)], y_arr, "Region", "% Charging Cars", "", f"region_charging_frac_{suffix}", dir = "TablePlots")
#            self.plot_stacked(df_table["t"], y_lst, label_lst = label_lst, xlabel = "Time Steps", ylabel = "% Cars", title = title, figname = f"region_charging_distribution_{suffix}")
    
    def get_table(self, markov_decision_process, action_lst, passenger_carrying_cars, rerouting_cars, idling_cars, charging_cars, detailed = False):
        passenger_carrying_cars = passenger_carrying_cars.numpy()
        rerouting_cars = rerouting_cars.numpy()
        idling_cars = idling_cars.numpy()
        charging_cars = charging_cars.numpy()
        prev_t = -1
        begin = False
        num_active_requests_begin, num_traveling_cars_begin, num_idling_cars_begin, num_charging_cars_begin = 0, 0, 0, 0
        num_active_requests_end, num_traveling_cars_end, num_idling_cars_end, num_charging_cars_end = 0, 0, 0, 0
        num_new_requests = 0
        num_regions = len(markov_decision_process.regions)
        time_horizon = markov_decision_process.time_horizon
        passenger_carrying_cars_arr = np.zeros(time_horizon)
        t_lst = []
        frac_requests_fulfilled_lst = []
        frac_traveling_cars_lst = []
        frac_passenger_carrying_cars_lst = []
        frac_rerouting_cars_lst = []
        frac_charging_cars_lst = []
        frac_idling_cars_lst = []
        num_new_requests_lst = []
        num_fulfilled_requests_lst = []
        num_queued_requests_lst = []
        num_abandoned_requests_lst = []
#        frac_low_battery_cars_lst = []
#        frac_med_battery_cars_lst = []
#        frac_high_battery_cars_lst = []
        frac_battery_cars_lst = [[] for _ in range(self.vis_battery_level)]
        num_trip_requests_dct = {}
        num_cars_dct = {}
        charging_car_dct = {}
        num_total_cars = markov_decision_process.num_total_cars
        for i in range(num_regions):
            num_trip_requests_dct[i] = []
            num_cars_dct[i] = []
            charging_car_dct[i] = []
        trip_time_mat = markov_decision_process.get_trip_time()
        for i in tqdm(range(len(action_lst)), leave = False):
            tup = action_lst[i]
            curr_state_counts, action, t, car_idx = tup
            if t != prev_t:
                begin = True
            else:
                begin = False
            prev_t = t
            if begin:
                num_active_requests_begin = markov_decision_process.get_num_active_trip_requests(curr_state_counts)
                num_new_requests = markov_decision_process.get_num_new_trip_requests(curr_state_counts)
                num_active_requests_vec_begin = markov_decision_process.get_num_active_trip_requests_od(curr_state_counts)
                for region in range(num_regions):
                    if detailed:
                        num_trip_requests_region = markov_decision_process.get_num_trip_requests_region(region, state_counts = curr_state_counts)
                        num_cars_region = markov_decision_process.get_num_cars_region(region, state_counts = curr_state_counts)
                    else:
                        num_trip_requests_region = 0
                        num_cars_region = 0
                    num_trip_requests_dct[region].append(num_trip_requests_region)
                    num_cars_dct[region].append(num_cars_region)
            if car_idx is None:
                num_active_requests_end = markov_decision_process.get_num_active_trip_requests(curr_state_counts)
                num_active_requests_vec_end = markov_decision_process.get_num_active_trip_requests_od(curr_state_counts)
                num_fulfilled_requests_vec = num_active_requests_vec_begin - num_active_requests_vec_end
                for origin in range(num_regions):
                    for dest in range(num_regions):
                        id = origin * num_regions + dest
                        trip_time = max(int(trip_time_mat[t, id]), 1)
                        passenger_carrying_cars_arr[t:(t + trip_time)] += num_fulfilled_requests_vec[id]
                num_traveling_cars_end = passenger_carrying_cars[t] + rerouting_cars[t] #markov_decision_process.get_num_traveling_cars(curr_state_counts)
                num_idling_cars_end = num_total_cars - num_traveling_cars_end - charging_cars[t] #idling_cars[t] #markov_decision_process.get_num_idling_cars(curr_state_counts)
                num_charging_cars_end = charging_cars[t] #markov_decision_process.get_num_charging_cars(curr_state_counts)
                num_total_cars = markov_decision_process.num_total_cars #num_traveling_cars_end + num_idling_cars_end + num_charging_cars_end
                if num_active_requests_begin > 0:
                    frac_requests_fulfilled = (num_active_requests_begin - num_active_requests_end) / num_active_requests_begin
                else:
                    frac_requests_fulfilled = 1
                num_requests_fulfilled = num_active_requests_begin - num_active_requests_end
                num_requests_queued = markov_decision_process.get_num_queued_trip_requests(curr_state_counts)
                num_requests_abandoned = markov_decision_process.get_num_abandoned_trip_requests(curr_state_counts)
                num_battery_cars = markov_decision_process.get_num_cars_w_battery_fine(curr_state_counts, levels = self.vis_battery_level)
#                num_low_battery_cars = markov_decision_process.get_num_cars_w_battery(curr_state_counts, "L")
#                num_med_battery_cars = markov_decision_process.get_num_cars_w_battery(curr_state_counts, "M")
#                num_high_battery_cars = markov_decision_process.get_num_cars_w_battery(curr_state_counts, "H")
                for region in range(num_regions):
                    if detailed:
                        num_charging_cars_region = markov_decision_process.get_num_charging_cars_region(region, t, state_counts = curr_state_counts)
                    else:
                        num_charging_cars_region = 0
                    charging_car_dct[region].append(num_charging_cars_region)
                frac_traveling_cars = num_traveling_cars_end / num_total_cars
                frac_charging_cars = num_charging_cars_end / num_total_cars
                frac_idling_cars = num_idling_cars_end / num_total_cars
                frac_passenger_carrying_cars = passenger_carrying_cars[t] / num_total_cars
                frac_rerouting_cars = (num_traveling_cars_end - passenger_carrying_cars[t]) / num_total_cars
                t_lst.append(t)
                frac_requests_fulfilled_lst.append(frac_requests_fulfilled)
                frac_traveling_cars_lst.append(frac_traveling_cars)
                frac_passenger_carrying_cars_lst.append(frac_passenger_carrying_cars)
                frac_rerouting_cars_lst.append(frac_rerouting_cars)
                frac_charging_cars_lst.append(frac_charging_cars)
                frac_idling_cars_lst.append(frac_idling_cars)
                num_new_requests_lst.append(num_new_requests)
                num_fulfilled_requests_lst.append(num_requests_fulfilled)
                num_queued_requests_lst.append(num_requests_queued)
                num_abandoned_requests_lst.append(num_requests_abandoned)
                for i in range(self.vis_battery_level):
                    frac_battery_cars_lst[i].append(num_battery_cars[i] / num_total_cars)
#                frac_low_battery_cars_lst.append(num_low_battery_cars / num_total_cars)
#                frac_med_battery_cars_lst.append(num_med_battery_cars / num_total_cars)
#                frac_high_battery_cars_lst.append(num_high_battery_cars / num_total_cars)
        dct = {"t": t_lst, "num_new_requests": num_new_requests_lst, "frac_requests_fulfilled": frac_requests_fulfilled_lst, "frac_passenger-carrying_cars": frac_passenger_carrying_cars_lst, "frac_rerouting_cars": frac_rerouting_cars_lst, "frac_charging_cars": frac_charging_cars_lst, "frac_idling_cars": frac_idling_cars_lst, "num_fulfilled_requests": num_fulfilled_requests_lst, "num_queued_requests": num_queued_requests_lst, "num_abandoned_requests": num_abandoned_requests_lst}#, "frac_low_battery_cars": frac_low_battery_cars_lst, "frac_med_battery_cars": frac_med_battery_cars_lst, "frac_high_battery_cars": frac_high_battery_cars_lst}
        for i in range(self.vis_battery_level):
            dct[f"frac_battery_cars_{i}"] = frac_battery_cars_lst[i]
        for region in range(num_regions):
            dct[f"num_trip_requests_{region}"] = num_trip_requests_dct[region]
            dct[f"num_cars_region_{region}"] = num_cars_dct[region]
            dct[f"num_charging_cars_region_{region}"] = charging_car_dct[region]
        df = pd.DataFrame.from_dict(dct)
        return df

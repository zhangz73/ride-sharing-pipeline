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
import Utils.neural as neural
import Utils.lp_solvers as lp_solvers

class IL_Solver(train.Solver):
    def __init__(self, markov_decision_process = None, policy_model_name = "discretized_feedforward", policy_hidden_dim_lst = [10, 10], policy_activation_lst = ["relu", "relu"], policy_batch_norm = False, policy_lr = 1e-2, policy_epoch = 1, policy_batch = 100, policy_decay = 0.1, policy_scheduler_step = 10000, policy_solver = "Adam", policy_retrain = False, descriptor = "IL", dir = ".", device = "cpu", ts_per_network = 1, embedding_dim = 1, num_itr = 100, num_episodes = 100, traj_recollect = True, num_days = 1, gamma = 1, retrain = True, n_cpu = 1, state_reduction = True, **kargs):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process)
        self.reward_df = self.markov_decision_process.reward_df
        self.num_days = num_days
        self.gamma = gamma
        self.retrain = retrain
        self.n_cpu = n_cpu
        self.device = device
        self.descriptor = descriptor
        self.traj_recollect = traj_recollect
        self.state_reduction = state_reduction
        self.lp_solver = lp_solvers.LP_On_AugmentedGraph(markov_decision_process = self.markov_decision_process, num_days = self.num_days, gamma = self.gamma, retrain = self.retrain, n_cpu = self.n_cpu)
        self.num_regions = self.lp_solver.num_regions
        self.num_charging_rates = self.lp_solver.num_charging_rates
        self.num_battery_levels = self.lp_solver.num_battery_levels
        self.charging_rates = self.lp_solver.charging_rates
        self.all_actions = self.lp_solver.all_actions
        self.all_actions_reduced = self.markov_decision_process.get_all_actions(state_reduction = self.state_reduction)
        self.policy_input_dim = self.markov_decision_process.get_state_len(state_reduction = self.state_reduction, model = "policy", use_region = False, has_local = True)
        self.policy_output_dim = len(self.all_actions_reduced)
        self.ts_per_network = ts_per_network
        self.discretized_len = int(math.ceil(self.time_horizon / self.ts_per_network))
        self.num_embeddings = ts_per_network
        self.embedding_dim = embedding_dim
        self.use_embedding = self.ts_per_network > 1
        self.num_itr = num_itr
        self.num_episodes = num_episodes
        self.num_cars = self.markov_decision_process.num_total_cars
        self.pickup_patience = self.markov_decision_process.pickup_patience
        self.connection_patience = self.markov_decision_process.connection_patience
        self.patience_time = self.pickup_patience + self.connection_patience
        self.policy_retrain = policy_retrain
        self.policy_epoch = policy_epoch
        self.policy_batch = min(policy_batch, num_episodes)
        self.policy_model_factory = neural.ModelFactory(policy_model_name, self.policy_input_dim, policy_hidden_dim_lst, policy_activation_lst, self.policy_output_dim, policy_batch_norm, policy_lr, policy_decay, policy_scheduler_step, policy_solver, policy_retrain, self.discretized_len, descriptor + "_policy", dir, device, prob = True, use_embedding = self.use_embedding, num_embeddings = self.num_embeddings, embedding_dim = self.embedding_dim, ts_per_network = self.ts_per_network)
        self.policy_model = self.policy_model_factory.get_model()
        self.policy_optimizer, self.policy_scheduler = self.policy_model_factory.prepare_optimizer()
    
    ## Return a distribution of car actions
    def query_fluid_resolve(self, state_counts, curr_ts):
        ## Re-train the fluid LP given state_counts and curr_ts
        self.lp_solver.set_start_time(curr_ts)
        self.lp_solver.set_lp_option(state_counts = state_counts)
        self.lp_solver.construct_problem()
        self.lp_solver.train()
        lp_x = self.lp_solver.x
        self.lp_solver.reset_x_infer_detailed(strict = False, x_copy = lp_x)
        x_charge, x_travel = self.lp_solver.x_charge, self.lp_solver.x_travel
        ## Extract the policy from the first time-step of the model
        policy_dct = {}
        for dest in range(self.num_regions):
            _, charge_x_ids = self.lp_solver.get_relevant_x(0, dest, 0, 0, strict = False)
            policy_dct[dest] = {"charge": {}}
            for id in range(len(charge_x_ids)):
                policy_dct[dest]["charge"][self.charging_rates[id]] = x_charge[charge_x_ids[id]]
            for eta in range(self.pickup_patience + 1):
                travel_x_ids, _ = self.lp_solver.get_relevant_x(0, dest, 0, eta, strict = False)
                for region in range(self.num_regions):
                    if len(travel_x_ids) > 0:
                        policy_dct[dest][(eta, region)] = x_travel[travel_x_ids[region]]
                    else:
                        policy_dct[dest][(eta, region)] = 0
        return policy_dct
    
    def get_policy_indicator(self, x):
        if x >= 1:
            return 1, x - 1
        if x <= 0:
            return 0, 0
        rv = np.random.binomial(n = 1, p = x)
        if rv == 1:
            return 1, 0
        return 0, 0
    
    ## Get an atomic action for each category (region, battery, eta) of cars
    ## State: global state + car category
    def collect_data_single(self, n_traj = 1):
        data_dict = {}
        for t in range(self.time_horizon):
            data_dict[t] = {"state_counts": [], "atomic_actions": []}
        markov_decision_process = self.markov_decision_process
        payoff = 0
        for traj in tqdm(range(n_traj)):
            ## Simulate trajectory
            ## Query fluid resolve to get joint actions
            markov_decision_process.reset_states(new_episode = True, seed = None)
            for t in tqdm(range(self.time_horizon), leave = False):
                state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                fluid_policy_dct = self.query_fluid_resolve(state_counts_full, t)
                available_car_ids = markov_decision_process.get_available_car_ids(True)
                num_available_cars = len(available_car_ids)
                for car_idx in range(num_available_cars):
                    car = markov_decision_process.state_dict[available_car_ids[car_idx]]
                    car_id = available_car_ids[car_idx]
                    curr_state_counts = markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = car_id)
                    curr_state_counts = curr_state_counts.view((1, len(curr_state_counts)))
                    car_dest, car_eta, car_battery = car.get_dest(), car.get_time_to_dest(), car.get_battery()
                    ## Extract atomic action from the dct
                    action_assigned = False
                    if car_eta == 0:
                        for i in range(len(self.charging_rates)):
                            ind, new_x = self.get_policy_indicator(fluid_policy_dct[car_dest]["charge"][self.charging_rates[i]])
                            if ind == 1:
                                action_id = markov_decision_process.query_reduced_action(("charge", self.charging_rates[i]))
                                action = self.all_actions_reduced[action_id]
                                action_success = markov_decision_process.transit_within_timestamp(action, car_id = car_id, reduced = True)
                                if action_success:
                                    action_assigned = True
                                    fluid_policy_dct[car_dest]["charge"][self.charging_rates[i]] = new_x
                                    break
                    if not action_assigned:
                        for region in range(self.num_regions):
                            ind, new_x = self.get_policy_indicator(fluid_policy_dct[car_dest][(car_eta, region)])
                            if ind == 1:
                                action_id = markov_decision_process.query_reduced_action(("travel", region))
                                action = self.all_actions_reduced[action_id]
                                action_success = markov_decision_process.transit_within_timestamp(action, car_id = car_id, reduced = True)
                                if action_success:
                                    action_assigned = True
                                    fluid_policy_dct[car_dest][(car_eta, region)] = new_x
                                    break
                    if not action_assigned:
                        action_id = markov_decision_process.query_reduced_action(("nothing"))
                        action = self.all_actions_reduced[action_id]
                        markov_decision_process.transit_within_timestamp(action, car_id = car_id, reduced = True)
                    data_dict[t]["state_counts"].append(curr_state_counts)
                    data_dict[t]["atomic_actions"].append(action_id)
                markov_decision_process.transit_across_timestamp()
            curr_payoff = markov_decision_process.get_payoff_curr_ts(deliver = True)
            payoff += curr_payoff
        for t in range(self.time_horizon):
            data_dict[t]["state_counts"] = torch.cat(data_dict[t]["state_counts"], dim = 0)
            data_dict[t]["atomic_actions"] = torch.tensor(data_dict[t]["atomic_actions"])
        return data_dict, payoff
    
    def collect_data(self, n_traj = 1):
        batch_size = int(math.ceil(n_traj / self.n_cpu))
        results = Parallel(n_jobs = self.n_cpu)(delayed(self.collect_data_single)(
            min(n_traj, (i + 1) * batch_size) - i * batch_size
        ) for i in range(self.n_cpu))
        data_dict = {}
        for t in range(self.time_horizon):
            data_dict[t] = {"state_counts": [], "atomic_actions": []}
        total_payoff = 0
        for tup in results:
            res, payoff = tup
            total_payoff += payoff / n_traj
            for t in range(self.time_horizon):
                data_dict[t]["state_counts"].append(res[t]["state_counts"])
                data_dict[t]["atomic_actions"].append(res[t]["atomic_actions"])
        for t in range(self.time_horizon):
            data_dict[t]["state_counts"] = torch.cat(data_dict[t]["state_counts"], dim = 0)
            data_dict[t]["atomic_actions"] = torch.cat(data_dict[t]["atomic_actions"])
        print(total_payoff)
        return data_dict

    def remove_infeasible_actions(self, state_counts, ts, output, car_id = None, state_count_check = None):
        ## Eliminate infeasible actions
        ret = torch.ones(len(output))
        mask = self.markov_decision_process.state_counts_to_potential_feasible_actions(self.state_reduction, state_counts = state_count_check, car_id = car_id)
        ret = ret * mask
        return ret

    def policy_predict(self, state_counts, ts, prob = True, remove_infeasible = True, car_id = None, state_count_check = None):
        state_counts_input = state_counts
        output = self.policy_model((ts, state_counts_input))
        if len(state_counts.shape) == 1:
            ret = self.remove_infeasible_actions(state_counts.cpu(), ts, output, car_id = car_id, state_count_check = state_count_check)
        else:
            ret_lst = []
            for i in range(state_counts.shape[0]):
                ret = self.remove_infeasible_actions(state_counts[i,:].cpu(), ts, output[i,:], car_id = car_id, state_count_check = state_count_check)
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

    ## Train a neural network classifier for atomic actions given a state and car category
    def train(self):
        if self.traj_recollect:
            self.data_dict = self.collect_data(n_traj = self.num_episodes)
            # torch.save(self.data_dict, f"TrainingData/il_traj_{self.descriptor}.pt")
        else:
            self.data_dict = torch.load(f"TrainingData/il_traj_{self.descriptor}.pt")
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_arr = []
        self.policy_optimizer.zero_grad(set_to_none=True)
        for itr in tqdm(range(self.num_itr)):
            total_loss = 0
            for t in tqdm(range(self.time_horizon), leave = False):
                batch_idx = torch.from_numpy(np.random.choice(len(self.data_dict[t]["state_counts"]), size = min(self.policy_batch, len(self.data_dict[t]["state_counts"])), replace = False))
                curr_state_counts_lst = self.data_dict[t]["state_counts"][batch_idx, :]
                atomic_actions_lst = self.data_dict[t]["atomic_actions"][batch_idx]
                predicted_actions = self.policy_predict(state_counts = curr_state_counts_lst, ts = t, prob = True, remove_infeasible = False)
                loss = loss_fn(predicted_actions, atomic_actions_lst)
                total_loss += loss / self.time_horizon
                # loss.backward()
                # self.policy_optimizer.step()
                # self.policy_scheduler.step()
            loss_arr.append(float(total_loss.data))
            total_loss.backward()
            self.policy_optimizer.step()
            self.policy_scheduler.step()
        self.policy_model_factory.update_model(self.policy_model, update_ts = False)
        self.policy_model_factory.save_to_file(include_ts = True)
        return loss_arr
    
    def action_is_feasible(self, state_counts, ts, action_id, car_id = None, state_count_check = None):
        return self.markov_decision_process.action_is_potentially_feasible(action_id, reduced = self.state_reduction, car_id = car_id, state_counts = state_count_check)

    def evaluate(self, seed = None, train = False, return_data = False, return_action = False, debug = False, debug_dir = "debugging_log.txt", markov_decision_process = None, day_num = 0, log_policy = False, policy_log_dir = "PolicyLogs/policy_log.csv"):
        self.policy_model.eval()
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
            for car_idx in tqdm(range(num_available_cars), leave = False):
                ## Perform state transitions
                curr_state_counts = markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx])#.to(device = self.device)
#                if return_action:
                curr_state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                curr_state_counts = curr_state_counts.view((1, len(curr_state_counts)))
                action_id_prob = self.policy_predict(curr_state_counts, t, prob = True, car_id = available_car_ids[car_idx], state_count_check = curr_state_counts_full)
                action_id_prob = action_id_prob.flatten()
                curr_state_counts = curr_state_counts.flatten()
                if action_id_prob is not None:
                    action_id_prob = action_id_prob.cpu().detach().numpy()
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
                    action = self.all_actions_reduced[int(action_id)]
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
                        next_state_counts = markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx])#.to(device = self.device)
                    else:
                        next_state_counts = None
                    payoff = markov_decision_process.get_payoff_curr_ts().clone()
                    atomic_payoff_lst.append(payoff - curr_payoff)
                    discount_lst.append(self.gamma ** t)
                    ## Compute loss
                    if not return_data:
#                        ## Compute values
                        curr_payoff = float(markov_decision_process.get_payoff_curr_ts(deliver = True))
                        payoff_lst.append(curr_payoff)
            if not transit_applied:
                if return_action:
                    curr_state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                    action_lst.append((curr_state_counts_full, None, t, None))
                markov_decision_process.transit_across_timestamp()
        if not return_data:
            payoff_lst = torch.tensor(payoff_lst)
        discount_lst = torch.tensor(discount_lst)
        atomic_payoff_lst = torch.tensor(atomic_payoff_lst)
        discounted_payoff = torch.sum(discount_lst * atomic_payoff_lst) / markov_decision_process.get_total_market_revenue()
        payoff_end_raw = float(markov_decision_process.get_payoff_curr_ts(deliver = False))
        payoff_raw = payoff_end_raw - payoff_begin_raw
        if return_data:
            final_payoff = float(markov_decision_process.get_payoff_curr_ts(deliver = True))
            return curr_state_lst, next_state_lst, state_action_advantage_lst, final_payoff, discounted_payoff, payoff_raw, total_revenue
        passenger_carrying_cars = markov_decision_process.passenger_carrying_cars
        return None, None, payoff_lst, action_lst, discounted_payoff, passenger_carrying_cars

import sys
import gc
import math
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    def __init__(self, type = "sequential", markov_decision_process = None, state_reduction = False):
        assert type in ["sequential", "group"]
        self.type = type
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
    def __init__(self, markov_decision_process = None, value_model_name = "discretized_feedforward", value_hidden_dim_lst = [10, 10], value_activation_lst = ["relu", "relu"], value_batch_norm = False, value_lr = 1e-2, value_epoch = 1, value_batch = 100, value_decay = 0.1, value_scheduler_step = 10000, value_solver = "Adam", value_retrain = False, policy_model_name = "discretized_feedforward", policy_hidden_dim_lst = [10, 10], policy_activation_lst = ["relu", "relu"], policy_batch_norm = False, policy_lr = 1e-2, policy_epoch = 1, policy_batch = 100, policy_decay = 0.1, policy_scheduler_step = 10000, policy_solver = "Adam", policy_retrain = False, descriptor = "PPO", dir = ".", device = "cpu", num_itr = 100, num_episodes = 100, ckpt_freq = 100, benchmarking_policy = "uniform", eps = 0.2, policy_syncing_freq = 1, value_syncing_freq = 1, n_cpu = 1, n_threads = 4, lazy_removal = False, state_reduction = False):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process, state_reduction = state_reduction)
        ## Store some commonly used variables
        self.value_input_dim = self.markov_decision_process.get_state_len(state_reduction = state_reduction, model = "value")
        self.policy_input_dim = self.markov_decision_process.get_state_len(state_reduction = state_reduction, model = "policy")
        self.value_output_dim = 1
        self.policy_output_dim = len(self.action_lst)
        self.discretized_len = self.time_horizon
        self.num_itr = num_itr
        self.num_episodes = num_episodes
        self.num_cars = self.markov_decision_process.num_cars
        self.ckpt_freq = ckpt_freq
        self.value_epoch = value_epoch
        self.policy_epoch = policy_epoch
        self.value_batch = min(value_batch, num_episodes)
        self.policy_batch = min(policy_batch, num_episodes)
        self.benchmarking_policy = benchmarking_policy
        self.eps = eps
        self.policy_syncing_freq = policy_syncing_freq
        self.value_syncing_freq = value_syncing_freq
        self.device = device
#        self.payoff_map = self.payoff_map.to(device = self.device)
        self.one_minus_eps = torch.tensor(1 - eps).to(device = self.device)
        self.one_plus_eps = torch.tensor(1 + eps).to(device = self.device)
        self.n_cpu = n_cpu
        self.lazy_removal = lazy_removal
        self.state_reduction = state_reduction
        ## Construct models
        self.value_model_factory = neural.ModelFactory(value_model_name, self.value_input_dim, value_hidden_dim_lst, value_activation_lst, self.value_output_dim, value_batch_norm, value_lr, value_decay, value_scheduler_step, value_solver, value_retrain, self.discretized_len, descriptor + "_value", dir, device)
        self.policy_model_factory = neural.ModelFactory(policy_model_name, self.policy_input_dim, policy_hidden_dim_lst, policy_activation_lst, self.policy_output_dim, policy_batch_norm, policy_lr, policy_decay, policy_scheduler_step, policy_solver, policy_retrain, self.discretized_len, descriptor + "_policy", dir, device, prob = True)
        self.value_model = self.get_value_model()
        self.policy_model = self.get_policy_model()
        self.benchmark_policy_model = copy.deepcopy(self.policy_model)
        self.benchmark_value_model = copy.deepcopy(self.value_model)
        self.value_optimizer, self.value_scheduler = self.value_model_factory.prepare_optimizer()
        self.policy_optimizer, self.policy_scheduler = self.policy_model_factory.prepare_optimizer()
        self.markov_decision_process_pg = copy.deepcopy(markov_decision_process)
    
    def get_value_model(self):
        return self.value_model_factory.get_model()
    
    def get_policy_model(self):
        return self.policy_model_factory.get_model()
    
    def get_advantage(self, curr_state_counts, next_state_counts, action_id, ts, next_ts, payoff):
#        payoff = self.payoff_map[ts, action_id]
        if len(curr_state_counts.shape) > 1:
            curr, next = curr_state_counts[:,:self.value_input_dim], next_state_counts[:,:self.value_input_dim]
        else:
            curr, next = curr_state_counts[:self.value_input_dim], next_state_counts[:self.value_input_dim]
        curr_value = self.benchmark_value_model((ts, curr)).reshape((-1,))
        if next_ts < self.time_horizon - 1:
            next_value = self.benchmark_value_model((next_ts, next)).reshape((-1,))
        else:
            next_value = 0
        return payoff + next_value - curr_value
    
    def get_ratio(self, state_counts, action_id, ts, clipped = False, eps = 0.2, car_id = None):
        prob_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = False, car_id = car_id)
        action_id = action_id.reshape((len(action_id), 1))
        prob = prob_output.gather(1, action_id) #prob_output[action_id]
        prob_benchmark_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = False, use_benchmark = True, car_id = car_id)
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
    
    def get_value_loss(self, state_action_advantage_lst_episodes):
        total_value_loss = 0
        val_num = 0
        value_dct = {}
        for t in range(self.time_horizon):
            value_dct[t] = {"state_counts": [], "payoff": []}
        for day in range(self.value_batch):
            state_num = len(state_action_advantage_lst_episodes[day])
            payoff = 0
            if state_num > 0:
                final_payoff = state_action_advantage_lst_episodes[day][state_num - 1][4].clone()
            for i in range(state_num - 1, -1, -1):
                tup = state_action_advantage_lst_episodes[day][i]
                curr_state_counts, action_id, _, t, curr_payoff, _, _ = tup
                payoff = final_payoff - curr_payoff
                lens = len(curr_state_counts)
                value_dct[t]["payoff"].append(payoff)
                value_dct[t]["state_counts"].append(curr_state_counts.reshape((1, lens)))
                val_num += 1
        for t in range(self.time_horizon):
            payoff_lst = torch.tensor(value_dct[t]["payoff"]).to(device = self.device)
            if len(value_dct[t]["state_counts"]) > 0:
                state_counts_lst = torch.cat(value_dct[t]["state_counts"], dim = 0)[:,:self.value_input_dim]
                value_model_output = self.value_model((t, state_counts_lst)).reshape((-1,))
                total_value_loss += torch.sum((value_model_output - payoff_lst) ** 2) / val_num / self.num_cars / self.time_horizon
#        total_value_loss /= val_num #self.num_episodes
        value_dct = None
        return total_value_loss
    
    def get_data_single(self, num_episodes, worker_num = 0):
        state_action_advantage_lst_episodes = []
        total_payoff = 0
        for day in tqdm(range(num_episodes)):
            state_action_advantage_lst, payoff_val = self.evaluate(train = True, return_data = True, debug = False, debug_dir = None, lazy_removal = self.lazy_removal)
            state_action_advantage_lst_episodes.append(state_action_advantage_lst)
            total_payoff += payoff_val
        return state_action_advantage_lst_episodes, (num_episodes, total_payoff)
    
    def train(self, return_payoff = False, debug = False, debug_dir = "debugging_log.txt"):
        value_loss_arr = []
        policy_loss_arr = []
        payoff_arr = []
        self.value_model.train()
        self.policy_model.train()
        ## Policy Iteration
        dct_outer = {}
        if debug:
            with open(debug_dir, "w") as f:
                f.write("------------ Debugging output for day 0 ------------\n")
        if return_payoff:
            _, _, payoff_lst, _ = self.evaluate(return_action = False, seed = 0)
            payoff_val = float(payoff_lst[-1].data)
            payoff_arr.append(payoff_val)
        for itr in range(self.num_itr + 0):
            print(f"Iteration #{itr+1}/{self.num_itr}:")
            if debug:
                with open(debug_dir, "a") as f:
                    f.write(f"Itr = {itr+1}/{self.num_itr}:\n")
            ## Obtain simulated data from each episode
            state_action_advantage_lst_episodes = []
            print("\tGathering data...")
            if self.n_cpu == 1:
                state_action_advantage_lst_episodes, tup = self.get_data_single(self.num_episodes)
                payoff_val = float(tup[1] / tup[0])
            else:
                batch_size = int(math.ceil(self.num_episodes / self.n_cpu))
                results = Parallel(n_jobs = self.n_cpu)(delayed(
                    self.get_data_single
                )(min((i + 1) * batch_size, self.num_episodes) - i * batch_size, i) for i in range(self.n_cpu))
                print("Gathering results...")
                payoff_val = 0
                for res in results:
                    state_action_advantage_lst_episodes += res[0]
                    tup = res[1]
                    payoff_val += tup[1]
                payoff_val /= self.num_episodes
                results = None
#            payoff_arr.append(payoff_val)
            if itr == self.num_itr:
                break
                
            ## Update models
            ## Update value models
            print("\tUpdating value models...")
            for _ in tqdm(range(self.value_epoch)):
                batch_idx = np.random.choice(self.num_episodes, size = self.value_batch, replace = False)
                self.value_optimizer.zero_grad(set_to_none=True)
#                 total_value_loss = self.get_max_value_loss([state_action_advantage_lst_episodes[idx] for idx in batch_idx])
                total_value_loss = self.get_value_loss([state_action_advantage_lst_episodes[idx] for idx in batch_idx])
                total_value_loss.backward()
                self.value_optimizer.step()
                self.value_scheduler.step()
            if debug:
                with open(debug_dir, "a") as f:
                    f.write(f"\tFinal Value Loss = {float(total_value_loss.data)}\n")
            
            ## Update policy models
            print("\tUpdating policy models...")
            for _ in tqdm(range(self.policy_epoch)):
                policy_dct = {}
                for t in range(self.time_horizon):
                    for offset in [0, 1]:
                        tup = (t, t + offset)
                        policy_dct[tup] = {"curr_state_counts": [], "next_state_counts": [], "action_id": [], "atomic_payoff": []}
                batch_idx = np.random.choice(self.num_episodes, size = self.policy_batch, replace = False)
                total_policy_loss = 0
                self.policy_optimizer.zero_grad(set_to_none=True)
                for day in batch_idx:
                    state_num = len(state_action_advantage_lst_episodes[day])
                    for i in range(state_num):
                        tup = state_action_advantage_lst_episodes[day][i]
                        curr_state_counts, action_id, next_state_counts, t, _, next_t, atomic_payoff = tup
                        lens = len(curr_state_counts)
                        policy_dct[(t, next_t)]["curr_state_counts"].append(curr_state_counts.reshape((1, lens)))
                        policy_dct[(t, next_t)]["next_state_counts"].append(next_state_counts.reshape((1, lens)))
                        policy_dct[(t, next_t)]["action_id"].append(action_id)
                        policy_dct[(t, next_t)]["atomic_payoff"].append(atomic_payoff)
#                        advantage = self.get_advantage(curr_state_counts, next_state_counts, action_id, t, next_t)
#                        ratio, ratio_clipped = self.get_ratio(curr_state_counts, action_id, t, clipped = True, eps = self.eps)
#                        loss_curr = -torch.min(ratio * advantage, ratio_clipped * advantage)
#                        total_policy_loss += loss_curr[0]
                for t in range(self.time_horizon):
                    for offset in [0, 1]:
                        next_t = t + offset
                        if len(policy_dct[(t, next_t)]["curr_state_counts"]) > 0:
                            curr_state_counts_lst = torch.cat(policy_dct[(t, next_t)]["curr_state_counts"], dim = 0)
                            next_state_counts_lst = torch.cat(policy_dct[(t, next_t)]["next_state_counts"], dim = 0)
                            action_id_lst = torch.tensor(policy_dct[(t, next_t)]["action_id"]).to(device = self.device)
                            atomic_payoff_lst = torch.tensor(policy_dct[(t, next_t)]["atomic_payoff"]).to(device = self.device)
                            advantage = self.get_advantage(curr_state_counts_lst, next_state_counts_lst, action_id_lst, t, next_t, atomic_payoff_lst)
                            ## TODO: Fix it!!!
                            ratio, ratio_clipped = self.get_ratio(curr_state_counts_lst, action_id_lst, t, clipped = True, eps = self.eps)
                            loss_curr = -torch.min(ratio * advantage, ratio_clipped * advantage)
                            total_policy_loss += torch.sum(loss_curr) / len(batch_idx)
#                total_policy_loss /= len(batch_idx)
                total_policy_loss.backward()
                self.policy_optimizer.step()
                self.policy_scheduler.step()
                policy_dct = None
            if itr % self.value_syncing_freq == 0:
                self.benchmark_value_model = copy.deepcopy(self.value_model)
            if itr % self.policy_syncing_freq == 0:
                self.benchmark_policy_model = copy.deepcopy(self.policy_model)
            ## Save loss data
            value_loss_arr.append(float(total_value_loss.data))
            policy_loss_arr.append(float(total_policy_loss.data))
            ## Checkpoint
            if itr > 0 and itr % self.ckpt_freq == 0:
                self.value_model_factory.update_model(self.value_model, update_ts = False)
                self.policy_model_factory.update_model(self.policy_model, update_ts = False)
                descriptor = f"_itr={itr}"
                self.value_model_factory.save_to_file(descriptor, include_ts = True)
                self.policy_model_factory.save_to_file(descriptor, include_ts = True)
            
            if return_payoff:
                _, _, payoff_lst, _ = self.evaluate(return_action = False, seed = 0)
                payoff_val = float(payoff_lst[-1].data)
                payoff_arr.append(payoff_val)
        self.benchmark_policy_model = copy.deepcopy(self.policy_model)
        self.benchmark_value_model = copy.deepcopy(self.value_model)
        # Save final model
#         self.value_model_factory.update_model(self.value_model, update_ts = False)
#         self.policy_model_factory.update_model(self.benchmark_policy_model, update_ts = False)
#         self.value_model_factory.save_to_file(include_ts = True)
#         self.policy_model_factory.save_to_file(include_ts = True)
        return value_loss_arr, policy_loss_arr, payoff_arr

    def policy_describe(self, action_id_prob):
        ret = ""
        for action_id in range(len(action_id_prob)):
            prob = action_id_prob[action_id]
            action_msg = self.all_actions[action_id].describe()
            msg = f"{action_msg}: {prob}"
            ret += f"\t\t{msg}\n"
        return ret
    
    def policy_predict(self, state_counts, ts, prob = True, remove_infeasible = True, use_benchmark = False, lazy_removal = False, car_id = None, state_count_check = None):
        if not use_benchmark:
            output = self.policy_model((ts, state_counts))
        else:
            output = self.benchmark_policy_model((ts, state_counts))
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
    
    def remove_infeasible_actions(self, state_counts, ts, output, lazy_removal = False, car_id = None, state_count_check = None):
        ## Eliminate infeasible actions
        ret = torch.ones(len(output))
        mask = self.markov_decision_process.state_counts_to_potential_feasible_actions(self.state_reduction, state_counts)
        ret = ret * mask
        if not lazy_removal and self.state_reduction:
            potential_feasible_action_ids = torch.where(ret > 0)[0]
            for action_id in potential_feasible_action_ids:
                action_id = int(action_id)
                if not self.action_is_feasible(state_counts, ts, action_id, car_id = car_id, state_count_check = state_count_check):
                    ret[action_id] = 0
        return ret
    
    def action_is_feasible(self, state_counts, ts, action_id, car_id = None, state_count_check = None):
        return self.markov_decision_process.action_is_potentially_feasible(action_id, reduced = self.state_reduction, car_id = car_id)
#        self.markov_decision_process_pg.set_states(state_count_check, ts)
#        action = self.all_actions[action_id]
#        return self.markov_decision_process_pg.transit_within_timestamp(action, reduced = self.state_reduction, car_id = car_id)
    
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
    
    def evaluate(self, seed = None, train = False, return_data = False, return_action = False, debug = False, debug_dir = "debugging_log.txt", lazy_removal = False):
        if not train:
            self.value_model.eval()
            self.benchmark_policy_model.eval()
        return_data = return_data and train
        if seed is not None:
            torch.manual_seed(seed)
        self.markov_decision_process.reset_states()
        payoff_lst = []
        model_value_lst = []
        action_lst = []
        policy_loss = 0
        state_action_advantage_lst = []
        total_cars = 0
        total_trips = 0
        for t in tqdm(range(self.time_horizon), leave = False):
            ## Add up ||v_model - v_hat||^2
            ## Add up ratio * advantage
            available_car_ids = self.markov_decision_process.get_available_car_ids(self.state_reduction)
            num_available_cars = len(available_car_ids)
            total_cars += num_available_cars
#            if not train:
#                print("t = ", t)
#                print(self.markov_decision_process.describe_state_counts())
            transit_applied = False
            for car_idx in tqdm(range(num_available_cars), leave = False):
                ## Perform state transitions
                curr_state_counts = self.markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx]).to(device = self.device)
                if return_action:
                    curr_state_counts_full = self.markov_decision_process.get_state_counts(deliver = True)
                action_id_prob = self.policy_predict(curr_state_counts, t, prob = True, use_benchmark = True, lazy_removal = lazy_removal, car_id = available_car_ids[car_idx], state_count_check = None)
                if action_id_prob is not None:
                    action_id_prob = action_id_prob.cpu().detach().numpy()
                    if debug:
                        with open(debug_dir, "a") as f:
                            msg = self.policy_describe(action_id_prob)
                            payoff = self.markov_decision_process.get_payoff_curr_ts().clone()
                            self.value_model.eval()
                            inferred_value = float(self.value_model((t, curr_state_counts)).data)
                            f.write(f"\tt = {t}, car_id = {car_idx}, payoff = {payoff}, inferred state value = {inferred_value}:\n")
                            f.write(self.markov_decision_process.describe_state_counts(curr_state_counts))
                            f.write(msg)
                    if train:
                        is_feasible = False
                        while not is_feasible:
                            action_id = np.random.choice(len(action_id_prob), p = action_id_prob)
                            is_feasible = self.action_is_feasible(curr_state_counts, t, int(action_id), car_id = available_car_ids[car_idx], state_count_check = None)
                            if not is_feasible:
                                print(self.all_actions[action_id].describe())
                                print(self.markov_decision_process.describe_state_counts())
                                assert False
                    else:
                        action_id = np.argmax(action_id_prob)
                    action = self.all_actions[int(action_id)]
                    if action.get_type() in ["pickup", "rerouting"]:
                        total_trips += 1
                    if return_action:
                        action_lst.append((curr_state_counts_full, action, t, car_idx))
                    curr_payoff = self.markov_decision_process.get_payoff_curr_ts().clone().to(device = self.device)
                    res = self.markov_decision_process.transit_within_timestamp(action, self.state_reduction, available_car_ids[car_idx])
                    next_t = t
                    if car_idx == num_available_cars - 1:
                        transit_applied = True
                        self.markov_decision_process.transit_across_timestamp()
                        next_t += 1
                    next_state_counts = self.markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx]).to(device = self.device)
                    payoff = self.markov_decision_process.get_payoff_curr_ts().clone()
                    if return_data: # and t < self.time_horizon - 1:
                        state_action_advantage_lst.append((curr_state_counts, action_id, next_state_counts, t, curr_payoff, next_t, payoff - curr_payoff))
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
                        curr_payoff = float(self.markov_decision_process.get_payoff_curr_ts(deliver = True))
#                        curr_value = self.value_model((t, curr))
                        payoff_lst.append(curr_payoff)
#                        if t == 0:
#                            if action.get_type() in ["pickup", "rerouting"]:
#                                print(action.get_origin(), action.get_dest(), curr_payoff)
#                        model_value_lst.append(curr_value)
            if not transit_applied:
                self.markov_decision_process.transit_across_timestamp()
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
        if return_data:
            final_payoff = float(self.markov_decision_process.get_payoff_curr_ts(deliver = True))
            return state_action_advantage_lst, final_payoff
#        print("total cars", total_cars)
#        print("total trips", total_trips)
#        print("payoff", float(payoff_lst[-1].data))
        #return value_loss.cpu(), policy_loss.cpu(), payoff_lst, action_lst
        return None, None, payoff_lst, action_lst

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
    def __init__(self):
        pass
    
    def get_plot(self):
        pass
    
    def get_training_loss_plot(self, loss_arr, loss_name, figname):
        final_loss = float(loss_arr[-1])
        plt.plot(loss_arr)
        plt.xlabel("Training Episodes")
        plt.ylabel("Loss")
        plt.title(loss_name + f"\nFinal Loss = {final_loss:.2f}")
        plt.savefig(f"Plots/{figname}.png")
        plt.clf()
        plt.close()
    
    def get_table(self):
        pass

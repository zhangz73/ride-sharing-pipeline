import sys
import gc
import math
import copy
import numpy as np
import pandas as pd
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
    def __init__(self, markov_decision_process = None, value_model_name = "discretized_feedforward", value_hidden_dim_lst = [10, 10], value_activation_lst = ["relu", "relu"], value_batch_norm = False, value_lr = 1e-2, value_epoch = 1, value_batch = 100, value_decay = 0.1, value_scheduler_step = 10000, value_solver = "Adam", value_retrain = False, policy_model_name = "discretized_feedforward", policy_hidden_dim_lst = [10, 10], policy_activation_lst = ["relu", "relu"], policy_batch_norm = False, policy_lr = 1e-2, policy_epoch = 1, policy_batch = 100, policy_decay = 0.1, policy_scheduler_step = 10000, policy_solver = "Adam", policy_retrain = False, descriptor = "PPO", dir = ".", device = "cpu", num_itr = 100, num_episodes = 100, ckpt_freq = 100, benchmarking_policy = "uniform", eps = 0.2, eps_sched = 1000, eps_eta = 0.5, policy_syncing_freq = 1, value_syncing_freq = 1, n_cpu = 1, n_threads = 4, lazy_removal = False, state_reduction = False):
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
        self.markov_decision_process_lst = []
        for i in range(self.n_cpu): #self.n_cpu
            cp = copy.deepcopy(markov_decision_process)
            self.markov_decision_process_lst.append(cp)
        self.value_scale = self.value_model_factory.get_value_scale()
    
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
        with torch.no_grad():
            curr_value = self.value_model((ts, curr)).reshape((-1,))
        mu, sd = self.value_scale[ts]
        curr_value = curr_value * sd + mu
        if next_ts < self.time_horizon - 1:
            with torch.no_grad():
                next_value = self.value_model((next_ts, next)).reshape((-1,))
            mu2, sd2 = self.value_scale[next_ts]
            next_value = next_value * sd2 + mu2
        else:
            next_value = 0
        return (payoff + next_value - curr_value) / sd
    
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
    
    def get_value_loss(self, state_action_advantage_lst_episodes, update_value_scale = False):
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
            if update_value_scale:
                mu, sd = torch.mean(payoff_lst), torch.std(payoff_lst) + 1 #(self.time_horizon - t)
                self.value_scale[t] = (mu, sd)
            mu, sd = self.value_scale[t]
            payoff_lst = (payoff_lst - mu) / sd
            if len(value_dct[t]["state_counts"]) > 0:
                state_counts_lst = torch.cat(value_dct[t]["state_counts"], dim = 0)[:,:self.value_input_dim]
                value_model_output = self.value_model((t, state_counts_lst)).reshape((-1,))
                total_value_loss += torch.sum((value_model_output - payoff_lst) ** 2) / val_num #/ self.num_cars / self.time_horizon
#        total_value_loss /= val_num #self.num_episodes
        value_dct = None
        if update_value_scale:
            self.value_model_factory.set_value_scale(self.value_scale)
        return total_value_loss
    
    def get_data_single(self, num_episodes, worker_num = 0):
        state_action_advantage_lst_episodes = []
        total_payoff = 0
        for day in tqdm(range(num_episodes)):
            state_action_advantage_lst, payoff_val = self.evaluate(train = True, return_data = True, debug = False, debug_dir = None, lazy_removal = self.lazy_removal, markov_decision_process = self.markov_decision_process_lst[worker_num])
            state_action_advantage_lst_episodes.append(state_action_advantage_lst)
            total_payoff += payoff_val
        return state_action_advantage_lst_episodes, (num_episodes, total_payoff)
    
    def train(self, return_payoff = False, debug = False, debug_dir = "debugging_log.txt", label = ""):
        value_loss_arr = []
        policy_loss_arr = []
        payoff_arr = []
        self.value_model.train()
        self.policy_model.train()
        ## Policy Iteration
        dct_outer = {}
        report_factory = ReportFactory()
        if debug:
            with open(debug_dir, "w") as f:
                f.write("------------ Debugging output for day 0 ------------\n")
#         payoff_tot = 0
#         num_trials = 50
#         for i in tqdm(range(num_trials), leave = False):
# #         if return_payoff:
#             _, _, payoff_lst, _ = self.evaluate(return_action = True, seed = None)
# #            _, payoff_val = self.evaluate(train = True, return_data = True, seed = None, lazy_removal = self.lazy_removal, markov_decision_process = self.markov_decision_process_lst[0])
#             payoff_val = float(payoff_lst[-1].data)
#             payoff_tot += payoff_val
#         print(payoff_tot / num_trials)
#         assert False
#            payoff_arr.append(payoff_val)
        with open(f"payoff_log_{label}.txt", "w") as f:
            f.write("Payoff Logs:\n")
        eps = self.eps
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
            payoff_arr.append(payoff_val)
            with open(f"payoff_log_{label}.txt", "a") as f:
                f.write(f"{float(payoff_val)}\n")
            if itr == self.num_itr:
                break
                
            ## Update models
            ## Update value models
            print("\tUpdating value models...")
            value_curr_arr = []
            for _ in tqdm(range(self.value_epoch)):
                batch_idx = np.random.choice(self.num_episodes, size = self.value_batch, replace = False)
                self.value_optimizer.zero_grad(set_to_none=True)
#                 total_value_loss = self.get_max_value_loss([state_action_advantage_lst_episodes[idx] for idx in batch_idx])
                total_value_loss = self.get_value_loss([state_action_advantage_lst_episodes[idx] for idx in batch_idx], update_value_scale = itr == 0)
                value_curr_arr.append(float(total_value_loss.data))
                total_value_loss.backward()
                self.value_optimizer.step()
                self.value_scheduler.step()
#             plt.plot(value_curr_arr)
#             plt.title(f"Itr = {itr + 1}\nFinal Loss = {value_curr_arr[-1]}")
#             plt.savefig(f"DebugPlots/value_itr={itr+1}.png")
#             plt.clf()
#             plt.close()
            if debug:
                with open(debug_dir, "a") as f:
                    f.write(f"\tFinal Value Loss = {float(total_value_loss.data)}\n")
            
            ## Update policy models
            print("\tUpdating policy models...")
            policy_curr_arr = []
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
                            ratio, ratio_clipped = self.get_ratio(curr_state_counts_lst, action_id_lst, t, clipped = True, eps = eps)
                            loss_curr = -torch.min(ratio * advantage, ratio_clipped * advantage)
                            total_policy_loss += torch.sum(loss_curr) / len(batch_idx)
#                total_policy_loss /= len(batch_idx)
                policy_curr_arr.append(float(total_policy_loss.data))
                total_policy_loss.backward()
                self.policy_optimizer.step()
                self.policy_scheduler.step()
                policy_dct = None
#             plt.plot(policy_curr_arr)
#             plt.title(f"Itr = {itr + 1}\nFinal Loss = {policy_curr_arr[-1]}")
#             plt.savefig(f"DebugPlots/policy_itr={itr+1}.png")
#             plt.clf()
#             plt.close()
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
                payoff = 0
                df_table_all = None
                for i in tqdm(range(num_trials)):
                    _, _, payoff_lst, action_lst = self.evaluate(return_action = True, seed = None)
                    payoff += float(payoff_lst[-1].data)
                    df_table = report_factory.get_table(self.markov_decision_process, action_lst)
                    df_table["trial"] = i
                    if df_table_all is None:
                        df_table_all = df_table
                    else:
                        df_table_all = pd.concat([df_table_all, df_table], axis = 0)
                payoff /= num_trials
                df_table_all = df_table_all.groupby("trial").mean().reset_index()
                report_factory.visualize_table(df_table, f"{label}_itr={itr}", title = f"Total Payoff: {payoff:.2f}")
            
#            if return_payoff:
#                _, _, payoff_lst, _ = self.evaluate(return_action = False, seed = 0)
#                payoff_val = float(payoff_lst[-1].data)
#                payoff_arr.append(payoff_val)
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
    
    def policy_predict(self, state_counts, ts, prob = True, remove_infeasible = True, use_benchmark = False, lazy_removal = True, car_id = None, state_count_check = None):
        if not use_benchmark:
            output = self.policy_model((ts, state_counts))
        else:
            with torch.no_grad():
                output = self.benchmark_policy_model((ts, state_counts))
        if remove_infeasible:
            if len(state_counts.shape) == 1:
                ret = self.remove_infeasible_actions(state_counts.cpu(), ts, output, car_id = car_id, state_count_check = state_count_check)
#                print(self.markov_decision_process.describe_state_counts(state_count_check))
#                print(ret)
#                print("")
#                assert False
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
        mask = self.markov_decision_process.state_counts_to_potential_feasible_actions(self.state_reduction, state_count_check)
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
    
    def evaluate(self, seed = None, train = False, return_data = False, return_action = False, debug = False, debug_dir = "debugging_log.txt", lazy_removal = False, markov_decision_process = None):
        if True: #not train:
            self.value_model.eval()
            self.benchmark_policy_model.eval()
        if markov_decision_process is None:
            markov_decision_process = self.markov_decision_process
        return_data = return_data and train
        if seed is not None:
            torch.manual_seed(seed)
        markov_decision_process.reset_states()
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
            available_car_ids = markov_decision_process.get_available_car_ids(self.state_reduction)
            num_available_cars = len(available_car_ids)
            total_cars += num_available_cars
#            if not train:
#                print("t = ", t)
#                print(self.markov_decision_process.describe_state_counts())
            transit_applied = False
            for car_idx in tqdm(range(num_available_cars), leave = False):
                ## Perform state transitions
                curr_state_counts = markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx]).to(device = self.device)
#                if return_action:
                curr_state_counts_full = markov_decision_process.get_state_counts(deliver = True)
                action_id_prob = self.policy_predict(curr_state_counts, t, prob = True, use_benchmark = True, lazy_removal = lazy_removal, car_id = available_car_ids[car_idx], state_count_check = curr_state_counts_full)
                if action_id_prob is not None:
                    action_id_prob = action_id_prob.cpu().detach().numpy()
                    if debug:
                        with open(debug_dir, "a") as f:
                            msg = self.policy_describe(action_id_prob)
                            payoff = markov_decision_process.get_payoff_curr_ts().clone()
                            self.value_model.eval()
                            inferred_value = float(self.value_model((t, curr_state_counts)).data)
                            f.write(f"\tt = {t}, car_id = {car_idx}, payoff = {payoff}, inferred state value = {inferred_value}:\n")
                            f.write(self.markov_decision_process.describe_state_counts(curr_state_counts))
                            f.write(msg)
                    if True:#train:
                        is_feasible = False
                        while not is_feasible:
                            action_id = np.random.choice(len(action_id_prob), p = action_id_prob)
                            is_feasible = self.action_is_feasible(curr_state_counts, t, int(action_id), car_id = available_car_ids[car_idx], state_count_check = curr_state_counts_full)
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
                    next_state_counts = markov_decision_process.get_state_counts(state_reduction = self.state_reduction, car_id = available_car_ids[car_idx]).to(device = self.device)
                    payoff = markov_decision_process.get_payoff_curr_ts().clone()
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
                        curr_payoff = float(markov_decision_process.get_payoff_curr_ts(deliver = True))
#                        curr_value = self.value_model((t, curr))
                        payoff_lst.append(curr_payoff)
#                        if t == 0:
#                            if action.get_type() in ["pickup", "rerouting"]:
#                                print(action.get_origin(), action.get_dest(), curr_payoff)
#                        model_value_lst.append(curr_value)
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
        if return_data:
            final_payoff = float(markov_decision_process.get_payoff_curr_ts(deliver = True))
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
        self.plot_single(loss_arr, xlabel = "Policy Iterations", ylabel = type, title = loss_name + f"\nFinal {type} = {final_loss:.2f}", figname = figname, dir = "Plots")
    
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
        plt.legend()
        plt.title(title)
        plt.savefig(f"TablePlots/{figname}.png")
        plt.clf()
        plt.close()
    
    def visualize_table(self, df_table, suffix, title = "", detailed = False):
        ## Visualize new requests
        self.plot_single(df_table["num_new_requests"], xlabel = "Time Steps", ylabel = "# New Trip Requests", title = title, figname = f"new_requests_{suffix}", x_arr = df_table["t"], dir = "TablePlots")
        ## Visualize fulfilled requests
        self.plot_single(df_table["frac_requests_fulfilled"], xlabel = "Time Steps", ylabel = "% Fulfilled Trip Requests", title = title, figname = f"fulfilled_requests_{suffix}", x_arr = df_table["t"], dir = "TablePlots")
        ## Visualize car status
        self.plot_stacked(df_table["t"], [df_table["frac_traveling_cars"], df_table["frac_charging_cars"], df_table["frac_idling_cars"]], label_lst = ["% Traveling Cars", "% Charging Cars", "% Idling Cars"], xlabel = "Time Steps", ylabel = "% Cars", title = title, figname = f"car_status_{suffix}")
        ## Visualize trip status
        self.plot_stacked(df_table["t"], [df_table["num_fulfilled_requests"], df_table["num_queued_requests"], df_table["num_abandoned_requests"]], label_lst = ["# Fulfilled Requests", "# Queued Requests", "# Abandoned Requests"], xlabel = "Time Steps", ylabel = "# Requests", title = title, figname = f"trip_status_{suffix}")
        ## Visualize car battery status
        self.plot_stacked(df_table["t"], [df_table["frac_high_battery_cars"], df_table["frac_med_battery_cars"], df_table["frac_low_battery_cars"]], label_lst = ["% High Battery Cars", "% Med Battery Cars", "% Low Battery Cars"], xlabel = "Time Steps", ylabel = "% Cars", title = title, figname = f"battery_status_{suffix}")
        ## Visualize region supply & demand balancedness
        if detailed:
            regions = [int(x.split("_")[-1]) for x in df_table.columns if x.startswith("num_cars")]
            num_regions = np.max(regions) + 1
            for region in range(num_regions):
                num_trip_requests_region = df_table[f"num_trip_requests_{region}"]
                num_cars_region = df_table[f"num_cars_region_{region}"]
                self.plot_multi([num_trip_requests_region, num_cars_region], ["# Trip Requests", "# Available Cars"], "Time Steps", "Supply & Demand", "", f"supply_demand_{region}_{suffix}", dir = "TablePlots")
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
            self.plot_multi(y_lst, label_lst, "Time Steps", "# Charging Cars", "", f"region_charging_distribution_{suffix}", dir = "TablePlots")
            y_arr = np.array([np.sum(x) for x in y_lst])
            y_arr = y_arr / np.sum(y_arr)
            self.plot_bar([str(x) for x in range(num_regions)], y_arr, "Region", "% Charging Cars", "", f"region_charging_frac_{suffix}", dir = "TablePlots")
#            self.plot_stacked(df_table["t"], y_lst, label_lst = label_lst, xlabel = "Time Steps", ylabel = "% Cars", title = title, figname = f"region_charging_distribution_{suffix}")
    
    def get_table(self, markov_decision_process, action_lst, detailed = False):
        prev_t = -1
        begin = False
        num_active_requests_begin, num_traveling_cars_begin, num_idling_cars_begin, num_charging_cars_begin = 0, 0, 0, 0
        num_active_requests_end, num_traveling_cars_end, num_idling_cars_end, num_charging_cars_end = 0, 0, 0, 0
        num_new_requests = 0
        num_regions = len(markov_decision_process.regions)
        t_lst = []
        frac_requests_fulfilled_lst = []
        frac_traveling_cars_lst = []
        frac_charging_cars_lst = []
        frac_idling_cars_lst = []
        num_new_requests_lst = []
        num_fulfilled_requests_lst = []
        num_queued_requests_lst = []
        num_abandoned_requests_lst = []
        frac_low_battery_cars_lst = []
        frac_med_battery_cars_lst = []
        frac_high_battery_cars_lst = []
        num_trip_requests_dct = {}
        num_cars_dct = {}
        charging_car_dct = {}
        for i in range(num_regions):
            num_trip_requests_dct[i] = []
            num_cars_dct[i] = []
            charging_car_dct[i] = []
        for i in tqdm(range(len(action_lst)), leave = False):
            tup = action_lst[i]
            curr_state_counts, action, t, car_idx = tup
            if t > prev_t:
                begin = True
            else:
                begin = False
            prev_t = t
            if begin:
                num_active_requests_begin = markov_decision_process.get_num_active_trip_requests(curr_state_counts)
                num_new_requests = markov_decision_process.get_num_new_trip_requests(curr_state_counts)
                for region in range(num_regions):
                    if detailed:
                        num_trip_requests_region = markov_decision_process.get_num_trip_requests_region(region, state_counts = curr_state_counts)
                        num_cars_region = markov_decision_process.get_num_cars_region(region, state_counts = curr_state_counts)
                    else:
                        num_trip_requests_region = None
                        num_cars_region = None
                    num_trip_requests_dct[region].append(num_trip_requests_region)
                    num_cars_dct[region].append(num_cars_region)
            if car_idx is None:
                num_active_requests_end = markov_decision_process.get_num_active_trip_requests(curr_state_counts)
                num_traveling_cars_end = markov_decision_process.get_num_traveling_cars(curr_state_counts)
                num_idling_cars_end = markov_decision_process.get_num_idling_cars(curr_state_counts)
                num_charging_cars_end = markov_decision_process.get_num_charging_cars(curr_state_counts)
                num_total_cars = num_traveling_cars_end + num_idling_cars_end + num_charging_cars_end
                if num_active_requests_begin > 0:
                    frac_requests_fulfilled = (num_active_requests_begin - num_active_requests_end) / num_active_requests_begin
                else:
                    frac_requests_fulfilled = 1
                num_requests_fulfilled = num_active_requests_begin - num_active_requests_end
                num_requests_queued = markov_decision_process.get_num_queued_trip_requests(curr_state_counts)
                num_requests_abandoned = markov_decision_process.get_num_abandoned_trip_requests(curr_state_counts)
                num_low_battery_cars = markov_decision_process.get_num_cars_w_battery(curr_state_counts, "L")
                num_med_battery_cars = markov_decision_process.get_num_cars_w_battery(curr_state_counts, "M")
                num_high_battery_cars = markov_decision_process.get_num_cars_w_battery(curr_state_counts, "H")
                for region in range(num_regions):
                    if detailed:
                        num_charging_cars_region = markov_decision_process.get_num_charging_cars_region(region, state_counts = curr_state_counts)
                    else:
                        num_charging_cars_region = None
                    charging_car_dct[region].append(num_charging_cars_region)
                frac_traveling_cars = num_traveling_cars_end / num_total_cars
                frac_charging_cars = num_charging_cars_end / num_total_cars
                frac_idling_cars = num_idling_cars_end / num_total_cars
                t_lst.append(t)
                frac_requests_fulfilled_lst.append(frac_requests_fulfilled)
                frac_traveling_cars_lst.append(frac_traveling_cars)
                frac_charging_cars_lst.append(frac_charging_cars)
                frac_idling_cars_lst.append(frac_idling_cars)
                num_new_requests_lst.append(num_new_requests)
                num_fulfilled_requests_lst.append(num_requests_fulfilled)
                num_queued_requests_lst.append(num_requests_queued)
                num_abandoned_requests_lst.append(num_requests_abandoned)
                frac_low_battery_cars_lst.append(num_low_battery_cars / num_total_cars)
                frac_med_battery_cars_lst.append(num_med_battery_cars / num_total_cars)
                frac_high_battery_cars_lst.append(num_high_battery_cars / num_total_cars)
        dct = {"t": t_lst, "num_new_requests": num_new_requests_lst, "frac_requests_fulfilled": frac_requests_fulfilled_lst, "frac_traveling_cars": frac_traveling_cars_lst, "frac_charging_cars": frac_charging_cars_lst, "frac_idling_cars": frac_idling_cars_lst, "num_fulfilled_requests": num_fulfilled_requests_lst, "num_queued_requests": num_queued_requests_lst, "num_abandoned_requests": num_abandoned_requests_lst, "frac_low_battery_cars": frac_low_battery_cars_lst, "frac_med_battery_cars": frac_med_battery_cars_lst, "frac_high_battery_cars": frac_high_battery_cars_lst}
        for region in range(num_regions):
            dct[f"num_trip_requests_{region}"] = num_trip_requests_dct[region]
            dct[f"num_cars_region_{region}"] = num_cars_dct[region]
            dct[f"num_charging_cars_region_{region}"] = charging_car_dct[region]
        df = pd.DataFrame.from_dict(dct)
        return df

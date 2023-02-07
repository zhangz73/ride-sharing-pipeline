import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    def __init__(self, type = "sequential", markov_decision_process = None):
        assert type in ["sequential", "group"]
        self.type = type
        self.markov_decision_process = markov_decision_process
        ## Save some commonly used variables from MDP
        self.time_horizon = self.markov_decision_process.time_horizon
        self.reward_query = self.markov_decision_process.reward_query
        self.all_actions = self.markov_decision_process.get_all_actions()
        self.action_lst = [self.all_actions[k] for k in self.all_actions.keys()]
        self.payoff_map = self.markov_decision_process.payoff_map
        self.payoff_schedule_dct = self.markov_decision_process.payoff_schedule_dct
    
    def train(self, **kargs):
        return None

    ## Sequential Solvers: return an atomic action
    ## Group Solvers: return a list of actions and a list of corresponding car_type ids
    def predict(self, state_counts):
        return None

## This module is a child-class of Solver for the policy iteration solver
class PolicyIteration_Solver(Solver):
    def __init__(self, markov_decision_process = None, model_name = "discretized_feedforward", input_dim = 10, hidden_dim_lst = [10, 10], activation_lst = ["relu", "relu"], output_dim = 1, batch_norm = False, lr = 1e-2, decay = 0.1, scheduler_step = 10000, solver = "Adam", retrain = False, descriptor = None, dir = ".", device = "cpu", training_loss = "total_payoff", num_episodes = 100, ckpt_freq = 100):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process)
        assert training_loss in ["total_payoff", "surrogate"]
        self.training_loss = training_loss
        discretized_len = self.time_horizon
        input_dim = len(self.markov_decision_process.state_counts)
        output_dim = len(self.action_lst)
        self.model_factory = neural.ModelFactory(model_name, input_dim, hidden_dim_lst, activation_lst, output_dim, batch_norm, lr, decay, scheduler_step, solver, retrain, discretized_len, descriptor, dir, device)
        self.model = self.model_factory.get_model()
        self.optimizer, self.scheduler = self.model_factory.prepare_optimizer()
        self.metric_factory = MetricFactory()
        self.num_episodes = num_episodes
        self.ckpt_freq = ckpt_freq
        self.markov_decision_process_pg = copy.deepcopy(markov_decision_process)
    
    def get_model(self):
        return self.model_factory.get_model()
    
    def get_loss(self, total_payoff_lst):
        if self.training_loss == "total_payoff":
            return self.metric_factory.get_total_payoff_loss(total_payoff_lst)
        elif self.training_loss == "surrogate":
            return self.metric_factory.get_surrogate_loss(total_payoff_lst)
    
    def train(self):
        loss_arr = []
        for day in tqdm(range(self.num_episodes)):
            self.markov_decision_process.reset_states()
            self.optimizer.zero_grad()
            total_payoff_loss_lst = torch.zeros(self.time_horizon)
            for t in range(self.time_horizon):
                num_cars = self.markov_decision_process.get_available_car_counts()
                curr_action_lst = []
                total_payoff_loss = 0
                for car in range(num_cars):
                    state_counts = self.markov_decision_process.state_counts
                    output = self.predict(state_counts, t)
                    total_payoff_loss += torch.sum(self.payoff_map[t,:] * output)
                total_payoff_loss_lst[t] = total_payoff_loss
                self.markov_decision_process.transit_across_timestamp()
            ## Compute loss
            loss = self.get_loss(total_payoff_loss_lst)
            loss.backward()
            loss_arr.append(float(loss.data))
            self.optimizer.step()
            self.scheduler.step()
            ## Checkpoint
            if day > 0 and day % self.ckpt_freq == 0:
                self.model_factory.update_model(self.model, update_ts = False)
                descriptor = f"_episode={day}"
                self.model_factory.save_to_file(descriptor, include_ts = True)
        ## Save final model
        self.model_factory.update_model(self.model, update_ts = False)
        self.model_factory.save_to_file(descriptor = "", include_ts = True)
        return loss_arr
    
    def predict(self, state_counts, ts):
        model = self.get_model()
        output = model((ts, state_counts))
        #sorted_action_id_lst = torch.argsort(output, descending = True)
        ## Eliminate infeasible actions
        for action_id in range(len(output)):
            self.markov_decision_process_pg.set_states(state_counts, ts)
            action = self.all_actions[action_id]
            if not self.markov_decision_process_pg.transit_within_timestamp(action):
                output[action_id] = 0
        output = output / torch.sum(output)
        return output

class PPO_Solver(Solver):
    def __init__(self, markov_decision_process = None, value_model_name = "discretized_feedforward", value_hidden_dim_lst = [10, 10], value_activation_lst = ["relu", "relu"], value_batch_norm = False, value_lr = 1e-2, value_epoch = 1, value_batch = 100, value_decay = 0.1, value_scheduler_step = 10000, value_solver = "Adam", value_retrain = False, policy_model_name = "discretized_feedforward", policy_hidden_dim_lst = [10, 10], policy_activation_lst = ["relu", "relu"], policy_batch_norm = False, policy_lr = 1e-2, policy_epoch = 1, policy_batch = 100, policy_decay = 0.1, policy_scheduler_step = 10000, policy_solver = "Adam", policy_retrain = False, descriptor = "PPO", dir = ".", device = "cpu", num_itr = 100, num_episodes = 100, ckpt_freq = 100, benchmarking_policy = "uniform", eps = 0.2, policy_syncing_freq = 1):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process)
        ## Store some commonly used variables
        self.value_input_dim = len(self.markov_decision_process.state_counts)
        self.policy_input_dim = self.value_input_dim
        self.value_output_dim = 1
        self.policy_output_dim = len(self.action_lst)
        self.discretized_len = self.time_horizon
        self.num_itr = num_itr
        self.num_episodes = num_episodes
        self.ckpt_freq = ckpt_freq
        self.value_epoch = value_epoch
        self.policy_epoch = policy_epoch
        self.value_batch = value_batch
        self.policy_batch = policy_batch
        self.benchmarking_policy = benchmarking_policy
        self.eps = eps
        self.policy_syncing_freq = policy_syncing_freq
        ## Construct models
        self.value_model_factory = neural.ModelFactory(value_model_name, self.value_input_dim, value_hidden_dim_lst, value_activation_lst, self.value_output_dim, value_batch_norm, value_lr, value_decay, value_scheduler_step, value_solver, value_retrain, self.discretized_len, descriptor + "_value", dir, device)
        self.policy_model_factory = neural.ModelFactory(policy_model_name, self.policy_input_dim, policy_hidden_dim_lst, policy_activation_lst, self.policy_output_dim, policy_batch_norm, policy_lr, policy_decay, policy_scheduler_step, policy_solver, policy_retrain, self.discretized_len, descriptor + "_policy", dir, device, prob = True)
        self.value_model = self.get_value_model()
        self.policy_model = self.get_policy_model()
        self.benchmark_policy_model = copy.deepcopy(self.policy_model)
        self.value_optimizer, self.value_scheduler = self.value_model_factory.prepare_optimizer()
        self.policy_optimizer, self.policy_scheduler = self.policy_model_factory.prepare_optimizer()
        self.markov_decision_process_pg = copy.deepcopy(markov_decision_process)
    
    def get_value_model(self):
        return self.value_model_factory.get_model()
    
    def get_policy_model(self):
        return self.policy_model_factory.get_model()
    
    def get_advantange(self, curr_state_counts, next_state_counts, action_id, ts, next_ts):
        payoff = self.payoff_map[ts, action_id]
        curr_value = self.value_model((ts, curr_state_counts)).reshape((-1,))
        ## TODO: Fix the bug in ts
        if next_ts < self.time_horizon - 1:
            next_value = self.value_model((next_ts, next_state_counts)).reshape((-1,))
        else:
            next_value = 0
        return payoff + next_value - curr_value
    
    def get_ratio(self, state_counts, action_id, ts, clipped = False, eps = 0.2):
        prob_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = True)
        action_id = action_id.reshape((len(action_id), 1))
        prob = prob_output.gather(1, action_id) #prob_output[action_id]
        prob_benchmark_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = True, use_benchmark = True)
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
    
    def get_ratio_single(self, state_counts, action_id, ts, clipped = False, eps = 0.2):
        prob_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = True)
        prob = prob_output[action_id]
        prob_benchmark_output = self.policy_predict(state_counts, ts, prob = True, remove_infeasible = True, use_benchmark = True)
#         prob_benchmark_output = self.policy_benchmark_predict(state_counts, ts, prob = True, remove_infeasible = False)
        prob_benchmark = prob_benchmark_output[action_id]
        if prob > 0:
            ratio = prob / prob_benchmark
            if clipped:
                clipped_ratio = torch.min(torch.max(ratio, torch.tensor(1 - eps)), torch.tensor(1 + eps))
            else:
                clipped_ratio = ratio
        else:
            ratio, clipped_ratio = 0
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
#                payoff += self.payoff_map[t, int(action_id)]
                payoff = final_payoff - curr_payoff
                value_model_output = self.value_model((t, curr_state_counts))
                total_value_loss += (value_model_output - payoff) ** 2
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
                curr_state_counts, action_id, _, t, curr_payoff, _ = tup
#                payoff += self.payoff_map[t, int(action_id)]
                payoff = final_payoff - curr_payoff
                lens = len(curr_state_counts)
                value_dct[t]["payoff"].append(payoff)
                value_dct[t]["state_counts"].append(curr_state_counts.reshape((1, lens)))
                val_num += 1
        for t in range(self.time_horizon):
            payoff_lst = torch.tensor(value_dct[t]["payoff"])
            state_counts_lst = torch.cat(value_dct[t]["state_counts"], dim = 0)
            value_model_output = self.value_model((t, state_counts_lst)).reshape((-1,))
            total_value_loss += torch.sum((value_model_output - payoff_lst) ** 2)
        total_value_loss /= val_num #self.num_episodes
        return total_value_loss
    
    def get_max_value_loss(self, state_action_advantage_lst_episodes, debug = False, dct = {}):
        total_value_loss = 0
        for day in range(self.value_batch):
            state_num = len(state_action_advantage_lst_episodes[day])
#            payoff = 0
            if state_num > 0:
                final_payoff = state_action_advantage_lst_episodes[day][state_num - 1][4].clone()
                for i in range(state_num - 1, -1, -1):
                    tup = state_action_advantage_lst_episodes[day][i]
                    curr_state_counts, action_id, _, t, curr_payoff = tup
                    action = self.all_actions[int(action_id)]
                    #payoff += self.payoff_map[t, int(action_id)]
                    payoff = final_payoff - curr_payoff
                    key = tuple(curr_state_counts.numpy())
                    if key not in dct:
                        dct[key] = payoff
                    else:
                        curr_payoff = dct[key]
                        max_payoff = torch.max(payoff, dct[key])
                        dct[key] = max_payoff
        for key in dct:
            payoff = dct[key]
            t = int(key[-1])
            curr_state_counts = torch.tensor(list(key))
            value_model_output = self.value_model((t, curr_state_counts))
            total_value_loss += (value_model_output - payoff) ** 2
        total_value_loss /= len(dct.keys())
        return total_value_loss
    
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
        for itr in range(self.num_itr):
            print(f"Iteration #{itr}:")
            if debug:
                with open(debug_dir, "a") as f:
                    f.write(f"Itr = {itr}:\n")
            ## Obtain simulated data from each episode
            state_action_advantage_lst_episodes = []
            print("\tGathering data...")
            for day in tqdm(range(self.num_episodes)):
#                value_loss, policy_loss, _ = self.evaluate(train = True)
#                total_value_loss += value_loss / self.num_episodes
#                total_policy_loss += policy_loss / self.num_episodes
                state_action_advantage_lst = self.evaluate(train = True, return_data = True, explore = False, debug = debug and day == 0, debug_dir = debug_dir)
                state_action_advantage_lst_episodes.append(state_action_advantage_lst)
            ## Update models
            ## Update value models
            print("\tUpdating value models...")
            for _ in tqdm(range(self.value_epoch)):
                batch_idx = np.random.choice(self.num_episodes, size = self.value_batch, replace = False)
                self.value_optimizer.zero_grad()
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
            policy_dct = {}
            for t in range(self.time_horizon):
                for offset in [0, 1]:
                    tup = (t, t + offset)
                    policy_dct[tup] = {"curr_state_counts": [], "next_state_counts": [], "action_id": []}
                
            for _ in tqdm(range(self.policy_epoch)):
                batch_idx = np.random.choice(self.num_episodes, size = self.policy_batch, replace = False)
                total_policy_loss = 0
                self.policy_optimizer.zero_grad()
                for day in batch_idx:
                    state_num = len(state_action_advantage_lst_episodes[day])
                    for i in range(state_num):
                        tup = state_action_advantage_lst_episodes[day][i]
                        curr_state_counts, action_id, next_state_counts, t, _, next_t = tup
                        lens = len(curr_state_counts)
                        policy_dct[(t, next_t)]["curr_state_counts"].append(curr_state_counts.reshape((1, lens)))
                        policy_dct[(t, next_t)]["next_state_counts"].append(next_state_counts.reshape((1, lens)))
                        policy_dct[(t, next_t)]["action_id"].append(action_id)
#                        advantage = self.get_advantange(curr_state_counts, next_state_counts, action_id, t, next_t)
#                        ratio, ratio_clipped = self.get_ratio(curr_state_counts, action_id, t, clipped = True, eps = self.eps)
#                        loss_curr = -torch.min(ratio * advantage, ratio_clipped * advantage)
#                        total_policy_loss += loss_curr[0]
                for t in range(self.time_horizon):
                    for offset in [0, 1]:
                        next_t = t + offset
                        if len(policy_dct[(t, next_t)]["curr_state_counts"]) > 0:
                            curr_state_counts_lst = torch.cat(policy_dct[(t, next_t)]["curr_state_counts"], dim = 0)
                            next_state_counts_lst = torch.cat(policy_dct[(t, next_t)]["next_state_counts"], dim = 0)
                            action_id_lst = torch.tensor(policy_dct[(t, next_t)]["action_id"])
                            advantage = self.get_advantange(curr_state_counts_lst, next_state_counts_lst, action_id_lst, t, next_t)
                            ratio, ratio_clipped = self.get_ratio(curr_state_counts_lst, action_id_lst, t, clipped = True, eps = self.eps)
                            loss_curr = -torch.min(ratio * advantage, ratio_clipped * advantage)
                            total_policy_loss += torch.sum(loss_curr)
                total_policy_loss /= len(batch_idx)
                total_policy_loss.backward()
                self.policy_optimizer.step()
                self.policy_scheduler.step()
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
                _, _, payoff_lst, _ = self.evaluate(return_action = True, seed = 0)
                payoff_val = float(payoff_lst[-1].data)
                payoff_arr.append(payoff_val)
        self.benchmark_policy_model = copy.deepcopy(self.policy_model)
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
    
    def policy_predict(self, state_counts, ts, prob = True, remove_infeasible = True, use_benchmark = False):
        if not use_benchmark:
            output = self.policy_model((ts, state_counts))
        else:
            output = self.benchmark_policy_model((ts, state_counts))
        if remove_infeasible:
            if len(state_counts.shape) == 1:
                ret = self.remove_infeasible_actions(state_counts, ts, output)
            else:
                ret_lst = []
                for i in range(state_counts.shape[0]):
                    ret = self.remove_infeasible_actions(state_counts[i,:], ts, output[i,:])
                    ret_lst.append(ret.reshape((1, len(ret))))
                ret = torch.cat(ret_lst, dim = 0)
            output = output * ret
        if torch.sum(output) == 0:
            return None
        output = output / torch.sum(output)
        if prob:
            return output
        return torch.argmax(output, dim = 1).unsqueeze(-1)
    
    def remove_infeasible_actions(self, state_counts, ts, output):
        ## Eliminate infeasible actions
        ret = torch.ones(len(output))
        for action_id in range(len(output)):
            self.markov_decision_process_pg.set_states(state_counts, ts)
            action = self.all_actions[action_id]
            if not self.markov_decision_process_pg.transit_within_timestamp(action):
                ret[action_id] = 0
        return ret
    
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
    
    def evaluate(self, seed = None, train = False, return_data = False, return_action = False, explore = False, debug = False, debug_dir = "debugging_log.txt"):
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
        for t in range(self.time_horizon):
            ## Add up ||v_model - v_hat||^2
            ## Add up ratio * advantage
            num_available_cars = self.markov_decision_process.get_available_car_counts()
            transit_applied = False
            for car_idx in range(num_available_cars):
                ## Perform state transitions
                curr_state_counts = self.markov_decision_process.state_counts.clone()
                action_id_prob = self.policy_predict(curr_state_counts, t, prob = True, use_benchmark = True)
                if action_id_prob is not None:
                    action_id_prob = action_id_prob.detach().numpy()
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
                        if not explore:
                            action_id = np.random.choice(len(action_id_prob), p = action_id_prob)
                        else:
                            prob_arr = action_id_prob > 0
                            prob_arr = prob_arr / np.sum(prob_arr)
                            action_id = np.random.choice(len(action_id_prob), p = prob_arr)
                    else:
                        action_id = np.argmax(action_id_prob)
                    action = self.all_actions[int(action_id)]
                    if return_action:
                        action_lst.append((curr_state_counts, action, t, car_idx))
                    curr_payoff = self.markov_decision_process.get_payoff_curr_ts().clone()
                    res = self.markov_decision_process.transit_within_timestamp(action)
                    next_t = t
                    if car_idx == num_available_cars - 1:
                        transit_applied = True
                        self.markov_decision_process.transit_across_timestamp()
                        next_t += 1
                    next_state_counts = self.markov_decision_process.state_counts.clone()
                    payoff = self.markov_decision_process.get_payoff_curr_ts().clone()
                    if return_data: # and t < self.time_horizon - 1:
                        state_action_advantage_lst.append((curr_state_counts, action_id, next_state_counts, t, curr_payoff, next_t))
                    ## Compute loss
                    if not return_data:
                        if car_idx < num_available_cars - 1:
                            next_t = t
                        else:
                            next_t = t + 1
                        advantage = self.get_advantange(curr_state_counts, next_state_counts, action_id, t, next_t)
                        ratio, ratio_clipped = self.get_ratio_single(curr_state_counts, action_id, t, clipped = False)
                        loss_curr = -torch.min(ratio * advantage, ratio_clipped * advantage)
                        policy_loss += loss_curr[0]
                        ## Compute values
                        curr_payoff = float(self.markov_decision_process.get_payoff_curr_ts()) #self.payoff_map[t, int(action_id)]
                        curr_value = self.value_model((t, curr_state_counts))
                        payoff_lst.append(curr_payoff)
                        model_value_lst.append(curr_value)
            if not transit_applied:
                self.markov_decision_process.transit_across_timestamp()
        if not return_data:
            payoff_lst = torch.tensor(payoff_lst)
#            payoff_lst[-1] = 0
#            empirical_value_lst = torch.cumsum(payoff_lst.flip(0), 0).flip(0)
            payoff_lst[-1] = payoff_lst[-2]
            empirical_value_lst = payoff_lst
            value_loss = 0
            assert len(empirical_value_lst) == len(model_value_lst)
            for t in range(len(model_value_lst) - 1, -1, -1):
                value_loss += torch.sum((model_value_lst[t] - empirical_value_lst[t]) ** 2)
            #model_value_lst = torch.tensor(model_value_lst)
            #value_loss = torch.sum((model_value_lst - empirical_value_lst) ** 2)
        if return_data:
            return state_action_advantage_lst
        return value_loss, policy_loss, payoff_lst, action_lst

## This module is a child-class of Solver for the dynamic programming solver
class DP_Solver(Solver):
    def __init__(self, markov_decision_process = None):
        super().__init__(type = "group", markov_decision_process = markov_decision_process)
        ## Auxiliary variables
        self.feasible_state_transitions = {}
        ## A list of length time_horizon, storing state_counts
        self.optimal_states = []
        ## A list of length time_horizon, storing (car_type_id_lst, action_lst)
        ##  The last index being ([], [])
        self.optimal_atomic_actions = []
        ## A list of length time_horizon, with the last one being 0
        self.optimal_values = []
    
    ## Use Dynamic Programming to find the best actions given states
    ## Currently only support setting with small state space and short time horizon
    ## Procedures:
    ##      1. Build a graph of feasible state transition (depth = time horizon, edge = payoff)
    ##      2. Assign values of 0 to the last layer
    ##      3. Traverse the graph backward to find the optimal matches
    ##      4. Convert state transitions to car-atomic action matching
    def train(self):
        ## Build graph
        self.build_graph()
        ## Assign values of 0 to the last layer
        for state_counts in self.feasible_state_transitions[self.time_horizon - 1]:
            self.feasible_state_transitions[self.time_horizon - 1][state_counts]["value"] = 0
        ## Traverse the graph backward and populate max values
        for t in range(self.time_horizon - 2, -1, -1):
            for curr_state_counts in self.feasible_state_transitions[t]:
                max_value = -np.inf
                opt_action = None
                for tup in self.feasible_state_transitions[t][curr_state_counts]:
                    if tup not in ["back_pointers"]:
                        next_state_counts, curr_payoff = self.feasible_state_transitions[t][curr_state_counts][tup]
                        next_value = self.feasible_state_transitions[t + 1][next_state_counts]["value"]
    #                     final_payoff = self.feasible_state_transitions[t + 1][next_state_counts]["final_payoff"]
    #                     atomic_payoff = final_payoff - curr_payoff
                        curr_value = curr_payoff + next_value
                        if curr_value >= max_value:
                            max_value = curr_value
                            opt_action = tup
                self.feasible_state_transitions[t][curr_state_counts]["value"] = max_value
                self.feasible_state_transitions[t][curr_state_counts]["opt_action"] = opt_action
        ## Traverse the graph forward record the optimal states & atomic actions
        ## Assume the initial timestamp starts with only 1 possible state counts
        for key in self.feasible_state_transitions[0]:
            self.optimal_states.append(key)
        for t in range(self.time_horizon - 1):
            opt_state = self.optimal_states[-1]
            opt_action = self.feasible_state_transitions[t][opt_state]["opt_action"]
            next_state = self.feasible_state_transitions[t][opt_state][opt_action][0]
            opt_value = self.feasible_state_transitions[t][opt_state]["value"]
            self.optimal_states.append(next_state)
            self.optimal_atomic_actions.append(opt_action)
            self.optimal_values.append(opt_value)
        self.optimal_values.append(0)
    
    ## Construct a graph of feasible state transitions
    def build_graph(self):
        self.feasible_state_transitions[0] = {tuple(self.markov_decision_process.state_counts.numpy()): {}}
        ## Iterate through timestamps
        for t in range(1, self.time_horizon):
            self.feasible_state_transitions[t] = {}
            ## Iterate through all state values in the previous timestamp
            for prev_state_counts in self.feasible_state_transitions[t - 1]:
                self.markov_decision_process.set_states(torch.tensor(prev_state_counts), t - 1)
                available_car_id_lst = self.markov_decision_process.get_all_available_existing_car_types()
                available_car_cnts = [prev_state_counts[id] for id in available_car_id_lst]
                car_idx = 0
                tmp = [([], [], torch.tensor(prev_state_counts), 0)]
                ## Apply atomic actions to each available car
                while car_idx < len(available_car_id_lst) and available_car_cnts[car_idx] > 0:
                    ## Apply the action to each of the scenarios
                    ## I.e. Construct an action graph of cars at the current timestamp
                    for tup in tmp:
                        car_type_id_lst_tmp, action_lst_tmp, curr_state_counts, payoff = tup[0], tup[1], tup[2], tup[3]
                        curr = []
                        ## Try all feasible actions
                        for action in self.action_lst:
                            self.markov_decision_process.set_states(curr_state_counts, t - 1)
                            is_feasible = self.markov_decision_process.transit_within_timestamp(action, available_car_id_lst[car_idx])
                            if is_feasible:
                                next_state_counts = self.markov_decision_process.state_counts
                                car_type_id_lst_curr = list(car_type_id_lst_tmp) + [car_idx]
                                action_lst_curr = list(action_lst_tmp) + [action]
                                payoff_curr = self.markov_decision_process.get_payoff_curr_ts().clone()
                                curr.append((car_type_id_lst_curr, action_lst_curr, next_state_counts, payoff_curr))
                        tmp = curr
                    available_car_cnts[car_idx] -= 1
                    if available_car_cnts[car_idx] == 0:
                        car_idx += 1
                ## Load the state values after applying feasible actions to all available cars
                for tup in tmp:
                    val = tup[2]
                    key = (tup[0], tup[1])
                    payoff = tup[3]
                    action_id_lst = [self.markov_decision_process.action_to_id[x] for x in tup[1]]
                    key = (tuple(tup[0]), tuple(action_id_lst))
                    payoff_before = payoff
                    self.markov_decision_process.set_states(val, t - 1, payoff) # payoff
                    self.markov_decision_process.transit_across_timestamp()
                    ## Record the current payoff
                    payoff = float(self.markov_decision_process.get_payoff_curr_ts()) #self.reward_query.atomic_actions_to_payoff(list(tup[1]), t - 1)
                    ## Zero out the payoff
                    self.markov_decision_process.reset_payoff_curr_ts()
                    curr_state_counts = self.markov_decision_process.state_counts.numpy()
                    if key not in self.feasible_state_transitions[t - 1][tuple(prev_state_counts)]:
                        self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key] = (tuple(curr_state_counts), payoff)
                    else:
                        prev_payoff = self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key][1]
                        if payoff > prev_payoff:
                            self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key] = (tuple(curr_state_counts), payoff)
                    if tuple(curr_state_counts) not in self.feasible_state_transitions[t]:
                        self.feasible_state_transitions[t][tuple(curr_state_counts)] = {"back_pointers": None}
                    backpointer_tup = self.feasible_state_transitions[t][tuple(curr_state_counts)]["back_pointers"]
                    if backpointer_tup is None or payoff > backpointer_tup[3]:
                        self.feasible_state_transitions[t][tuple(curr_state_counts)]["back_pointers"] = (tuple(prev_state_counts), key[0], key[1], payoff)
                if len(tmp) == 0:
                    ## Load the state value after applying no actions at all
                    self.markov_decision_process.set_states(torch.tensor(prev_state_counts), t - 1, 0)
                    self.markov_decision_process.transit_across_timestamp()
                    payoff = float(self.markov_decision_process.get_payoff_curr_ts())
                    ## Zero out the payoff
                    self.markov_decision_process.reset_payoff_curr_ts()
                    curr_state_counts = self.markov_decision_process.state_counts.numpy()
                    key = ((),())
                    if key not in self.feasible_state_transitions[t - 1][tuple(prev_state_counts)]:
                        self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key] = (tuple(curr_state_counts), payoff)
                    else:
                        prev_payoff = self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key][1]
                        if payoff > prev_payoff:
                            self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key] = (tuple(curr_state_counts), payoff)
                    if tuple(curr_state_counts) not in self.feasible_state_transitions[t]:
                        self.feasible_state_transitions[t][tuple(curr_state_counts)] = {"back_pointers": None}
                    backpointer_tup = self.feasible_state_transitions[t][tuple(curr_state_counts)]["back_pointers"]
                    if backpointer_tup is None or payoff > backpointer_tup[3]:
                        self.feasible_state_transitions[t][tuple(curr_state_counts)]["back_pointers"] = (tuple(prev_state_counts), key[0], key[1], payoff)
    
    ## Return a tuple of (car_type_id_lst, action_lst)
    def predict(self, state_counts):
        ts_id = self.markov_decision_process.state_to_id["timestamp"]
        ts = state_counts[ts_id]
        return self.optimal_atomic_actions[ts]
    
    ## Print a readable version of the constructed feasible state transition graph
    ## Mostly for debugging purpose
    def describe_feasible_state_transitions(self):
        for t in range(self.time_horizon - 1):
            print(f"t = {t}:")
            for curr_state_counts in self.feasible_state_transitions[t]:
                opt_action_tup = self.feasible_state_transitions[t][curr_state_counts]["opt_action"]
                if opt_action_tup is not None and len(opt_action_tup[1]) > 0:
                    opt_action_id = opt_action_tup[1][0]
                    opt_value = self.feasible_state_transitions[t][curr_state_counts]["value"]
                    action = self.all_actions[opt_action_id]
                    print(action.describe(), f"action_id = {opt_action_id}, max_value = {opt_value}")
                else:
                    print("None")
                print(self.markov_decision_process.describe_state_counts(torch.tensor(curr_state_counts)))
                

## This module is a child-class of Solver for the greedy solver
## TODO: Define the exact behaviors of the greedy solver
class Greedy_Solver(Solver):
    def __init__(self, markov_decision_process = None):
        super().__init__(type = "group", markov_decision_process = markov_decision_process)

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

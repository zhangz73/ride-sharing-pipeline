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
        return torch.sum(total_payoff_lst)
    
    def get_total_payoff_loss(self, total_payoff_lst):
        return -torch.sum(total_payoff_lst)

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
    
    def train(self, **kargs):
        return None

    ## Sequential Solvers: return an atomic action
    ## Group Solvers: return a list of actions and a list of corresponding car_type ids
    def predict(self, state_counts):
        return None

## This module is a child-class of Solver for the reinforcement learning solver
class RL_Solver(Solver):
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
        self.payoff_map = torch.zeros((self.time_horizon, len(self.action_lst)))
        self.construct_payoff_map()
        self.markov_decision_process_pg = copy.deepcopy(markov_decision_process)
    
    def get_model(self):
        return self.model_factory.get_model()
    
    def get_loss(self, total_payoff_lst):
        if self.training_loss == "total_payoff":
            return self.metric_factory.get_total_payoff_loss(total_payoff_lst)
        elif self.training_loss == "surrogate":
            return self.metric_factory.get_surrogate_loss(total_payoff_lst)
    
    def construct_payoff_map(self):
        for t in range(self.time_horizon):
            for action_id in self.all_actions:
                action = self.all_actions[action_id]
                self.payoff_map[t, action_id] = self.reward_query.query(action, t)
    
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
#                    sorted_action_id_lst = self.predict(state_counts, t)
#                    feasible_action_applied = False
#                    action_idx = 0
#                    while action_idx < len(sorted_action_id_lst) and not feasible_action_applied:
#                        action_id = sorted_action_id_lst[action_idx]
#                        action = self.all_actions[int(action_id)]
#                        feasible_action_applied = self.markov_decision_process.transit_within_timestamp(action)
#                        action_idx += 1
#                        if feasible_action_applied:
#                            curr_action_lst.append(action)
#                            total_payoff_loss += self.payoff_map[t, action_id]
#                    assert feasible_action_applied
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
                max_value = 0
                opt_action = None
                for tup in self.feasible_state_transitions[t][curr_state_counts]:
                    next_state_counts, curr_payoff = self.feasible_state_transitions[t][curr_state_counts][tup]
                    next_value = self.feasible_state_transitions[t + 1][next_state_counts]["value"]
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
                tmp = [([], [], torch.tensor(prev_state_counts))]
                ## Apply atomic actions to each available car
                while car_idx < len(available_car_id_lst) and available_car_cnts[car_idx] > 0:
                    ## Apply the action to each of the scenarios
                    ## I.e. Construct an action graph of cars at the current timestamp
                    for tup in tmp:
                        car_type_id_lst_tmp, action_lst_tmp, curr_state_counts = tup[0], tup[1], tup[2]
                        curr = []
                        ## Try all feasible actions
                        for action in self.action_lst:
                            self.markov_decision_process.set_states(curr_state_counts, t - 1)
                            is_feasible = self.markov_decision_process.transit_within_timestamp(action, available_car_id_lst[car_idx])
                            if is_feasible:
                                next_state_counts = self.markov_decision_process.state_counts
                                car_type_id_lst_curr = list(car_type_id_lst_tmp) + [car_idx]
                                action_lst_curr = list(action_lst_tmp) + [action]
                                curr.append((car_type_id_lst_curr, action_lst_curr, next_state_counts))
                        tmp = curr
                    available_car_cnts[car_idx] -= 1
                    if available_car_cnts[car_idx] == 0:
                        car_idx += 1
                ## Load the state values after applying feasible actions to all available cars
                for tup in tmp:
                    val = tup[2]
                    key = (tup[0], tup[1])
                    action_id_lst = [self.markov_decision_process.action_to_id[x] for x in tup[1]]
                    key = (tuple(tup[0]), tuple(action_id_lst))
                    self.markov_decision_process.set_states(val, t - 1)
                    self.markov_decision_process.transit_across_timestamp()
                    ## Record the current payoff
                    payoff = self.reward_query.atomic_actions_to_payoff(list(tup[1]), t - 1)
                    curr_state_counts = self.markov_decision_process.state_counts.numpy()
                    if key not in self.feasible_state_transitions[t - 1][tuple(prev_state_counts)]:
                        self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key] = (tuple(curr_state_counts), payoff)
                    else:
                        prev_payoff = self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key][1]
                        if payoff > prev_payoff:
                            self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key] = (tuple(curr_state_counts), payoff)
                    self.feasible_state_transitions[t][tuple(curr_state_counts)] = {}
    
    ## Return a tuple of (car_type_id_lst, action_lst)
    def predict(self, state_counts):
        ts_id = self.markov_decision_process.state_to_id["timestamp"]
        ts = state_counts[ts_id]
        return self.optimal_atomic_actions[ts]

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
        plt.plot(loss_arr)
        plt.xlabel("Training Episodes")
        plt.ylabel("Loss")
        plt.title(loss_name)
        plt.savefig(f"Plots/{figname}.png")
        plt.clf()
        plt.close()
    
    def get_table(self):
        pass

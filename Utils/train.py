import numpy as np
import torch

## This module computes different types of losses and performance metrics
class MetricFactory:
    def __init__(self):
        pass
    
    def get_surrogate_loss(self):
        pass

    def get_total_payoff(self):
        pass

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
    def __init__(self, type = "sequential", markov_decision_process = None, action_lst = None):
        assert type in ["sequential", "group"]
        self.type = type
        self.markov_decision_process = markov_decision_process
        self.action_lst = action_lst
    
    def train(self, **kargs):
        return None

    ## Sequential Solvers: return an atomic action
    ## Group Solvers: return a list of actions and a list of corresponding car_type ids
    def predict(self, state_counts):
        return None

## This module is a child-class of Solver for the reinforcement learning solver
def RL_Solver(Solver):
    def __init__(self, markov_decision_process = None, action_dict = None):
        super().__init__(type = "sequential", markov_decision_process = markov_decision_process, action_lst = action_lst)
    
    def prepare_model(self):
        pass
    
    def train(self, num_episodes = 100, ckpt_freq = 100):
        pass
    
    def predict(self, state_counts):
        pass

## This module is a child-class of Solver for the dynamic programming solver
def DP_Solver(Solver):
    def __init__(self, markov_decision_process = None, action_dict = None):
        super().__init__(type = "group", markov_decision_process = markov_decision_process, action_lst = action_lst)
        ## Save some commonly used variables from MDP
        self.time_horizon = self.markov_decision_process.time_horizon
        self.reward_query = self.markov_decision_process.reward_query
        ## Auxiliary variables
        self.feasible_state_transitions = {}
        ## A list of length time_horizon, storing state_counts
        self.optimal_states = []
        ## A list of length time_horizon, storing (car_type_id_lst, action_lst)
        ##  The last index being ([], [])
        self.optimal_atomic_actions = []
    
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
        ## Traverse the graph backward and populate payoffs
        for t in range(self.time_horizon - 2, -1, -1):
            for curr_state_counts in self.feasible_state_transitions[t]:
                max_value = 0
                for tup in self.feasible_state_transitions[t][curr_state_counts]:
                    next_state_counts, curr_payoff = self.feasible_state_transitions[t][curr_state_counts][tup]
                    next_value = self.feasible_state_transitions[t + 1][next_state_counts]["value"]
                    curr_value = curr_payoff + next_value
                    if curr_value > max_value:
                        max_value = curr_value
                        self.feasible_state_transitions[t][curr_state_counts]["value"] = max_value
                        self.feasible_state_transitions[t][curr_state_counts]["opt_action"] = tup
        ## Traverse the graph forward record the optimal states & atomic actions
        ## Assume the initial timestamp starts with only 1 possible state counts
        for key in self.feasible_state_transitions[0]:
            self.optimal_states.append(self.feasible_state_transitions[0][key])
        for t in range(self.time_horizon - 1):
            opt_state = self.optimal_states[-1]
            opt_action = self.feasible_state_transitions[t][opt_state]["opt_action"]
            next_state = self.feasible_state_transitions[t][opt_state][opt_action][0]
            self.optimal_states.append(next_state)
            self.optimal_atomic_actions.append(opt_action)
    
    ## Construct a graph of feasible state transitions
    def build_graph(self):
        self.feasible_state_transitions[0] = {tuple(self.markov_decision_process.state_counts): {}}
        ## Iterate through timestamps
        for t in range(1, self.time_horizon):
            self.feasible_state_transitions[t] = {}
            ## Iterate through all state values in the previous timestamp
            for prev_state_counts in self.feasible_state_transitions[t - 1]:
                self.markov_decision_process.set_state_counts(prev_state_counts, t - 1)
                available_car_id_lst = self.markov_decision_process.get_all_available_existing_car_types()
                available_car_cnts = [self.state_counts[id] for id in available_car_id_lst]
                car_idx = 0
                tmp = {([], []): prev_state_counts}
                ## Apply atomic actions to each available car
                while car_idx < len(available_car_id_lst) and available_car_cnts[car_idx] > 0:
                    ## Apply the action to each of the scenarios
                    ## I.e. Construct an action graph of cars at the current timestamp
                    for car_type_id_lst_tmp, action_lst_tmp in tmp:
                        curr_state_counts = tmp[(car_type_id_lst_tmp, action_lst_tmp)]
                        self.markov_decision_process.set_state_counts(curr_state_counts, t - 1)
                        curr = {}
                        ## Try all feasible actions
                        for action in self.action_lst:
                            is_feasible = self.markov_decision_process.transit_within_timestamp(action, available_car_id_lst[car_idx])
                            next_state_counts = self.markov_decision_process.state_counts
                            car_type_id_lst_curr = car_type_id_lst_tmp + [car_idx]
                            action_lst_curr = action_lst_tmp + [action]
                            curr[(car_type_id_lst_curr, action_lst_curr)] = next_state_counts
                        tmp = curr
                        available_car_cnts[car_idx] -= 1
                        if available_car_cnts[car_idx] == 0:
                            car_idx += 1
                ## Load the state values after applying feasible actions to all available cars
                for key in tmp:
                    val = tmp[key]
                    self.markov_decision_process.set_state_counts(val, t - 1)
                    self.markov_decision_process.transit_across_timestamp()
                    ## Record the current payoff
                    payoff = self.atomic_actions_to_payoff(val[1], t - 1)
                    curr_state_counts = self.markov_decision_process.state_counts
                    self.feasible_state_transitions[t - 1][tuple(prev_state_counts)][key] = (tuple(curr_state_counts), payoff)
                    self.feasible_state_transitions[t][tuple(curr_state_counts)] = {}
    
    ## Compute the total payoff given a list of atomic actions and a timestamp
    def atomic_actions_to_payoff(self, action_lst, ts):
        total_payoff = 0
        for action in action_lst:
            curr_payoff = self.reward_query.query(action, ts)
            total_payoff += curr_payoff
        return total_payoff
    
    ## Return a tuple of (car_type_id_lst, action_lst)
    def predict(self, state_counts):
        ts_id = self.markov_decision_process.state_to_id["timestamp"]
        ts = state_counts[ts_id]
        return self.optimal_atomic_actions[ts]

## This module is a child-class of Solver for the greedy solver
## TODO: Define the exact behaviors of the greedy solver
def Greedy_Solver(Solver):
    def __init__(self, markov_decision_process = None, action_dict = None):
        super().__init__(type = "group", markov_decision_process = markov_decision_process, action_lst = action_lst)

## This module generate plots and tables
class ReportFactory:
    def __init__(self):
        pass
    
    def get_plot(self):
        pass
    
    def get_table(self):
        pass

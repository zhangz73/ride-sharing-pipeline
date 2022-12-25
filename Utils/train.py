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
    def __init__(self):
        pass

    def construct_rl_solver(self):
        pass
    
    def construct_dp_solver(self):
        pass
    
    def construct_greedy_solver(self):
        pass
    
    def train(self):
        pass

    def predict(self):
        pass

## This module is a child-class of Solver for the reinforcement learning solver
def RL_Solver(Solver):
    def __init__(self):
        pass

## This module is a child-class of Solver for the dynamic programming solver
def DP_Solver(Solver):
    def __init__(self):
        pass

## This module is a child-class of Solver for the greedy solver
def Greedy_Solver(Solver):
    def __init__(self):
        pass

## This module generate plots and tables
class ReportFactory:
    def __init__(self):
        pass
    
    def get_plot(self):
        pass
    
    def get_table(self):
        pass

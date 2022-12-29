import json
import numpy as np
import pandas as pd
import torch
import Utils.setup as setup
import Utils.neural as neural
import Utils.mdp as mdp
import Utils.train as train

def clean_data():
    pass

def main(args):
    ## Setup
    map = setup.Map(**args["map"])
    time_horizon = args["mdp"]["time_horizon"]
    trip_demands = setup.TripDemands(time_horizon = time_horizon, **args["trip_demand"])
    reward_query = mdp.Reward(**args["reward"])
    markov_decision_process = mdp.MarkovDecisionProcess(map, trip_demands, reward_query, **args["mdp"])
    solver_factory = train.SolverFactory(type = "dp", markov_decision_process = markov_decision_process)
    solver = solver_factory.get_solver()
    ## Training
    solver.train()
    print(solver.optimal_values)
#    for t in range(time_horizon):
#        print(t, solver.optimal_states[t])
    ## Evaluation
    ## TODO: Implement it!!!

with open("Args/simple_test.json", "r") as f:
    args = json.load(f)

main(args)

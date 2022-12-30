import json
import numpy as np
import pandas as pd
import torch
import Utils.setup as setup
import Utils.neural as neural
import Utils.mdp as mdp
import Utils.train as train

## Check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    DEVICE = "cpu"
else:
    print('CUDA is available!  Training on GPU ...')
    DEVICE = "cuda"

def clean_data():
    pass

def main(args):
    ## Setup
    map = setup.Map(**args["map"])
    time_horizon = args["mdp"]["time_horizon"]
    trip_demands = setup.TripDemands(time_horizon = time_horizon, **args["trip_demand"])
    reward_query = mdp.Reward(**args["reward"])
    markov_decision_process = mdp.MarkovDecisionProcess(map, trip_demands, reward_query, **args["mdp"])
    solver_type = args["solver"]["type"]
    if solver_type == "dp":
        solver = train.DP_Solver(markov_decision_process = markov_decision_process)
    elif solver_type == "rl":
        solver = train.RL_Solver(markov_decision_process = markov_decision_process, device = DEVICE, **args["neural"])
    ## Training
    if solver_type == "dp":
        solver.train()
        print("Max Values", solver.optimal_values)
        ## Printing out the first optimal actions in each timestamp
        print("Optimal Actions:")
        for t in range(time_horizon - 1):
            action_id = solver.optimal_atomic_actions[t][1][0]
            action = markov_decision_process.all_actions[action_id]
            print(f"\tt = {t}", action.describe())
    #    for t in range(time_horizon):
    #        print(t, solver.optimal_states[t])
    elif solver_type == "rl":
        loss_arr = solver.train()
        report_factory = train.ReportFactory()
        if "training_loss" in args["report"]["plot"]:
            report_factory.get_training_loss_plot(loss_arr, "Total Payoff Loss", "train_loss")
    ## Evaluation
    ## TODO: Implement it!!!
    

with open("Args/1car_2region_rl.json", "r") as f:
    args = json.load(f)

main(args)

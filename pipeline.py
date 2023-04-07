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

def main(args, json_name = ""):
    ## Setup
    if "n_threads" in args["solver"]:
        n_threads = args["solver"]["n_threads"]
    else:
        n_threads = 4
    torch.set_num_threads(n_threads)
    map = setup.Map(**args["map"])
    time_horizon = args["mdp"]["time_horizon"]
    trip_demands = setup.TripDemands(time_horizon = time_horizon, **args["trip_demand"])
    reward_query = mdp.Reward(**args["reward"])
    markov_decision_process = mdp.MarkovDecisionProcess(map, trip_demands, reward_query, **args["mdp"])
    solver_type = args["solver"]["type"]
    if solver_type == "dp":
        solver = train.DP_Solver(markov_decision_process = markov_decision_process)
    elif solver_type == "policy_iteration":
        solver = train.PolicyIteration_Solver(markov_decision_process = markov_decision_process, device = DEVICE, **args["neural"])
    elif solver_type == "ppo":
        solver = train.PPO_Solver(markov_decision_process = markov_decision_process, device = DEVICE, **args["neural"])
    
    ## Training
    if solver_type == "dp":
        solver.train()
        print("Max Values", solver.optimal_values)
        ## Printing out the first optimal actions in each timestamp
        print("Optimal Actions:")
        for t in range(time_horizon - 1):
            tup = solver.optimal_atomic_actions[t][1]
            if len(tup) > 0:
                action_id = tup[0]
                action = markov_decision_process.all_actions[action_id]
                print(f"\tt = {t}", action.describe())
            else:
                print(f"\tt = {t}", "No new actions applied")
        for t in range(time_horizon):
            print(t, markov_decision_process.describe_state_counts(solver.optimal_states[t]))
    elif solver_type == "policy_iteration":
        loss_arr = solver.train()
        report_factory = train.ReportFactory()
        if "training_loss" in args["report"]["plot"]:
            report_factory.get_training_loss_plot(loss_arr, "Total Payoff Loss", "train_loss")
    elif solver_type == "ppo":
        value_loss_arr, policy_loss_arr, payoff_arr = solver.train(return_payoff = True, debug = True, debug_dir = f"debugging_log_{json_name}.txt")
        report_factory = train.ReportFactory()
        suffix = f"epi={args['neural']['num_episodes']}_batch={args['neural']['value_batch']}_itr={args['neural']['num_itr']}_eps={args['neural']['eps']}"
        report_factory.get_training_loss_plot(value_loss_arr, "Value Loss", f"value_loss_{json_name}")
        report_factory.get_training_loss_plot(policy_loss_arr, "Policy Loss", f"policy_loss_{json_name}")
        report_factory.get_training_loss_plot(payoff_arr, "Total Payoff", f"total_payoff_{json_name}")
        report_factory.get_training_loss_plot(payoff_arr, "Total Payoff", f"total_payoff_{json_name}_{suffix}")
        _, _, payoff_lst, action_lst = solver.evaluate(return_action = True, seed = 0)
#        print(f"Value Loss = {value_loss}")
        with open(f"ppo_output_{json_name}.txt", "w") as f:
#            print(f"Policy Loss = {policy_loss}")
            print(f"Total Payoff = {float(payoff_lst[-1].data)}")
            #print(f"Total Payoff = {float(torch.sum(payoff_lst).data)}")
#            print(payoff_lst)
            
#            f.write(f"Policy Loss = {policy_loss}\n")
            f.write(f"Total Payoff = {float(payoff_lst[-1].data)}\n")
            f.write(f"{payoff_lst}\n")
    #        print(markov_decision_process.describe_state_counts())
            for tup in action_lst:
                curr_state_counts, action, t, car_idx = tup
#                print(f"t = {t}, car = {car_idx}:")
#                print(markov_decision_process.describe_state_counts(curr_state_counts))
#                print(f"action = {action.describe()}")
                
                f.write(f"t = {t}, car = {car_idx}:\n")
                f.write(f"{markov_decision_process.describe_state_counts(curr_state_counts)}\n")
                f.write(f"action = {action.describe()}\n")
        
    ## Evaluation
    ## TODO: Implement it!!!
    
JSON_NAME = "100car_3region_ppo" #"500car_5region_nyc_ppo" #"1car_3region_patience_ppo" #"1car_3region_dp" #

with open(f"Args/{JSON_NAME}.json", "r") as f:
    args = json.load(f)

main(args, json_name = JSON_NAME)

import json
import numpy as np
import pandas as pd
import torch
import Utils.setup as setup
import Utils.neural as neural
import Utils.mdp as mdp
import Utils.train as train
from tqdm import tqdm

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
    if "neural" in args and "state_reduction" in args["neural"] and args["neural"]["state_reduction"]:
        descriptor = "reduced"
    else:
        descriptor = "full"
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
        model_descriptor = args["neural"]["descriptor"] + f"_{json_name}_{descriptor}"
        del args["neural"]["descriptor"]
        solver = train.PPO_Solver(markov_decision_process = markov_decision_process, descriptor = model_descriptor, device = DEVICE, **args["neural"])
    elif solver_type == "d_closest":
        solver = train.D_Closest_Car_Solver(markov_decision_process = markov_decision_process, **args["d_closest"])
    
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
        report_factory = train.ReportFactory()
        if args["neural"]["num_itr"] > 0:
            value_loss_arr, policy_loss_arr, payoff_arr = solver.train(return_payoff = True, debug = True, debug_dir = f"debugging_log_{json_name}.txt", label = f"{json_name}_{descriptor}")
            suffix = f"epi={args['neural']['num_episodes']}_batch={args['neural']['value_batch']}_itr={args['neural']['num_itr']}_eps={args['neural']['eps']}"
            report_factory.get_training_loss_plot(value_loss_arr, "Value Loss", f"value_loss_{json_name}_{descriptor}")
            report_factory.get_training_loss_plot(policy_loss_arr, "Policy Loss", f"policy_loss_{json_name}_{descriptor}")
            report_factory.get_training_loss_plot(payoff_arr, "Total Payoff", f"total_payoff_{json_name}_{descriptor}")
    #        report_factory.get_training_loss_plot(payoff_arr, "Total Payoff", f"total_payoff_{json_name}_{suffix}")
            
    #        print(f"Value Loss = {value_loss}")
            with open(f"Logs/ppo_payoff_{json_name}_{descriptor}.txt", "w") as f:
                
    #            f.write(f"Policy Loss = {policy_loss}\n")
    #            f.write(f"Total Payoff = {float(payoff_lst[-1].data)}\n")
    #            f.write(f"{payoff_lst}\n")
                f.write(",".join([str(x) for x in payoff_arr]))
        #        print(markov_decision_process.describe_state_counts())
    elif solver_type == "d_closest":
        report_factory = train.ReportFactory()
    
    df_table_all = None
    payoff = 0
    num_trials = 10#args["neural"]["num_episodes"]
    for i in tqdm(range(num_trials)):
        _, _, payoff_lst, action_lst, discounted_payoff = solver.evaluate(return_action = True, seed = None)
        #            print(f"Policy Loss = {policy_loss}")
#            print(f"Total Payoff = {float(payoff_lst[-1].data)}")
            #print(f"Total Payoff = {float(torch.sum(payoff_lst).data)}")
#            print(payoff_lst)
        payoff += float(discounted_payoff.data) #float(payoff_lst[-1].data)
        df_table = report_factory.get_table(markov_decision_process, action_lst, detailed = True)
        df_table["trial"] = i
        if df_table_all is None:
            df_table_all = df_table
        else:
            df_table_all = pd.concat([df_table_all, df_table], axis = 0)
    payoff /= num_trials
    print(f"Total Payoff = {payoff}")
    df_table_all_cp = df_table_all.copy()
    df_table_all_cp = df_table_all_cp.groupby("t").quantile(0.95).reset_index().sort_values("t")
    df_table_all = df_table_all.groupby(["t"]).mean().reset_index().sort_values("t")
    for col in [x for x in df_table_all.columns if x.startswith("num_charging_cars_region_")]:
        df_table_all[col] = df_table_all_cp[col].copy()
    df_table_all.to_csv(f"Tables/table_{json_name}_{descriptor}.csv", index = False)
    report_factory.visualize_table(df_table_all, f"{json_name}_{descriptor}", detailed = True)
#    for tup in action_lst:
#        curr_state_counts, action, t, car_idx = tup
#        if action is not None:
#            print(f"t = {t}, car = {car_idx}:")
#            print(markov_decision_process.describe_state_counts(curr_state_counts))
#            print(f"action = {action.describe()}")
#        else:
#            print(markov_decision_process.describe_state_counts(curr_state_counts))
        
    ## Evaluation
    ## TODO: Implement it!!!
    
JSON_NAME = "1car_2region_ppo" #"12car_4region_48charger_15min_fullycharged_nyc_combo_ppo" #"st-stc_12car4region48chargers_xi=1" #"12car_4region_2charger_15min_fullycharged_work_nyc_combo_ppo" #"1car_2region_ppo" #"100car_4region_400charger_15min_fullycharged_nyc_ppo" #"10car_5region_d-closest" #"12car_4region_48charger_15min_demandScale2_fullycharged_nyc_d-closest" #"12car_4region_2charger_15min_fullycharged_workair_nyc_ppo" #"200car_4region_nyc_ppo" #"100car_3region_ppo" # "1car_3region_patience_ppo" #"1car_3region_dp" #

with open(f"Args/{JSON_NAME}.json", "r") as f:
    args = json.load(f)

main(args, json_name = JSON_NAME)

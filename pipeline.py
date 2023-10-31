import json
import math
import numpy as np
import pandas as pd
import torch
import Utils.setup as setup
import Utils.neural as neural
import Utils.mdp as mdp
import Utils.train as train
import Utils.lp_solvers as lp_solvers
from joblib import Parallel, delayed
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

def evaluate_batch(solver, markov_decision_process, time_horizon, num_trials, eval_days, seed_lst = None, randomized_eval_time = 1, solver_type = "ppo", lp_eval_fractional_cars = True, lp_assume_full_knowledge = False, n_cpu = 1):
    df_table_all = None
    batch_size = int(math.ceil(num_trials / n_cpu))
    if seed_lst is None:
        seed_lst = [None] * num_trials
    res = Parallel(n_jobs = n_cpu)(delayed(evaluate_batch_single)(
        solver, markov_decision_process, time_horizon, min(num_trials, (i + 1) * batch_size) - i * batch_size, eval_days, seed_lst[(i * batch_size * randomized_eval_time):(min(num_trials, (i + 1) * batch_size) * randomized_eval_time)], randomized_eval_time, solver_type, lp_eval_fractional_cars, lp_assume_full_knowledge
    ) for i in range(n_cpu))
    payoff = 0
    for df_table, payoff_single in res:
        if df_table_all is None:
            df_table_all = df_table
        else:
            df_table_all = pd.concat([df_table_all, df_table], axis = 0)
        payoff += payoff_single
    payoff /= num_trials
    return df_table_all, payoff

def evaluate_batch_single(solver, markov_decision_process, time_horizon, num_trials, eval_days, seed_lst = None, randomized_eval_time = 1, solver_type = "ppo", lp_eval_fractional_cars = True, lp_assume_full_knowledge = False):
    df_table_all = None
    report_factory = train.ReportFactory()
    norm_factor = eval_days #torch.sum(self.gamma ** (self.time_horizon * torch.arange(self.eval_days)))
    payoff = 0
    for i in tqdm(range(num_trials)):
        for random_eval_round in tqdm(range(randomized_eval_time), leave = False):
            for day in range(eval_days):
                if solver_type != "LP-AugmentedGraph":
                    _, _, payoff_lst, action_lst, discounted_payoff, passenger_carrying_cars = solver.evaluate(return_action = True, seed = seed_lst[i * randomized_eval_time + random_eval_round], day_num = day, log_policy = False)
                else:
                    _, _, payoff_lst, action_lst, discounted_payoff, passenger_carrying_cars = solver.evaluate(return_action = True, seed = seed_lst[i * randomized_eval_time + random_eval_round], day_num = day, full_knowledge = lp_assume_full_knowledge, fractional_cars = lp_eval_fractional_cars, random_eval_round = random_eval_round)
                if len(payoff_lst) > 0:
                    curr_payoff = float(payoff_lst[-1].data - payoff_lst[0].data) / norm_factor / randomized_eval_time #float(payoff_lst[-1].data)
                    payoff += curr_payoff
                if solver_type != "LP-AugmentedGraph" or not lp_eval_fractional_cars:
                    df_table = report_factory.get_table(markov_decision_process, action_lst, passenger_carrying_cars, detailed = True)
                    df_table["trial"] = i
                    df_table["t"] += day * time_horizon
                    if df_table_all is None:
                        df_table_all = df_table
                    else:
                        df_table_all = pd.concat([df_table_all, df_table], axis = 0)
    return df_table_all, payoff

def main(args, json_name = ""):
    ## Setup
    if "n_threads" in args["solver"]:
        n_threads = args["solver"]["n_threads"]
    else:
        n_threads = 4
    if "n_cpu" in args["solver"]:
        n_cpu = args["solver"]["n_cpu"]
    else:
        n_cpu = 1
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
    lp_assume_full_knowledge = False
    lp_eval_fractional_cars = True
    randomized_eval_time = 1
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
    elif solver_type == "LP-AugmentedGraph":
        solver = lp_solvers.LP_On_AugmentedGraph(markov_decision_process = markov_decision_process, **args["LP-AugmentedGraph"])
        if "full_knowledge" in args["LP-AugmentedGraph"]:
            lp_assume_full_knowledge = args["LP-AugmentedGraph"]["full_knowledge"]
        else:
            lp_assume_full_knowledge = False
        if "fractional_cars" in args["LP-AugmentedGraph"]:
            lp_eval_fractional_cars = args["LP-AugmentedGraph"]["fractional_cars"]
        else:
            lp_eval_fractional_cars = True
        if "randomized_eval_time" in args["LP-AugmentedGraph"]:
            randomized_eval_time = args["LP-AugmentedGraph"]["randomized_eval_time"]
        else:
            randomized_eval_time = 1
    if solver_type != "LP-AugmentedGraph":
        randomized_eval_time = 1
    if "eval_days" in args["report"]:
        eval_days = args["report"]["eval_days"]
    else:
        eval_days = 1
    if "gamma" in args["report"]:
        gamma = args["report"]["gamma"]
    else:
        gamma = 1
    num_trials = 10
    np.random.seed(123)
    seed_lst = np.random.choice(10000, num_trials * randomized_eval_time)
    
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
    elif solver_type == "LP-AugmentedGraph":
        report_factory = train.ReportFactory()
        if not lp_assume_full_knowledge:
            solver.train()
            solver.plot_fleet_status(f"{json_name}_{descriptor}")
    
    df_table_all, payoff = evaluate_batch(solver, markov_decision_process, time_horizon, num_trials, eval_days, seed_lst = seed_lst, randomized_eval_time = randomized_eval_time, solver_type = solver_type, lp_eval_fractional_cars = lp_eval_fractional_cars, lp_assume_full_knowledge = lp_assume_full_knowledge, n_cpu = min(n_cpu, num_trials))
#    vis_day = 3
#    df_table_all = df_table_all[(df_table_all["t"] >= vis_day * time_horizon) & (df_table_all["t"] < (vis_day + 1) * time_horizon)]
#    df_table_all["t"] = df_table_all["t"].apply(lambda x: x % time_horizon)

    print(f"Total Payoff = {payoff}")
    if solver_type == "LP-AugmentedGraph" and lp_eval_fractional_cars:
        return None
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
    
JSON_NAME = "100car_5region_500charger_5min_fullycharged_nyc_combo_fullday_ppo" #"12car_4region_48charger_15min_fullycharged_nyc_combo_fullday_ppo" #"12car_4region_4charger_15min_fullycharged_nyc_combo_fullday_slowchargers_ppo" #"12car_4region_48charger_15min_fullycharged_nyc_combo_fullday_ppo" #"50car_5region_250charger_5min_fullycharged_nyc_combo_fullday_lp-augmented" #"12car_4region_48charger_15min_fullycharged_nyc_combo_fullday_ppo" #"100car_10region_1000charger_5min_fullycharged_nyc_combo_fullday_ppo" #"1car_2region_ppo" #"st-stc_12car4region48chargers_xi=1" #"12car_4region_2charger_15min_fullycharged_work_nyc_combo_ppo" #"1car_2region_ppo" #"100car_4region_400charger_15min_fullycharged_nyc_ppo" #"10car_5region_d-closest" #"12car_4region_48charger_15min_demandScale2_fullycharged_nyc_d-closest" #"12car_4region_2charger_15min_fullycharged_workair_nyc_ppo" #"200car_4region_nyc_ppo" #"100car_3region_ppo" # "1car_3region_patience_ppo" #"1car_3region_dp" #

with open(f"Args/{JSON_NAME}.json", "r") as f:
    args = json.load(f)

main(args, json_name = JSON_NAME)

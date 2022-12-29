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

test_args = {
    "map": {
        "map_system": "graph",
        "num_nodes": 2,
        "graph_edge_lst": [(0, 0), (0, 1), (1, 0), (1, 1)]
    },
    "trip_demand": {
        "parameter_source": "given",
        "arrival_type": "constant",
        "parameter_fname": "trip_demand_test.tsv",
        "data": None
    },
    "reward": {
        "reward_fname": "payoff_test.tsv"
    },
    "mdp": {
        "time_horizon": 5,
        "connection_patience": 0,
        "pickup_patience": 0,
        "num_battery_levels": 3,
        "battery_jump": 0.3,
        "charging_rates": [2],
        "battery_offset": 0,
        "region_battery_car_fname": "region_battery_car_test.tsv",
        "region_rate_plug_fname": "region_rate_plug_test.tsv"
    },
    "solver": {
        "type": "dp"
    },
    "metric": [],
    "report": {
        "plot": [],
        "table": []
    }
}

args = {
    "map": {
        "map_system": "grid",
        "num_layers": 3,
        "num_nodes": 2
    },
    "trip_demand": {
        "parameter_source": "given",
        "arrival_type": "poisson",
        "parameter_fname": "trip_demand.tsv",
        "data": None
    },
    "reward": {
        "reward_fname": "payoff.tsv"
    },
    "mdp": {
        "time_horizon": 5,
        "connection_patience": 2,
        "pickup_patience": 3,
        "num_battery_levels": 3,
        "battery_jump": 0.5,
        "charging_rates": [5, 6],
        "battery_offset": 1,
        "region_battery_car_fname": "region_battery_car.tsv",
        "region_rate_plug_fname": "region_rate_plug.tsv"
    },
    "solver": {
        "type": "rl"
    },
    "neural": {
        "model_name": "discretized_feedforward",
        "input_dim": 10,
        "hidden_dim_lst": [10, 10],
        "activation_lst": ["relu", "relu"],
        "output_dim": 1,
        "batch_norm": False,
        "lr": 1e-2,
        "decay": 0.1,
        "scheduler_step": 10000,
        "solver": "Adam",
        "retrain": False
    },
    "metric": [],
    "report": {
        "plot": [],
        "table": []
    }
}

main(test_args)

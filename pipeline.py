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
    pass

args = {
    "map": {
        "map_system": "grid",
        "num_layers": 3,
        "num_nodes": 2
    },
    "trip_demand": {
        "parameter_source": "given",
        "arrival_type": "poisson",
        "parameter_lst": None,
        "data": None
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

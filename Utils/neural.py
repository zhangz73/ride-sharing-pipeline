import os, os.path
import pytz
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

## This module defines a single feed-forward neural network
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim_lst, activation_lst, output_dim = 1, batch_norm = False, prob = False, use_embedding = False, num_embeddings = 1, embedding_dim = 1):
        super(Net, self).__init__()
        for act in activation_lst:
            assert act in ["relu", "softmax", "tanh"]
        self.layer_lst = nn.ModuleList()
#         self.bn = nn.ModuleList()
        self.batch_norm = batch_norm
        self.activation_lst = activation_lst
        self.prob = prob
        self.use_embedding = use_embedding
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

        if self.use_embedding:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.input_dim += embedding_dim
        self.layer_lst.append(nn.Linear(self.input_dim, hidden_dim_lst[0]))
#         self.bn.append(nn.BatchNorm1d(hidden_dim_lst[0],momentum=0.1))
        for i in range(1, len(hidden_dim_lst)):
            self.layer_lst.append(nn.Linear(hidden_dim_lst[i - 1], hidden_dim_lst[i]))
#             self.bn.append(nn.BatchNorm1d(hidden_dim_lst[i],momentum=0.1))
        self.layer_lst.append(nn.Linear(hidden_dim_lst[-1], output_dim))
#         if self.prob:
#             for i in range(len(self.layer_lst)):
#                 self.layer_lst[i].bias.data.fill_(1)
#                 self.layer_lst[i].weight.data.fill_(1)

    def forward(self, tup):
        t, x = tup
        if self.use_embedding:
            embeds = self.embedding(t)
            embeds = embeds.view(embeds.shape[0], embeds.shape[2])
            x = torch.cat((x, embeds), dim=1)
        for i in range(len(self.layer_lst) - 1):
            x = self.layer_lst[i](x)
#             if self.batch_norm:
#                 x = self.bn[i](x)
            if self.activation_lst[i] == "relu":
                x = F.relu(x)
            elif self.activation_lst[i] == "softmax":
                x = F.softmax(x, dim = 1)
            elif self.activation_lst[i] == "tanh":
                x = torch.tanh(x)
        x = self.layer_lst[-1](x)
        if self.prob:
            x = F.softmax(x, dim = 1)
        return x

## This module wraps time-discretized neural models and the non time-discretized models into a uniform format
class ModelFull(nn.Module):
    def __init__(self, predefined_model, is_discretized = False, ts_per_network = 1):
        super(ModelFull, self).__init__()
        self.model = predefined_model
        self.is_discretized = is_discretized
        self.ts_per_network = ts_per_network
    
    ## Uniformize the input format as (t, x) for both discretized and non-discretized models
    def forward(self, tup):
        t, x = tup
        if self.is_discretized:
            t_idx = t // self.ts_per_network
            t_remainder = torch.ones((x.shape[0], 1)).long() * (t % self.ts_per_network) #torch.tensor([t % self.ts_per_network] * x.shape[0]).reshape((x.shape[0], 1))
            return self.model[t_idx]((t_remainder, x))
        else:
            return self.model(x)

## This module constructs neural network models and prepares for the training pipeline
class ModelFactory:
    ## discretized_len will only be used when model_name = "discretized_feedforward"
    ## descriptor is used only when retrain = False, so that it loads the latest model
    ##  attached to the given descriptor
    ## dir specifies the current working directory. It will be "." unless we are on Google Colab
    def __init__(self, model_name, input_dim, hidden_dim_lst, activation_lst, output_dim, batch_norm, lr, decay, scheduler_step, solver = "Adam", retrain = False, discretized_len = 1, descriptor = None, dir = ".", device = "cpu", prob = False, use_embedding = False, num_embeddings = 1, embedding_dim = 1, ts_per_network = 1):
        assert solver in ["Adam", "SGD", "RMSprop"]
        assert model_name in ["discretized_feedforward", "rnn"]
        assert len(hidden_dim_lst) == len(activation_lst)
        self.model_name = model_name
        self.input_dim = input_dim
        self.hidden_dim_lst = hidden_dim_lst
        self.activation_lst = activation_lst
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.lr = lr
        self.decay = decay
        self.scheduler_step = scheduler_step
        self.solver = solver
        self.retrain = retrain
        self.discretized_len = discretized_len
        self.descriptor = descriptor
        self.dir = dir
        self.device = device
        self.prob = prob
        self.use_embedding = use_embedding
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.ts_per_network = ts_per_network
        ## The timestamp when the latest model is stored into self.model
        self.model_ts = self.get_curr_ts()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        if self.model is None:
            if model_name == "discretized_feedforward":
                self.model = self.discretized_feedforward()
                self.model = ModelFull(self.model, is_discretized = True, ts_per_network = self.ts_per_network)
            else:
                self.model = self.rnn()
                self.model = ModelFull(self.model, is_discretized = False)
        if not retrain:
#             self.model = self.load_latest(self.descriptor)
            self.load_latest(self.descriptor)
        self.model = self.model.to(device = self.device)
        self.value_scale = {}

    ## Construct a discretized feedforward neural network
    ## The neural network is discretized along the timestamps
    ##   with each timestamp being a shallow neural network
    ## Note that we can construct a single shallow network by setting discretized_len = 1
    def discretized_feedforward(self):
        model_list = nn.ModuleList()
        for _ in range(self.discretized_len):
            model = Net(self.input_dim, self.hidden_dim_lst, self.activation_lst, self.output_dim, self.batch_norm, self.prob, self.use_embedding, self.num_embeddings, self.embedding_dim)
            model_list.append(model)
        return model_list
    
    ## Set the value scale
    def set_value_scale(self, value_scale):
        self.value_scale = value_scale.copy()
    
    ## Get the value scale
    def get_value_scale(self):
        return self.value_scale.copy()
    
    ## Recurrent neural network. To be implemented
    def rnn(self):
        ## TODO: Implement it if needed
        pass
    
    ## Fetch the model
    def get_model(self):
        return self.model
    
    ## Update model from the external source, such as previously trained model checkpoints
    ## Update model timestamp as well if update_ts = True
    def update_model(self, model, update_ts = True):
        self.model = model
        if update_ts:
            self.model_ts = self.get_curr_ts()

    ## Set up the model optimizer and training scheduler objects
    def prepare_optimizer(self, weight_decay = 0):
        if self.solver == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = weight_decay)
        elif self.solver == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, weight_decay = weight_decay)
        else:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.lr, weight_decay = weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = self.scheduler_step, gamma = self.decay)
        return self.optimizer, self.scheduler
    
    ## Save the model to file named according to the given descriptor
    ## Append the current timestamp to the model if include_ts = True
    def save_to_file(self, descriptor = "", include_ts = False):
        ## Need to convert model to CPU device before saving
        ##  otherwise it cannot be loaded without GPU devices
        model_save = self.model.cpu()
        name = self.descriptor + descriptor
        if include_ts:
            name += "__" + self.model_ts
#        ckpt = {"model_state_dict": model_save.state_dict(), "opt_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(), "value_scale": self.value_scale}
        ckpt = {"model_state_dict": model_save.state_dict(), "value_scale": self.value_scale}
        torch.save(ckpt, f"{self.dir}/Models/{name}.pt")
        ## Convert the model back to its original device
        self.model = self.model.to(device = self.device)
    
    ## Load the latest model from file given the descriptor
    ##  A more delicate wrapper for load_model()
    def load_latest(self, descriptor):
#         return self.load_model(descriptor, update = False, include_ts = True)
        self.load_model(descriptor, update = False, include_ts = True)

    ## Load the model from file given the descriptor
    ## Updates the current model to the loaded one if update = True.
    ##  Otherwise, return the model directly
    ## Loads the model with the specific timestamp ts if include_ts = True
    ##  Loads the model with the latest timestamp if ts = None
    ##  Note that if there is no timestamps attached to the descriptor, then
    ##  try to load the model with purely the descriptor as its name
    ## If the model does not exist, returns None
    def load_model(self, descriptor, update = True, include_ts = False, ts = None):
        name = descriptor
        if include_ts:
            if ts is None:
                ts = self.get_latest_ts(descriptor)
            if ts is not None:
                name += "__" + ts
                self.model_ts = ts
        dir_fname = f"{self.dir}/Models/{name}.pt"
        if not os.path.isfile(dir_fname):
            return None
        print(f"Loading {dir_fname}...")
        ckpt = torch.load(dir_fname)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.set_value_scale(ckpt["value_scale"])
        self.model.eval()
#         model = torch.load(dir_fname)
#         model = model.to(device = self.device)
#         model.eval()
        if update:
            self.update_model(self.model, update_ts = False)
#         return model

    ## Compute the current timestamp
    def get_curr_ts(self):
        ## Get the current timestamp using New York timezone
        ts = datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d-%H-%M-%S")
        return ts

    ## Get the latest timestamp attached to a given descriptor
    def get_latest_ts(self, descriptor):
        ## Get all timestamps attached to the given descriptor
        ts_lst = [f.strip(".pt").split("__")[1] for f in os.listdir(f"{self.dir}/Models/") if f.endswith(".pt") and f.startswith(descriptor)]
        ## Sort those timestamps in reverse order
        ts_lst = sorted(ts_lst, reverse=True)
        ## Return None if the list is empty
        if len(ts_lst) == 0:
            return None
        ## Otherwise, return the largest timestamp
        return ts_lst[0]

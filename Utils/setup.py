import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch

## This module discretizes the map system of a designated type
##  Currently it does NOT keep track of exact GPS locations
## Functionalities:
##      1. Construct map system of graph, grid, or hexagon
##      2. Enabling some arbitrary merging of cells
##      3. Compute the traveling related information:
##          The adjacent cells of a given region (self-exclusive)
##          The next cell on the shortest path from origin to destination
##          Number of steps required to travel from origin to destination
##          Cells that can get to the given region within L steps (self-inclusive)
##      4. Map GPS data into regions
##  Note that the Map module is merely used for describing geographical relationships
##  We assume that the map of all regions is connected
class Map:
    ## num_layers is used for grid and hexagon systems, while num_nodes is used for graph systems
    ## graph_edge_lst is only used for graph system. A fully connected graph
    ##  will be built if graph_edge_lst is None or empty
    ## gps_data is a dataframe containing columns "lon" and "lat"
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.nodes = []
        self.df = None
        self.distance_df = None
        self.construct_map()
        ## Compute the max trip time between any two regions
        self.max_travel_time = self.df["TripTime"].max()
        ## Auxiliary variables to speedup queries
        self.curr_ts = 0
        self.curr_origin = 0
        self.df_cache = self.df[(self.df["T"] == 0) & (self.df["Origin"] == 0)]
        self.distance_cache = self.distance_df[self.distance_df["Origin"] == 0]
    
    ## Construct the map
    ## Remove the absorbing nodes
    def construct_map(self):
        ## df schema: Origin, Destination, Distance, TripTime, T
        df = pd.read_csv(f"Data/{self.data_dir}", sep = "\t")
        origin_nodes = set(df["Origin"].unique())
        dest_nodes = set(df["Destination"].unique())
        nodes = origin_nodes.intersection(dest_nodes)
        df = df[(df["Origin"].isin(nodes)) & (df["Destination"].isin(nodes))]
        self.df = df
        self.nodes = nodes
        self.distance_df = df[["Origin", "Destination", "Distance"]].drop_duplicates()
    
    ## Get the list of all regions
    def get_regions(self):
        return list(self.nodes)
    
    ## Get the max travel steps
    def get_max_travel_time(self):
        return self.max_travel_time
        
    ## The travel time from origin to dest at epoch ts
    def time_to_location(self, origin, dest, ts):
        ## Update cache if not applicable
        if ts != self.curr_ts:
            self.curr_ts = ts
            self.df_cache = self.df[self.df["T"] == ts]
        ## Query from cache
        tmp_df = self.df_cache[(self.df_cache["Destination"] == dest) & (self.df_cache["Origin"] == origin)]
        if tmp_df.shape[0] > 0:
            trip_time = tmp_df.iloc[0]["TripTime"]
        else:
            trip_time = 0
        return trip_time
    
    ## The distance (miles) from origin to dest
    def distance(self, origin, dest):
        ## Update cache if not applicable
        if origin != self.curr_origin:
            self.curr_origin = origin
            self.distance_cache = self.distance_df[self.distance_df["Origin"] == origin]
        tmp_df = self.distance_cache[self.distance_cache["Destination"] == dest]
        if tmp_df.shape[0] > 0:
            dist = tmp_df.iloc[0]["Distance"]
        else:
            dist = 0
        return dist
    
## This module randomly generate passenger demand requests at a given region at a given time
## Functionalities:
##      1. Generate demand request: constant, Poisson, data-driven
##      2. Infer the arrival parameters from data
class TripDemands:
    ## parameter_lst is only used when:
    ##      1. The arrival process is constant for Poisson, and
    ##      2. The parameter_source = "given"
    ## data is only used when:
    ##      1. The parameter_source = "inferred", or
    ##      2. The arrival_type is data-driven
    ## data input format: A dataframe containing at least 3 columns:
    ##      1. Timestamp: Exact timestamps (str HH:MM:SS) of trip requests
    ##      2. Trip Origin: The origin of the passenger trip request
    ##      3. Trip Destination: The origin of the passenger trip request
    ## smooth_time_window and smooth_region_window is only used when there are no trip requests
    ##  at a specific region at a specific timestamp.
    def __init__(self, time_horizon, parameter_source = "given", arrival_type = "poisson", parameter_fname = "trip_demand.tsv", data = None, start_ts = "00:00:00", end_ts = "23:59:59", smooth_time_window = 3, smooth_region_window = 3, scaling_factor = 1):
        assert parameter_source in ["given", "inferred"]
        assert arrival_type in ["constant", "poisson", "data-driven"]
        self.time_horizon = time_horizon
        self.parameter_source = parameter_source
        self.arrival_type = arrival_type
        self.parameter_fname = parameter_fname
        self.input_data = data
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.smooth_time_window = smooth_time_window
        self.smooth_region_window = smooth_region_window
        self.scaling_factor = scaling_factor
        ## Auxiliary objects
        self.data = None
        ## Populate the arrival rate df
        if parameter_source == "inferred" or arrival_type == "data-driven":
            assert data is not None
            self.infer_parameters()
        if parameter_source == "given":
            self.data = pd.read_csv(f"Data/{self.parameter_fname}", sep = "\t")
        self.data_wide = self.long_to_wide()
    
    ## Infer parameters from the data
    ## Procedures:
    ##      1. Discretize the timestamps based on time_horizon
    ##      2. Aggregate the frequencies of each trip time at each discretized timestamp
    def infer_parameters(self):
        ## Convert timestamps to seconds
        start_sec = datatime.strptime(self.start_ts, "%H:%M:%S")
        end_sec = datatime.strptime(self.end_ts, "%H:%M:%S")
        total_secs = end_sec - start_sec
        time_dist = int(math.ceil(total_secs / self.time_horizon))
        self.input_data["ts"] = self.input_data["Timestamp"].apply(lambda x: (datetime.strptime(x, "%H:%M:%S") - baseline_ts).seconds)
        ## Discretize timestamps
        self.input_data["T"] = self.input_data["ts"].apply(lambda x: x // time_dist)
        ## Aggregate
        self.data = self.input_data[["T", "Origin", "Destination"]]
        self.data["Count"] = 1
        self.data = self.data.groupby(["T", "Origin", "Destination"]).sum().reset_index()
    
    ## Smooth the parameters for sparse timestamps and regions
    def smooth_parameters(self):
        pass
    
    ## Transform the data from long to wide format
    def long_to_wide(self):
        num_regions = int(max(self.data["Origin"].max(), self.data["Destination"].max())) + 1
        mat = np.zeros((self.time_horizon, num_regions * num_regions))
        for t in range(self.time_horizon):
            for origin in range(num_regions):
                for dest in range(num_regions):
                    tmp_df = self.data[(self.data["T"] == t) & (self.data["Origin"] == origin) & (self.data["Destination"] == dest)]
                    if tmp_df.shape[0] > 0:
                        cnt = tmp_df.iloc[0]["Count"]
                    else:
                        cnt = 0
                    mat[t, origin * num_regions + dest] = cnt
        return mat
    
    ## Simulate 1 trial of the passenger arrivals
    def generate_arrivals(self):
        if self.arrival_type == "constant":
            return self.generate_constant_arrivals()
        elif self.arrival_type == "poisson":
            return self.generate_poisson_arrivals()
        return self.generate_data_driven_arrivals()
    
    ## Get arrival rates
    def get_arrival_rates(self):
        return self.data_wide.copy()

    ## Generate 1 trial of the constant arrival process
    def generate_constant_arrivals(self):
#        ret = self.data[["T", "Origin", "Destination", "Count"]].copy()
#        ret["Count"] = ret["Count"].round(0).astype(int)
        ret = self.data_wide.round(0).astype(int)
        return ret
    
    ## Generate 1 trial of poisson arrival process
    def generate_poisson_arrivals(self):
#        ret = self.data[["T", "Origin", "Destination", "Count"]].copy()
#        ret["Count"] = ((np.random.poisson(lam = ret["Count"] * self.scaling_factor)) / self.scaling_factor).round(0).astype(int)
        ret = (np.random.poisson(self.data_wide * self.scaling_factor) / self.scaling_factor).round(0).astype(int)
        return ret
    
    ## TODO: Define the exact behavior of data-driven arrivals
    def generate_data_driven_arrivals(self):
        pass

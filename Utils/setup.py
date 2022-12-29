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
    def __init__(self, map_system = "grid", num_layers = 3, num_nodes = 2, graph_edge_lst = []):
        assert map_system in ["graph", "grid", "hexagon"]
        self.map_system = map_system
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.graph_edge_lst = graph_edge_lst
        self.map = {}
        
        ## Construct the map system
        if self.map_system == "graph":
            self.construct_graph_system()
        elif self.map_system == "grid":
            self.construct_grid_system()
        else:
            self.construct_hexagon_system()
        self.remove_map_duplicates()
        
        ## Helper objects for speeding up queryings
        ##  next_moves: (origin, dest) -> region
        ##  steps_to_loc: (origin, dest) -> # of steps
        self.steps_to_loc = {}
        self.next_moves = {}
        
        ## Populate the steps_to_loc and next_moves objects
        self.compute_cell_steps()
        self.compute_next_moves()
    
    ## Get the list of all regions
    def get_regions(self):
        return list(self.map.keys())
    
    ## Clean up duplicates from the map
    def remove_map_duplicates(self):
        for src in self.map:
            arr = np.unique(self.map[src])
            self.map[src] = list(arr)
    
    ## Construct the map system using the directed graph structure
    ## For debugging purposes only
    def construct_graph_system(self):
        if self.graph_edge_lst is None or len(self.graph_edge_lst) == 0:
            self.graph_edge_lst = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    self.graph_edge_lst.append((i, j))
        for edge in self.graph_edge_lst:
            src, dst = edge
            if src not in self.map:
                self.map[src] = []
            self.map[src].append(dst)
    
    def construct_grid_system(self):
        pass
    
    def construct_hexagon_system(self):
        pass
    
    def merge_cells(self):
        pass
    
    ## Compute the next adjacent region from src to dst
    def compute_next_moves(self):
        if self.map_system == "graph":
            self.compute_graph_next_moves()
        elif self.map_system == "grid":
            self.compute_grid_next_moves()
        else:
            self.compute_hexagon_next_moves()
    
    ## Compute the next adjacent region from src to dst for graph systems
    ##  Note that steps(src, dst) = steps(next_move, dst) + 1 if src != dst
    def compute_graph_next_moves(self):
        region_lst = self.get_regions()
        ## Populate next adjacent regions for self-loops
        for region in region_lst:
            self.next_moves[(region, region)] = region
        ## Populate next adjacent regions for distinct src and dst
        for src in region_lst:
            for dst in region_lst:
                if src != dst:
                    total_steps = self.steps_to_loc[(src, dst)]
                    for region in self.map[src]:
                        if self.steps_to_loc[(region, dst)] == total_steps - 1:
                            self.next_moves[(src, dst)] = region
                            break
    
    ## Compute the next adjacent region from src to dst for grid systems
    def compute_grid_next_moves(self):
        pass
    
    ## Compute the next adjacent region from src to dst for hexagon systems
    def compute_hexagon_next_moves(self):
        pass
    
    ## Compute the distance between every region-region pair
    def compute_cell_steps(self):
        if self.map_system == "graph":
            self.compute_graph_cell_steps()
        elif self.map_system == "grid":
            self.compute_grid_cell_steps()
        else:
            self.compute_hexagon_cell_steps()
    
    ## Compute the distance between every region-region pair for the graph system
    ##  The algorithm is implemented in BFS and DP
    def compute_graph_cell_steps(self):
        region_lst = self.get_regions()
        ## Initialize the steps_to_loc with max possible steps
        for src in region_lst:
            for dst in region_lst:
                self.steps_to_loc[(src, dst)] = len(region_lst)
        for region in region_lst:
            self.steps_to_loc[(region, region)] = 0
        for _ in range(len(region_lst)):
            for tup in self.steps_to_loc:
                src, dst = tup
                n_steps = self.steps_to_loc[(src, dst)]
                for region in self.map[dst]:
                    self.steps_to_loc[(src, region)] = min(self.steps_to_loc[(src, region)], n_steps + 1)
    
    ## Compute the distance between every region-region pair for the grid system
    def compute_grid_cell_steps(self):
        pass
    
    ## Compute the distance between every region-region pair for the hexagon system
    def compute_hexagon_cell_steps(self):
        pass
        
    ## The next adjacent region on the shortest path from origin to dest
    ##      The next move of a region to itself is defined to be itself
    def next_cell_to_move(self, origin, dest):
        return self.next_moves[(origin, dest)]
    
    ## The shortest number of steps to reach dest from origin
    def steps_to_location(self, origin, dest):
        return self.steps_to_loc[(origin, dest)]
    
    ## Get the neighboring cells to the given region that can be reachable within max_steps steps
    def get_neighboring_cells(self, region, max_steps = 5):
        ret = []
        for tup in self.steps_to_loc:
            src, dst = tup
            if src == region and self.steps_to_loc[tup] <= max_steps:
                ret.append(dst)
        return ret
    
    ## Get the adjacent cells to the given region
    def get_adjacent_cells(self, region):
        return self.map[region]
    
    ## Map GPS data to discretized regions
    def gps_to_regions(self, gps_lst, min_lon, max_lon, min_lat, max_lat):
        pass

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
    def __init__(self, time_horizon, parameter_source = "given", arrival_type = "poisson", parameter_fname = "trip_demand.tsv", data = None, start_ts = "00:00:00", end_ts = "23:59:59", smooth_time_window = 3, smooth_region_window = 3):
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
        ## Auxiliary objects
        self.data = None
        ## Populate the arrival rate df
        if parameter_source == "inferred" or arrival_type == "data-driven":
            assert data is not None
            self.infer_parameters()
        if parameter_source == "given":
            self.data = pd.read_csv(f"Data/{self.parameter_fname}", sep = "\t")
    
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
    
    ## Simulate 1 trial of the passenger arrivals
    def generate_arrivals(self):
        if self.arrival_type == "constant":
            return self.generate_constant_arrivals()
        elif self.arrival_type == "poisson":
            return self.generate_poisson_arrivals()
        return self.generate_data_driven_arrivals()

    ## Generate 1 trial of the constant arrival process
    def generate_constant_arrivals(self):
        return self.data[["T", "Origin", "Destination", "Count"]].copy()
    
    ## Generate 1 trial of poisson arrival process
    def generate_poisson_arrivals(self):
        ret = self.data[["T", "Origin", "Destination", "Count"]].copy()
        ret["Count"] = np.random.poisson(lam = ret["Count"])
        return ret
    
    ## TODO: Define the exact behavior of data-driven arrivals
    def generate_data_driven_arrivals(self):
        pass

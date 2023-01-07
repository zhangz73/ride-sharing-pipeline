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
    def __init__(self, map_system = "grid", num_layers = 3, num_nodes = 2, graph_edge_lst = [], lon_range = None, lat_range = None, gps_data = None):
        assert map_system in ["graph", "grid", "hexagon"]
        self.map_system = map_system
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.graph_edge_lst = graph_edge_lst
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.gps_data = gps_data
        self.map = {}
        self.gps_region_map = {}
        self.hexagon_triple_to_id = {}
        self.hexagon_id_to_triple = {}
        
        ## Process Longitude and Latitude ranges
        if self.lon_range is None:
            self.lon_range = (None, None)
        if self.lat_range is None:
            self.lat_range = (None, None)
        if self.gps_data is not None:
            if self.lon_range[0] is None:
                self.lon_range = (self.gps_data["lon"].min(), self.lon_range[1])
            if self.lon_range[1] is None:
                self.lon_range = (self.lon_range[0], self.gps_data["lon"].max())
            if self.lat_range[0] is None:
                self.lat_range = (self.gps_data["lat"].min(), self.lat_range[1])
            if self.lat_range[1] is None:
                self.lat_range = (self.lat_range[0], self.gps_data["lat"].max())
        
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
        
        ## Compute the max travel steps between any two regions
        self.max_travel_steps = 0
        for origin in self.map:
            for dest in self.map:
                self.max_travel_steps = max(self.max_travel_steps, self.steps_to_location(origin, dest))
    
    def assert_valid_lon_lat_range(self):
        assert self.lon_range[0] is not None
        assert self.lon_range[1] is not None
        assert self.lat_range[0] is not None
        assert self.lat_range[1] is not None
    
    ## Get the list of all regions
    def get_regions(self):
        return list(self.map.keys())
    
    ## Get the max travel steps
    def get_max_travel_steps(self):
        return self.max_travel_steps
    
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
    
    ## Construct the map system using the grid system
    def construct_grid_system(self):
        sidelen = 2 * self.num_layers - 1
        num_cells = sidelen ** 2
        for i in range(num_cells):
            vertical = i // sidelen
            horizontal = i % sidelen
            neighbors = []
            if vertical > 0:
                neighbors.append((vertical - 1, horizontal))
            if vertical < sidelen - 1:
                neighbors.append((vertical + 1, horizontal))
            if horizontal > 0:
                neighbors.append((vertical, horizontal - 1))
            if horizontal < sidelen - 1:
                neighbors.append((vertical, horizontal + 1))
            neighbors = [x[0] * sidelen + x[1] for x in neighbors]
            self.map[i] = neighbors
    
    ## Construct the map system using the hexagon system
    def construct_hexagon_system(self):
        curr_id = 0
        for i in range(-self.num_layers + 1, self.num_layers):
            for j in range(-self.num_layers + 1, self.num_layers):
                k = -i - j
                self.hexagon_triple_to_id[(i, j, k)] = curr_id
                self.hexagon_id_to_triple[curr_id] = (i, j, k)
                curr_id += 1
        for id in self.hexagon_id_to_triple:
            curr_triple = self.hexagon_id_to_triple[id]
            x, y, z = curr_triple
            neighbors = [(x + 1, y - 1, z), (x + 1, y, z - 1), (x - 1, y + 1, z), (x - 1, y, z + 1), (x, y + 1, z - 1), (x, y - 1, z + 1)]
            neighbors = [self.hexagon_triple_to_id[(x, y, z)] for x, y, z in neighbors if abs(x) < self.num_layers and abs(y) < self.num_layers and abs(z) < self.num_layers]
            self.map[id] = neighbors
    
    def merge_cells(self):
        pass
    
    ## Compute the next adjacent region from src to dst
    def compute_next_moves(self):
        if self.map_system == "graph":
            self.compute_graph_next_moves()
#        elif self.map_system == "grid":
#            self.compute_grid_next_moves()
#        else:
#            self.compute_hexagon_next_moves()
    
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
#        elif self.map_system == "grid":
#            self.compute_grid_cell_steps()
#        else:
#            self.compute_hexagon_cell_steps()
    
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
        if self.map_system == "graph":
            return self.next_moves[(origin, dest)]
        elif self.map_system == "grid":
            ## Currently, we assume the car matches vertically before matching horizontally
            sidelen = 2 * self.num_layers - 1
            vertical_origin = origin // sidelen
            horizontal_origin = origin % sidelen
            vertical_dest = dest // sidelen
            horizontal_dest = dest % sidelen
            vertical_next = vertical_origin
            horizontal_next = horizontal_origin
            if vertical_origin > vertical_dest:
                vertical_next -= 1
            elif vertical_origin < vertical_dest:
                vertical_next += 1
            elif horizontal_origin > horizontal_dest:
                horizontal_next -= 1
            elif horizontal_origin < horizontal_dest:
                horizontal_next += 1
            return vertical_next * sidelen + horizontal_next
        else:
            ## Currently, we assume that for tuples (x, y, z), we match x first, then y and then z
            x_origin, y_origin, z_origin = self.hexagon_id_to_triple[origin]
            x_dest, y_dest, z_dest = self.hexagon_id_to_triple[dest]
            x_next, y_next, z_next = x_origin, y_origin, z_origin
            update = 0
            if x_origin > x_dest and y_origin < y_dest:
                x_next -= 1
                y_next += 1
            elif x_origin > x_dest and z_origin < z_dest:
                x_next -= 1
                z_next += 1
            elif x_origin < x_dest and y_origin > y_dest:
                x_next += 1
                y_next -= 1
            elif x_origin < x_dest and z_origin > z_dest:
                x_next += 1
                z_next -= 1
            elif y_origin > y_dest and z_origin < z_dest:
                y_next -= 1
                z_next += 1
            elif y_origin < y_dest and z_origin > z_dest:
                y_next += 1
                z_next -= 1
            return self.hexagon_triple_to_id[(x_next, y_next, z_next)]
    
    ## The shortest number of steps to reach dest from origin
    def steps_to_location(self, origin, dest):
        if self.map_system == "graph":
            return self.steps_to_loc[(origin, dest)]
        elif self.map_system == "grid":
            sidelen = 2 * self.num_layers - 1
            vertical_origin = origin // sidelen
            horizontal_origin = origin % sidelen
            vertical_dest = dest // sidelen
            horizontal_dest = dest % sidelen
            return abs(vertical_origin - vertical_dest) + abs(horizontal_origin - horizontal_dest)
        else:
            x_origin, y_origin, z_origin = self.hexagon_id_to_triple[origin]
            x_dest, y_dest, z_dest = self.hexagon_id_to_triple[dest]
            return (abs(x_origin - x_dest) + abs(y_origin - y_dest) + abs(z_origin - z_dest)) / 2
    
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
    
    ## Map a GPS data point to a discretized region
    ##  The gps data point is a tuple of (lon, lat)
    def gps_to_region(self, gps):
        self.assert_valid_lon_lat_range()
        if self.map_system == "grid":
            self.gps_to_region_grid(gps)
        elif self.map_system == "hexagon":
            self.gps_to_region_hexagon(gps)
    
    ## Map a GPS data point to a discretized region under the grid system
    def gps_to_regions_grid(self, gps):
        lon, lat = gps
        sidelen = 2 * self.num_layers - 1
        vertical = int((lat - self.lat_range[0]) / (self.lat_range[1] - self.lat_range[0]) * sidelen)
        horizontal = int((lon - self.lon_range[0]) / (self.lon_range[1] - self.lon_range[0]) * sidelen)
        vertical = min(vertical, sidelen - 1)
        horizontal = min(horizontal, sidelen - 1)
        return vertical * sidelen + horizontal
    
    ## Map a GPS data point to a discretized region under the hexagon system
    def gps_to_regions_hexagon(self, gps):
        lon, lat = gps
        lon_range = self.lon_range[1] - self.lon_range[0]
        lat_range = self.lat_range[1] - self.lat_range[0]
        horizontal_radius = lon_range / 2 / (3/2 * (self.num_layers - 1))
        vertical_radius = lat_range / 2 / (3 ** 0.5 * (self.num_layers - 2) + 3 ** 0.5 / 2)
        radius = max(horizontal_radius, vertical_radius)
        max_range = max(lon_range, lat_range)
        lon_center = (self.lon_range[0] + self.lon_range[1]) / 2
        lat_center = (self.lat_range[0] + self.lat_range[1]) / 2
#        ## Compute steps progressed in longitude
#        ## First chop-off the first element
#        lon_steps = int(abs(lon - lon_center) / radius)
#        if lon_steps > 0:
#            ## Group odd-even numbers into buckets
#            bucket = (lon_steps - 1) // 3
#            ## Compute the position in the bucket
#            pos = (lon_steps - 1) - bucket * 3
#            if pos >= 1:
#                pos = 1
#            ## Obtain the location on the positive side
#            lon_steps = bucket * 3 + pos + 1
#        ## Account for the sign and get the negative counterpart
#        lon_steps = int(2 * ((lon >= lon_steps) - 0.5)) * lon_steps
#        ## Compute steps progressed in latitude
#        lat_steps = int(2 * ((lat >= lat_center) - 0.5) * (int(abs(lat - lat_center) / radius / (3 ** 0.5) * 2) + 1) // 2)
        ## Compute the triple for gps
        ## Moving procedures:
        ## (0, 1, -1) for moving vertically up and (0, -1, 1) for moving vertically down
        ## (-1, 1, 0) for moving upper right and (1, -1, 0) for moving lower left
        ## (1, 0, -1) for moving upper left and (-1, 0, 1) for moving lower right
        x_next, y_next, z_next = 0, 0, 0
        lon_sign = int(2 * ((lon >= lon_center) - 0.5))
        lat_sign = int(2 * ((lat >= lat_center) - 0.5))
        lon_steps = abs(lon - lon_center) / radius / (3/2)
        if lon_steps < 0.5:
            lon_steps = 0
        else:
            lon_steps = int(lon_steps - 0.5) + 1
        lon_steps *= lon_sign
        if lon_sign * lat_sign > 0:
            x_next -= lon_steps
            y_next += lon_steps
            lat_new = lat_center + lon_steps * radius * (3**0.5 / 2)
        else:
            x_next -= lon_steps
            z_next += lon_steps
            lat_new = lat_center - lon_steps * radius * (3**0.5 / 2)
            
        lat_steps2 = abs(lat - lat_new) / radius / (3**0.5)
        if lat_steps2 < 0.5:
            lat_steps2 = 0
        else:
            lat_steps2 = int(lat_steps2 - 0.5) + 1
        if lat < lat_new:
            lat_steps2 = -lat_steps2
        y_next += lat_steps2
        z_next -= lat_steps2
#        x_next = max(min(x_next, self.num_layers - 1), -self.num_layers + 1)
#        y_next = max(min(y_next, self.num_layers - 1), -self.num_layers + 1)
#        z_next = max(min(z_next, self.num_layers - 1), -self.num_layers + 1)
        return self.hexagon_triple_to_id[(x_next, y_next, z_next)]

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

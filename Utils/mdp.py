import numpy as np
import pandas as pd
import torch

## This module stores information of a state
## State Types:
##      1. Car Types
##      2. Trip Types
##      3. Charging Plug Types
##      4. Peak Load
## Functionalties:
##      1. Fill in the state information
##      2. Keep track of counters
##      3. Update the counters
class State:
    def __init__(self, id):
        self.id = id
        self.cnt = 0
    
    def get_id(self):
        return self.id
    
    def get_cnt(self):
        return self.cnt
    
    def set(self, x):
        self.cnt = x

## This module is a child-class of State that is used to decribe car types
class Car(State):
    def __init__(self, id, dest, curr_region, battery, is_filled, type = "general", charged_rate = 0, battery_per_step = 1, time_to_dest = 0):
        super().__init__(id)
        assert type in ["general", "idling", "charged"]
        self.id = id
        self.type = type
        self.dest = dest
        self.all_stops = [dest]
        self.curr_region = curr_region
        self.battery = battery
        self.filled = is_filled
        self.battery_p_step = battery_per_step
        self.charged_rate = charged_rate
        self.time_to_dest = time_to_dest
    
    def get_dest(self):
        if len(self.all_stops) == 0:
            return self.dest
        return self.all_stops[0]
    
    def get_curr_region(self):
        return self.curr_region
    
    def get_battery(self):
        return self.battery
    
    def get_time_to_dest(self):
        return self.time_to_dest
    
    def battery_per_step(self):
        return self.battery_p_step
    
    def is_filled(self):
        return self.filled
    
    def add_stop(self, stop):
        self.all_stops.append(stop)
    
    def get_all_stops(self):
        return self.all_stops
    
    def get_charged_rate(self):
        return self.charged_rate
    
#    def set_charged_rate(self, rate):
#        self.charged_rate = rate
#
#    def unplug(self):
#        self.set_charged_rate(0)
    
    def describe(self):
        if not self.filled:
            msg = f"Car: current region {self.curr_region}, destination {self.dest}, battery {self.battery}, filled {self.filled}"
        else:
            msg = f"Car: time to destination {self.time_to_dest}, destination {self.dest}, battery {self.battery}, filled {self.filled}"
        return msg

## This module is a child-class of State that is used to decribe trip types
class Trip(State):
    def __init__(self, id, origin, dest, stag_time):
        super().__init__(id)
        self.origin = origin
        self.dest = dest
        self.stag_time = stag_time
    
    def get_origin(self):
        return self.origin
    
    def get_dest(self):
        return self.dest
    
    def get_stag_time(self):
        return self.stag_time
    
    def describe(self):
        msg = f"Trip: origin {self.origin}, destination {self.dest}, stag_time {self.stag_time}"
        return msg

## This module is a child-class of State that is used to decribe plug types
class Plug(State):
    def __init__(self, id, region, rate):
        super().__init__(id)
        self.region = region
        self.rate = rate
    
    def get_region(self):
        return self.region
    
    def get_rate(self):
        return self.rate
    
    def describe(self):
        msg = f"Plug: region {self.region}, rate {self.rate}"
        return msg

## This module is a child-class of State that is used to decribe charging load
class ChargingLoad(State):
    def __init__(self, id, load, type = "peak"):
        super().__init__(id)
        assert type in ["peak", "current"]
        self.type = type
        self.load = load
    
    def get_type(self):
        return self.type
    
    def get_load(self):
        return self.load
    
    def set_load(self, load):
        self.load = load
    
    def describe(self):
        msg = f"ChargingLoad: type {self.type}, load {self.load}"
        return msg

class Timestamp(State):
    def __init__(self, id):
        super().__init__(id)
    
    def describe(self):
        return "Timestamp: -"

## This module stores information of an action
## Action Types:
##      1. Travel Action
##      2. Charge Action
## Functionalties:
##      1. Fill in the action information
##      2. Keep track of counters
##      3. Update the counters
class Action:
    def __init__(self, id):
        self.id = id
    
    def get_id(self):
        return self.id
    
    def get_type(self):
        return "action"

## This module is a child-class of Action that is used to decribe the travel action
class Travel(Action):
    def __init__(self, id, origin, dest, is_filled):
        super().__init__(id)
        self.id = id
        self.origin = origin
        self.dest = dest
        self.filled = is_filled
    
    def get_origin(self):
        return self.origin
    
    def get_dest(self):
        return self.dest
    
    def is_filled(self):
        return self.filled
    
    def get_type(self):
        if self.filled:
            return "pickup"
        if self.origin != self.dest:
            return "rerouting"
        return "idling"
    
    def describe(self):
        return f"{self.get_type().title()} from {self.get_origin()} to {self.get_dest()}"

## This module is a child-class of Action that is used to describe the charging action
class Charge(Action):
    def __init__(self, id, region, rate):
        super().__init__(id)
        self.id = id
        self.region = region
        self.rate = rate
    
    def get_region(self):
        return self.region
    
    def get_rate(self):
        return self.rate
    
    def get_type(self):
        return "charged"
    
    def describe(self):
        return f"Charging in region {self.region} with rate {self.rate}"

## This module is a child-class of Action that is used to describe the nothing action
class Nothing(Action):
    def __init__(self, id):
        super().__init__(id)
        self.id = id
    
    def get_type(self):
        return "nothing"
    
    def describe(self):
        return "No new actions applied"

## This module allows users to query rewards for each atomic action
class Reward:
    def __init__(self, reward_fname = "payoff.tsv"):
        self.reward_fname = reward_fname
        self.reward_df = pd.read_csv(f"Data/{reward_fname}", sep = "\t")
    
    ## Return payoff given action and timestamp
    def query(self, action, timestamp):
        if action.get_type() == "charged":
            region, rate = action.get_region(), action.get_rate()
            tmp_df = self.reward_df[(self.reward_df["Region"] == region) & (self.reward_df["Rate"] == rate) & (self.reward_df["Type"] == "Charge")]
        else:
            origin, dest = action.get_origin(), action.get_dest()
            tmp_df = self.reward_df[(self.reward_df["Origin"] == origin) & (self.reward_df["Destination"] == dest) & (self.reward_df["Type"] == "Travel")]
            if action.get_type() == "pickup":
                tmp_df = tmp_df[tmp_df["Pickup"] == 1]
            else:
                tmp_df = tmp_df[tmp_df["Pickup"] != 1]
        if tmp_df.shape[0] > 0:
            return tmp_df.iloc[0]["Payoff"]
        return 0
    
    ## Compute the total payoff given a list of atomic actions and a timestamp
    def atomic_actions_to_payoff(self, action_lst, ts):
        total_payoff = 0
        for action in action_lst:
            curr_payoff = self.query(action, ts)
            total_payoff += curr_payoff
        return total_payoff

## This module defines the Markov decision process for state transitions
## Functionalities:
##      1. Construct all state types at the beginning
##      2. Progress state transitions within timestamp
##      3. Progress state transitions across timestamp
## Inputs:
##      map: The Map object containing the geographic info
##      trip_demands: The TripDemands object for inferring and generating trip requests
##      time_horizon: The number of time discretizations
##      connection_patience: The maximum time passengers can wait for the order to be handled
##      pickup_patience: The maximum time passengers can wait for the driver once the order has been handled
##      num_battery_levels: The number of battery level discretizations
##      battery_jump: The change in battery level per discretization
##      charging_rates: The possible charging rates for charging plugs. In multiple of battery_jump
##      region_battery_car_num: The number of cars of each battery level at each region deployed initially. Assume all cars start with dest == curr_region, is_filled = 0. (region, battery) -> num
##      region_rate_plug_num: The number of charging plugs of each rate at each region. (region, rate) -> num
class MarkovDecisionProcess:
    def __init__(self, map, trip_demands, reward_query, time_horizon, connection_patience, pickup_patience, num_battery_levels, battery_jump, charging_rates, battery_offset = 1, region_battery_car_fname = "region_battery_car.tsv", region_rate_plug_fname = "region_rate_plug.tsv"):
        self.map = map
        self.trip_demands = trip_demands
        self.reward_query = reward_query
        self.time_horizon = time_horizon
        self.connection_patience = connection_patience
        self.pickup_patience = pickup_patience
        self.num_battery_levels = num_battery_levels
        self.charging_rates = charging_rates
        self.num_charging_rates = len(charging_rates)
        self.battery_jump = battery_jump
        self.region_battery_car_num = None
        self.region_rate_plug_num = None
        self.battery_offset = battery_offset
        self.region_battery_car_fname = region_battery_car_fname
        self.region_rate_plug_fname = region_rate_plug_fname
        self.load_initial_data()
        ## Auxiliary variables
        self.regions = self.map.get_regions()
        self.max_travel_time = self.map.get_max_travel_steps()
        self.num_charging_rates = len(self.charging_rates)
        self.num_car_states = len(self.regions) ** 2 * self.num_battery_levels * 1 + len(self.regions) * self.num_battery_levels * 2 + len(self.regions) * self.num_battery_levels * (self.pickup_patience + self.max_travel_time + 1) * 2
        self.num_trip_states = len(self.regions) ** 2 * (self.connection_patience + 1)
        self.num_plug_states = len(self.regions) * self.num_charging_rates
        self.num_chargingload_states = 2
        self.num_total_states = self.num_car_states + self.num_trip_states + self.num_plug_states + self.num_chargingload_states + 1
        ## Variables keeping track of states
        self.state_dict = {}
        self.state_to_id = {}
        self.state_counts = torch.zeros(self.num_total_states)
        ## Variables keeping track of current timestamp
        self.curr_ts = 0
        self.reset_timestamp()
        ## Populate state variables
        self.define_all_states()
        ## Store a copy of previous state counts for reverting actions
        self.state_counts_prev = self.state_counts.clone()
        self.prev_ts = 0
        self.state_counts_init = self.state_counts.clone()
        ## Variables keeping track of available car types
        self.available_car_types, self.state_is_available_car = self.get_all_available_car_types()
        self.available_existing_car_types = self.get_all_available_existing_car_types()
        ## Variables keeping track of all actions
        self.all_actions = {}
        self.action_to_id = {}
        self.action_desc_to_id = {}
        self.construct_all_actions()
        self.action_lst = [self.all_actions[k] for k in self.all_actions.keys()]
        ## Variables keeping track of per timestamp payoffs
        self.payoff_map = torch.zeros((self.time_horizon, len(self.action_lst)))
        self.construct_payoff_map()
        ## TODO: Populate it at each state transition epoch
        self.payoff_curr_ts = torch.tensor(0.)
        self.payoff_schedule_dct = {}
        for t in range(self.time_horizon):
            self.payoff_schedule_dct[t] = 0 #{"pickup": 0, "reroute": 0, "charge": 0, "idle": 0}
    
    ## Reset all states to the initial one
    def reset_states(self):
        self.state_counts = self.state_counts_init.clone()
        self.state_counts_prev = self.state_counts.clone()
        self.available_existing_car_types = self.get_all_available_existing_car_types()
        self.reset_timestamp()
        self.payoff_curr_ts = torch.tensor(0.)
    
    ## Describe the state_counts
    def describe_state_counts(self, state_counts = None, indent = "\t"):
        msg = ""
        if state_counts is None:
            state_counts = self.state_counts
        for id in self.state_dict:
            if state_counts[id] > 0:
                state = self.state_dict[id]
                val = float(state_counts[id])
                msg += indent + f"val = {val} " + state.describe() + "\n"
        return msg
    
    ## A helper function that converts a dataframe to a dictionary
    ##  with the specified key-value format
    def df_to_dct(self, df, keynames = [], valname = None):
        dct = {}
        for i in range(df.shape[0]):
            lst = []
            val = df.iloc[i][valname]
            for keyname in keynames:
                lst.append(df.iloc[i][keyname])
            tup = tuple(lst)
            dct[tup] = val
        return dct
    
    ## Load region_battery_car_num and region_rate_plug_num from files
    def load_initial_data(self):
        region_battery_car_df = pd.read_csv(f"Data/{self.region_battery_car_fname}", sep = "\t")
        region_rate_plug_df = pd.read_csv(f"Data/{self.region_rate_plug_fname}", sep = "\t")
        self.region_battery_car_num = self.df_to_dct(region_battery_car_df, keynames = ["region", "battery"], valname = "num")
        self.region_rate_plug_num = self.df_to_dct(region_rate_plug_df, keynames = ["region", "rate"], valname = "num")
    
    ## Construct the list of all actions
    def construct_all_actions(self):
        curr_id = 0
        ## Construct all travel actions
        for origin in self.regions:
            for dest in self.regions:
                for is_filled in [0, 1]:
                    action = Travel(id = curr_id, origin = origin, dest = dest, is_filled = is_filled)
                    self.all_actions[curr_id] = action
                    self.action_to_id[action] = curr_id
                    self.action_desc_to_id[("travel", origin, dest, is_filled)] = curr_id
                    curr_id += 1
        ## Construct all charge actions
        for region in self.regions:
            for rate in self.charging_rates:
                action = Charge(id = curr_id, region = region, rate = rate)
                self.all_actions[curr_id] = action
                self.action_to_id[action] = curr_id
                self.action_desc_to_id[("charge", region, rate)] = curr_id
                curr_id += 1
        ## Construct the nothing action
        action = Nothing(id = curr_id)
        self.all_actions[curr_id] = action
        self.action_to_id[action] = curr_id
        self.action_desc_to_id[("nothing")] = curr_id
    
    ## Construct a map of payoffs (per timestamp) for atomic actions
    def construct_payoff_map(self):
        for t in range(self.time_horizon):
            for action_id in self.all_actions:
                action = self.all_actions[action_id]
                if action.get_type() == "nothing":
                    self.payoff_map[t, action_id] = 0
                else:
                    self.payoff_map[t, action_id] = self.reward_query.query(action, t)
    
    ## Return the list of all possible actions
    def get_all_actions(self):
        return self.all_actions
    
    ## Return the total number of cars in the system
    def get_num_cars(self):
        return np.sum(self.region_battery_car_num)
    
    ## Return the payoff at the current timestamp
    def get_payoff_curr_ts(self):
        return self.payoff_curr_ts
    
    ## Zero out the payoff at the current timestamp
    def reset_payoff_curr_ts(self):
        self.payoff_curr_ts = torch.tensor(0.)
    
    ## Reset timestamp to 0 and regenerate trip arrivals
    def reset_timestamp(self):
        self.curr_ts = 0
        self.trip_arrivals = self.trip_demands.generate_arrivals()
        self.payoff_curr_ts = torch.tensor(0.)
#        self.payoff_schedule_dct = {}
#        for t in range(self.time_horizon):
#            self.payoff_schedule_dct[t] = 0
    
    ## Revert the action performed on the states
    ## Deprecated!!!
    def revert_action(self):
        self.state_counts = self.state_counts_prev.clone()
        self.available_existing_car_types = self.get_all_available_existing_car_types()
        self.curr_ts = self.prev_ts
    
    ## Set the current state to be the base-state in case of rollback
    ## Deprecated!!!
    def step_action(self):
        self.state_counts_prev = self.state_counts.clone()
        self.prev_ts = self.curr_ts
    
    ## Set the states and payoff_schedule_dct according to the given ones
    def set_states(self, state_counts, ts, payoff_curr_ts = None):
        self.state_counts = state_counts.clone()
        self.available_existing_car_types = self.get_all_available_existing_car_types()
        if payoff_curr_ts is None:
            payoff_curr_ts = torch.tensor(0.)
        self.payoff_curr_ts = payoff_curr_ts
        self.curr_ts = ts
        self.prev_ts = ts
    
    ## Construct all states at the beginning
    ##  Car states: |region| x |region| x |battery level| x 2 + |region| x |battery level| x 2
    ##  Trip states: |region| x |region| x (max connection patience + 1)
    ##  Plug states: |region| x |rates|
    ##  Charging Load states: 2 (peak & current)
    def define_all_states(self):
        curr_id = 0
        ## Define car states -- General type for empty cars
        self.state_to_id["car"] = {}
        for dest in self.regions:
            for curr_region in self.regions:
                for battery in range(self.num_battery_levels):
                    #for is_filled in [0, 1]:
                    car = Car(curr_id, dest, curr_region, battery, 0, type = "general")
                    self.state_to_id["car"][(dest, curr_region, battery, 0, "general")] = curr_id
                    self.state_dict[curr_id] = car
                    curr_id += 1
        
        ## Define car states -- General type for filled cars
        for dest in self.regions:
            for eta in range(self.pickup_patience + self.max_travel_time + 1):
                for battery in range(self.num_battery_levels):
                    #for is_filled in [0, 1]:
                    car = Car(curr_id, dest, dest, battery, 1, type = "general", time_to_dest = eta)
                    self.state_to_id["car"][(dest, eta, battery, 1, "general")] = curr_id
                    self.state_dict[curr_id] = car
                    curr_id += 1
        
        ## Populate state counts for initial car deployment
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                if (region, battery) in self.region_battery_car_num:
                    cnt = self.region_battery_car_num[(region, battery)]
                else:
                    cnt = 0
                id = self.state_to_id["car"][(region, region, battery, 0, "general")]
                self.state_counts[id] = cnt
        
        ## Define car states -- idling
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                car = Car(curr_id, region, region, battery, 0, type = "idling")
                self.state_to_id["car"][(region, region, battery, 0, "idling")] = curr_id
                self.state_dict[curr_id] = car
                curr_id += 1
                
        ## Define car states -- idling type for filled cars
        for dest in self.regions:
            for eta in range(self.pickup_patience + self.max_travel_time + 1):
                for battery in range(self.num_battery_levels):
                    #for is_filled in [0, 1]:
                    car = Car(curr_id, dest, dest, battery, 1, type = "idling", time_to_dest = eta)
                    self.state_to_id["car"][(dest, eta, battery, 1, "idling")] = curr_id
                    self.state_dict[curr_id] = car
                    curr_id += 1
        
        ## Define car states -- charged
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                for rate in self.charging_rates:
                    car = Car(curr_id, region, region, battery, 0, type = "charged", charged_rate = rate)
                    self.state_to_id["car"][(region, region, battery, 0, "charged", rate)] = curr_id
                    self.state_dict[curr_id] = car
                    curr_id += 1
        
        ## Define trip states
        self.state_to_id["trip"] = {}
        for origin in self.regions:
            for dest in self.regions:
                for stag_time in range(self.connection_patience + 1):
                    trip = Trip(curr_id, origin, dest, stag_time)
                    self.state_to_id["trip"][(origin, dest, stag_time)] = curr_id
                    self.state_dict[curr_id] = trip
                    curr_id += 1
                ## Load new passenger requests
                trip_id_new = self.state_to_id["trip"][(origin, dest, 0)]
                tmp_df = self.trip_arrivals[(self.trip_arrivals["T"] == 0) & (self.trip_arrivals["Origin"] == origin) & (self.trip_arrivals["Destination"] == dest)]
                if tmp_df.shape[0] > 0:
                    self.state_counts[trip_id_new] = tmp_df.iloc[0]["Count"]
                else:
                    self.state_counts[trip_id_new] = 0
        
        ## Define plug states
        self.state_to_id["plug"] = {}
        for region in self.regions:
            for rate in self.charging_rates:
                plug = Plug(curr_id, region, rate)
                ## Set the number of available plugs
                if (region, rate) in self.region_rate_plug_num:
                    plug_num = self.region_rate_plug_num[(region, rate)]
                else:
                    plug_num = 0
                plug.set(plug_num)
                self.state_counts[curr_id] = plug_num
                self.state_to_id["plug"][(region, rate)] = curr_id
                self.state_dict[curr_id] = plug
                curr_id += 1
        
        ## Define charging load states
        self.state_to_id["charging_load"] = {}
        for type in ["current", "peak"]:
            load = ChargingLoad(curr_id, 0, type = type)
            self.state_to_id["charging_load"][type] = curr_id
            self.state_dict[curr_id] = load
            curr_id += 1
        
        ## Define timestamp state
        self.state_to_id["timestamp"] = curr_id
        time = Timestamp(id = curr_id)
        self.state_dict[curr_id] = time
        self.state_counts[curr_id] = 0
    
    ## Atomic state transitions within a timestamp
    ##  The action, given by a solver, is performed on each individual car
    ## Options:
    ##      - Picking up a passenger request
    ##          Feasibility:
    ##              Time to arrival <= pickup_patience
    ##              remaining battery >= battery required for travel
    ##              Has someone to be picked up
    ##          State Transitions:
    ##              Original car type - 1
    ##              New car type (with the certain destination) + 1
    ##              Trip type - 1
    ##      - Rerouting
    ##          Feasibility:
    ##              remaining battery >= battery required for travel
    ##          State Transitions:
    ##              Original car type - 1
    ##              New car type (with the certain destination) + 1
    ##      - To be plugged in
    ##          Feasibility:
    ##              Charging plug type >= 1
    ##          State Transitions:
    ##              Original car type - 1
    ##              New car type (charged) + 1
    ##              Current charging load + charging_rate * battery_jump
    ##              Max charging load = max(Max charging load, Current charging load)
    ##      - Idling
    ##          Feasibility:
    ##              Always feasible
    ##          State Transitions:
    ##              Original car type - 1
    ##              New car type (idling) + 1
    ## Return if the action has been successfully processed. False if the action is not feasible
    def transit_within_timestamp(self, action, car_id = None, debug = False):
        if action.get_type() == "nothing":
            return self.transit_travel_within_timestamp(action, "nothing", car_id = car_id, debug = debug)
        if action.get_type() == "pickup":
            return self.transit_travel_within_timestamp(action, "pickup", car_id = car_id, debug = debug)
        elif action.get_type() == "rerouting":
            return self.transit_travel_within_timestamp(action, "rerouting", car_id = car_id, debug = debug)
        elif action.get_type() == "idling":
            return self.transit_travel_within_timestamp(action, "idling", car_id = car_id, debug = debug)
        return self.transit_charged_within_timestamp(action, car_id = car_id)
    
    ## Atomic state transitions for pickup action within timestamp
    def transit_travel_within_timestamp(self, action, type, car_id = None, debug = False):
        if type != "nothing":
            origin, dest = action.get_origin(), action.get_dest()
        else:
            origin, dest = None, None
        car_type_idx = 0
        if car_id is None:
            available_car_lst = self.available_existing_car_types
        else:
            available_car_lst = [car_id]
        ## Get the first available existing car in the queue
        while car_type_idx < len(available_car_lst):
            ## Check for feasibility
            ##  time to arrival <= pickup_patience
            ##  car destination matches the origin of the action
            ##  remaining battery >= battery required for travel
            ##  car has to be empty in order to reroute or idle
            id = available_car_lst[car_type_idx]
            car = self.state_dict[id]
            ## Compute time to arrival and compare it with pickup patience
            #total_time_to_arrival = self.map.steps_to_location(car.get_curr_region(), car.get_dest()) + self.map.steps_to_location(origin, dest)
            if type in ["pickup"]:
                if car.is_filled():
                    time_to_arrival = car.get_time_to_dest() #self.map.steps_to_location(car.get_curr_region(), car.get_dest()) #
                else:
                    time_to_arrival = self.map.steps_to_location(car.get_curr_region(), origin)
                total_time_to_arrival = time_to_arrival + self.map.steps_to_location(origin, dest)
                close_to_dest = time_to_arrival <= self.pickup_patience #
            elif type in ["idling", "rerouting"]:
                close_to_dest = car.get_curr_region() == origin
                total_time_to_arrival = self.map.steps_to_location(origin, dest)
            else:
                close_to_dest = True
            ## Has someone to be picked up
            if type != "nothing":
                has_someone_to_pickup = False
                for stag_time in range(self.connection_patience, -1, -1):
                    trip_id = self.state_to_id["trip"][(origin, dest, stag_time)]
                    if self.state_counts[trip_id] > 0:
                        has_someone_to_pickup = True
                        break
                if type in ["idling", "rerouting"]:
                    has_someone_to_pickup = not has_someone_to_pickup
            else:
                has_someone_to_pickup = True
            ## Compute the battery required for travel
            if type in ["pickup", "rerouting"]:
                battery_required = car.battery_per_step() * total_time_to_arrival + self.battery_offset
                enough_battery = battery_required <= car.get_battery()
            else:
                enough_battery = True
            ## Check if car can head to the pickup origin
            can_reach_origin = car.get_dest() == origin or not car.is_filled()
            ## Check if car is filled
            if type in ["pickup", "nothing"]:
                car_not_filled = True
            else:
                car_not_filled = not car.is_filled()
            ## For the "nothing" action only
            if type == "nothing":
                if not car.is_filled():
                    nothing_action_feasible = False
                else:
                    nothing_action_feasible = True
            else:
                nothing_action_feasible = True
            ## Check all feasibility
            if close_to_dest and enough_battery and car_not_filled and has_someone_to_pickup and nothing_action_feasible:
                self.state_counts[id] -= 1
                ## Compute target car type
                if type not in ["idling", "nothing"]:
                    car_query_type = "general"
                else:
                    car_query_type = "idling"
                if type == "pickup":
                    car_filled_status = 1
                    target_car_state = (dest, total_time_to_arrival, car.get_battery(), 1, car_query_type)
                elif type in ["rerouting", "idling"]:
                    car_filled_status = 0
                    target_car_state = (dest, car.get_curr_region(), car.get_battery(), 0, car_query_type)
                else:
                    car_filled_status = 1
                    target_car_state = (car.get_dest(), car.get_time_to_dest(), car.get_battery(), 1, car_query_type)
                target_car_id = self.state_to_id["car"][target_car_state]
                self.state_counts[target_car_id] += 1
                ## Update payment scheduling dct
                if type != "nothing":
                    action_id = self.action_desc_to_id[("travel", origin, dest, car_filled_status)]
                    if type == "pickup":
                        ## First reroute
                        if not car.is_filled():
                            curr_region = car.get_curr_region()
                            time_len = self.map.steps_to_location(curr_region, origin)
                            reroute_action_id = self.action_desc_to_id[("travel", curr_region, origin, 0)]
                            reroute_payoff = self.payoff_map[self.curr_ts, reroute_action_id]
                            self.payoff_curr_ts += reroute_payoff
    #                        for t in range(time_len):
    #                            ts = self.curr_ts + t
    #                            self.payoff_schedule_dct[ts] += reroute_payoff / time_len
                        ## Then take new passengers
                        pickup_payoff = self.payoff_map[self.curr_ts, action_id]
                        self.payoff_curr_ts += pickup_payoff
    #                    for t in range(car.get_time_to_dest(), total_time_to_arrival):
    #                        ts = self.curr_ts + t
    #                        time_len = total_time_to_arrival - car.get_time_to_dest()
    #                        self.payoff_schedule_dct[ts] += pickup_payoff / time_len
                    else:
                        next_region = self.map.next_cell_to_move(car.get_curr_region(), dest)
#                    self.payoff_schedule_dct[self.curr_ts] += self.payoff_map[self.curr_ts, action_id]
                ## Update trip counts
                if type == "pickup":
                    stag_time = self.connection_patience
                    action_fulfilled = False
                    while stag_time >= 0 and not action_fulfilled:
                        trip_id = self.state_to_id["trip"][(origin, dest, stag_time)]
                        if self.state_counts[trip_id] > 0:
                            self.state_counts[trip_id] -= 1
                            action_fulfilled = True
                        stag_time -= 1
                ## Update available_existing_car_types
                if self.state_counts[id] == 0:
                    available_car_lst = available_car_lst[:car_type_idx] + available_car_lst[(car_type_idx + 1):]
                    car_type_idx -= 1
                    if car_id is None:
                        self.available_existing_car_types = available_car_lst
                return True
            car_type_idx += 1
        return False
    
    ## Atomic state transitions for charged action within timestamp
    def transit_charged_within_timestamp(self, action, car_id = None):
        region, rate = action.get_region(), action.get_rate()
        car_type_idx = 0
        plug_id = self.state_to_id["plug"][(region, rate)]
        if car_id is None:
            available_car_lst = self.available_existing_car_types
        else:
            available_car_lst = [car_id]
        ## Get the first available existing car in the queue
        while car_type_idx < len(available_car_lst):
            ## Check for feasibility
            ##  charging plug >= 1
            ##  car is already at the charging region
            ##  car is not filled
            id = available_car_lst[car_type_idx]
            car = self.state_dict[id]
            ## Check if the charging plug is still available
            has_plug = self.state_counts[plug_id] > 0
            ## Check if the car is already at the region
            car_at_region = car.get_curr_region() == region
            ## Check if the car is empty
            car_not_filled = not car.is_filled()
            ## Check all feasibility
            if has_plug and car_at_region and car_not_filled:
                self.state_counts[id] -= 1
                ## Compute target car type
                if type != "idling":
                    car_query_type = "general"
                else:
                    car_query_type = "idling"
                target_car_state = (region, region, car.get_battery(), 0, "charged", rate)
                target_car_id = self.state_to_id["car"][target_car_state]
                self.state_counts[target_car_id] += 1
                ## Update available_existing_car_types
                if self.state_counts[id] == 0:
                    available_car_lst = available_car_lst[:car_type_idx] + available_car_lst[(car_type_idx + 1):]
                    car_type_idx -= 1
                    if car_id is None:
                        self.available_existing_car_types = available_car_lst
                ## Update plug counts
                self.state_counts[plug_id] -= 1
                ## Update charging load
                peak_load_id = self.state_to_id["charging_load"]["peak"]
                current_load_id = self.state_to_id["charging_load"]["current"]
                self.state_counts[current_load_id] += rate
                self.state_counts[peak_load_id] = max(self.state_counts[peak_load_id], self.state_counts[current_load_id])
#                ## Set charging rate to the current car
#                car.set_charged_rate(rate)
                return True
            car_type_idx += 1
        return False
    
    ## State transitions on multiple cars within the timestamp
    ##  More likely used by matching-based solvers
    def transit_group_within_timestamp(self, car_type_id_lst, action_lst):
        assert len(car_type_id_lst) == len(action_lst)
        ## Save a copy of state counts in case of rollback
        state_counts_init = self.state_counts.clone()
        ## Iterate through both lists
        for car_type_id, action in zip(car_type_id_lst, action_lst):
            if action.get_type() == "charged":
                atomic_action_success = self.transit_charged_within_timestamp(action, car_id = car_type_id)
            else:
                atomic_action_success = self.transit_travel_within_timestamp(action, type = action.get_type(), car_id = car_type_id)
            if not atomic_action_success:
                self.state_counts = state_counts_init
                return False
        return True
    
    ## State transitions across timestamps
    ##  No actions given by solvers. Just refreshing state values
    ## Procedures:
    ##      - Update passenger requests: new requests + carry-over requests - abandoned requests
    ##      - Occupied cars make a movement by getting closer and consuming power
    ##      - Empty cars (empty, idling, charged) make a movement by getting closer and consuming power
    ##      - Charged cars gain power
    ##      - Zero out the idling and charged cars
    ##      - Set occupied cars to empty if they have arrived at destinations
    ##      - Reset all charging plug numbers to the given ones (i.e. all plugs are available)
    ##      - Zero out the current charging load
    def transit_across_timestamp(self):
        assert self.curr_ts < self.time_horizon
        state_counts_new = torch.zeros(self.num_total_states)
        ## Update passenger requests
        for origin in self.regions:
            for dest in self.regions:
                for stag_time in range(self.connection_patience, 0, -1):
                    trip_id_curr = self.state_to_id["trip"][(origin, dest, stag_time)]
                    trip_id_prev = self.state_to_id["trip"][(origin, dest, stag_time - 1)]
                    state_counts_new[trip_id_curr] = self.state_counts[trip_id_prev]
                ## Load new passenger requests
                trip_id_new = self.state_to_id["trip"][(origin, dest, 0)]
                if self.curr_ts + 1 < self.time_horizon:
                    tmp_df = self.trip_arrivals[(self.trip_arrivals["T"] == self.curr_ts + 1) & (self.trip_arrivals["Origin"] == origin) & (self.trip_arrivals["Destination"] == dest)]
                    if tmp_df.shape[0] > 0:
                        state_counts_new[trip_id_new] = tmp_df.iloc[0]["Count"]
                    else:
                        state_counts_new[trip_id_new] = 0
        ## Make movements for traveling empty cars
        for dest in self.regions:
            for curr_region in self.regions:
                for battery in range(self.num_battery_levels):
                    #for is_filled in [0, 1]:
                    car_id_curr = self.state_to_id["car"][(dest, curr_region, battery, 0, "general")]
                    car_curr = self.state_dict[car_id_curr]
                    if self.state_counts[car_id_curr] > 0:
                        ## Update payoff
                        next_region = self.map.next_cell_to_move(curr_region, dest)
                        curr_action_id = self.action_desc_to_id[("travel", curr_region, next_region, 0)]
                        self.payoff_curr_ts += self.payoff_map[self.curr_ts, curr_action_id] * self.state_counts[car_id_curr]
                        ## Update states
                        if next_region == curr_region:
                            next_battery = battery
                        else:
                            next_battery = battery - car_curr.battery_per_step()
                        next_state = (dest, next_region, next_battery, 0, "general")
                        if next_state in self.state_to_id["car"]:
                            car_id_next = self.state_to_id["car"][next_state]
                            state_counts_new[car_id_next] += self.state_counts[car_id_curr]
        ## Gather do nothing cars
        for dest in self.regions:
            for eta in range(1, self.pickup_patience + self.max_travel_time + 1):
                for battery in range(self.num_battery_levels):
                    car_id_curr = self.state_to_id["car"][(dest, eta, battery, 1, "idling")]
                    car_id_general = self.state_to_id["car"][(dest, eta, battery, 1, "general")]
                    state_counts_new[car_id_general] += self.state_counts[car_id_curr]
        ## Make movements for traveling filled cars
        for dest in self.regions:
            for eta in range(1, self.pickup_patience + self.max_travel_time + 1):
                for battery in range(self.num_battery_levels):
                    #for is_filled in [0, 1]:
                    car_id_curr = self.state_to_id["car"][(dest, eta, battery, 1, "general")]
                    car_curr = self.state_dict[car_id_curr]
                    if self.state_counts[car_id_curr] > 0:
                        next_battery = battery - car_curr.battery_per_step()
                        next_state = (dest, eta - 1, next_battery, 1, "general")
                        if next_state in self.state_to_id["car"]:
                            car_id_next = self.state_to_id["car"][next_state]
                            state_counts_new[car_id_next] += self.state_counts[car_id_curr]
        ## Drop-off passengers
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                car_id_filled = self.state_to_id["car"][(region, 0, battery, 1, "general")]
                car_id_empty = self.state_to_id["car"][(region, region, battery, 0, "general")]
                state_counts_new[car_id_empty] += state_counts_new[car_id_filled]
                state_counts_new[car_id_filled] = 0
        ## Gather idling cars
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                car_id_curr = self.state_to_id["car"][(region, region, battery, 0, "idling")]
                car_id_general = self.state_to_id["car"][(region, region, battery, 0, "general")]
                state_counts_new[car_id_general] += self.state_counts[car_id_curr]
        ## Gather charged cars
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                for rate in self.charging_rates:
                    car_id_curr = self.state_to_id["car"][(region, region, battery, 0, "charged", rate)]
                    next_battery = battery
                    ## charged cars gain power
                    car_curr = self.state_dict[car_id_curr]
                    next_battery += rate
                    next_battery = min(next_battery, self.num_battery_levels - 1)
                    ## Update payoff
                    curr_action_id = self.action_desc_to_id[("charge", region, rate)]
                    self.payoff_curr_ts += self.payoff_map[self.curr_ts, curr_action_id] * self.state_counts[car_id_curr]
#                        ## Unplug the cars
#                        car_curr.unplug()
                    car_id_general = self.state_to_id["car"][(region, region, next_battery, 0, "general")]
                    state_counts_new[car_id_general] += self.state_counts[car_id_curr]
        ## Reset charging plugs
        for region in self.regions:
            for rate in self.charging_rates:
                plug_id = self.state_to_id["plug"][(region, rate)]
                if (region, rate) in self.region_rate_plug_num:
                    plug_num = self.region_rate_plug_num[(region, rate)]
                else:
                    plug_num = 0
                state_counts_new[plug_id] = plug_num
        ## Set peak charging load
        peak_load_id = self.state_to_id["charging_load"]["peak"]
        state_counts_new[peak_load_id] = self.state_counts[peak_load_id]
        ## Set timestamp
        time_id = self.state_to_id["timestamp"]
        state_counts_new[time_id] = self.curr_ts + 1
        ## Update state counts
        self.state_counts = state_counts_new
        ## Recompute existing available cars
        self.available_existing_car_types = self.get_all_available_existing_car_types()
        ## Increment timestamp by 1
        self.curr_ts += 1
    
    ## Check if a given car is available
    def is_available_car(self, car):
        car_is_filled = car.is_filled()
        curr_region = car.get_curr_region()
        dest = car.get_dest()
        eta = car.get_time_to_dest()
        close_to_dest = eta <= self.pickup_patience
        return not car_is_filled or (car_is_filled and close_to_dest)

    ## Get a list of the ids of all available car types
    def get_all_available_car_types(self):
        ret = []
        state_is_available_car = torch.zeros(len(self.state_counts))
        for car_type in self.state_to_id["car"]:
            id = self.state_to_id["car"][car_type]
            car = self.state_dict[id]
            if car_type[4] == "general" and self.is_available_car(car):
                ret.append(id)
                state_is_available_car[id] = 1
        return ret, state_is_available_car
    
    ## Get a list of the ids of all available car types that are non-empty
    def get_all_available_existing_car_types(self):
#        ret = []
#        for id in self.available_car_types:
#            if self.state_counts[id] > 0:
#                ret.append(id)
#        self.available_existing_car_types = ret
        ret = (self.state_counts * self.state_is_available_car > 0).nonzero(as_tuple = True)[0]
        return list(ret.numpy())

    ## Get a count of all available cars
    def get_available_car_counts(self):
#        cnt = 0
#        for id in self.available_car_types:
#            cnt += int(self.state_counts[id])
        cnt = torch.sum(self.state_counts[self.available_car_types])
        return int(cnt.data)

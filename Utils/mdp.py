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
    def __init__(self, id, dest, curr_region, battery, is_filled, type = "general", battery_per_step = 1):
        super().__init__(id)
        assert type in ["general", "idling", "charged"]
        self.id = id
        self.type = type
        self.dest = dest
        self.curr_region = curr_region
        self.battery = battery
        self.is_filled = is_filled
        self.battery_per_step = battery_per_step
    
    def get_dest(self):
        return self.dest
    
    def get_curr_region(self):
        return self.curr_region
    
    def get_battery(self):
        return self.battery
    
    def battery_per_step(self):
        return self.battery_per_step
    
    def is_filled(self):
        return self.is_filled

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

## This module is a child-class of State that is used to decribe charging load
class ChargingLoad(State):
    def __init__(self, id, type = "peak", load):
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
        self.is_filled = is_filled
    
    def get_origin(self):
        return origin
    
    def get_dest(self):
        return dest
    
    def is_filled(self):
        return self.is_filled
    
    def get_type(self):
        if self.is_filled:
            return "pickup"
        if self.origin != self.dest:
            return "rerouting"
        return "idling"

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
    def __init__(self, map, trip_demands, time_horizon, connection_patience, pickup_patience, num_battery_levels, battery_jump, charging_rates, region_battery_car_num, region_rate_plug_num, battery_offset = 1):
        self.map = map
        self.trip_demands = trip_demands
        self.time_horizon = time_horizon
        self.connection_patience = connection_patience
        self.pickup_patience = pickup_patience
        self.num_battery_levels = num_battery_levels
        self.charging_rates = charging_rates
        self.battery_jump = battery_jump
        self.region_battery_car_num = region_battery_car_num
        self.region_rate_plug_num = region_rate_plug_num
        self.battery_offset = battery_offset
        ## Auxiliary variables
        self.regions = self.map.get_regions()
        self.num_charging_rates = len(self.charging_rates)
        self.num_car_states = len(self.regions) ** 2 * self.num_battery_levels * 2 + len(self.regions) * self.num_battery_levels * 2
        self.num_trip_states = len(self.regions) ** 2 * (self.connection_patience + 1)
        self.num_plug_states = len(self.regions) * num_charging_rates
        self.num_chargingload_states = 2
        self.num_total_states = self.num_car_states + self.num_trip_states + self.num_plug_states + self.num_chargingload_states
        ## Variables keeping track of states
        self.state_dict = {}
        self.state_to_id = {}
        self.state_counts = torch.zeros(self.num_total_states)
        ## Populate state variables
        self.define_all_states()
        ## Keep track of available car types
        self.available_car_types = self.get_all_available_car_types()
        self.available_existing_car_types = None
        self.get_all_available_existing_car_types()
        ## Keeping track of current timestamp
        self.reset_timestamp()
    
    ## Reset timestamp to 0 and regenerate trip arrivals
    def reset_timestamp(self):
        self.curr_ts = 0
        self.trip_arrivals = self.trip_demands.generate_arrivals()
    
    ## Construct all states at the beginning
    ##  Car states: |region| x |region| x |battery level| x 2 + |region| x |battery level| x 2
    ##  Trip states: |region| x |region| x (max connection patience + 1)
    ##  Plug states: |region| x |rates|
    ##  Charging Load states: 2 (peak & current)
    def define_all_states(self):
        curr_id = 0
        ## Define car states -- General type
        self.state_to_id["car"] = {}
        for dest in self.regions:
            for curr_region in self.regions:
                for battery in range(self.num_battery_levels):
                    for is_filled in [0, 1]:
                        car = Car(id = curr_id, dest, curr_region, battery, is_filled, type = "general")
                        self.state_to_id["car"][(dest, curr_region, battery, is_filled, "general")] = curr_id
                        self.state_dict[curr_id] = car
                        curr_id += 1
        
        ## Populate state counts for initial car deployment
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                cnt = self.region_battery_car_num[(region, battery)]
                id = self.state_to_id["car"][(region, region, battery, 0, "general")]
                self.state_counts[id] = cnt
        
        ## Define car states -- idling & charged
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                for type in ["idling", "charged"]:
                    car = Car(id = curr_id, region, region, battery, 0, type = type)
                    self.state_to_id["car"][(region, region, battery, 0, type)] = curr_id
                    self.state_dict[curr_id] = car
                    curr_id += 1
        
        ## Define trip states
        self.state_to_id["trip"] = {}
        for origin in self.regions:
            for dest in self.regions:
                for stag_time in range(self.connection_patience + 1):
                    trip = Trip(id = curr_id, origin, dest, stag_time)
                    self.state_to_id["trip"][(origin, dest, stag_time)] = curr_id
                    self.state_dict[curr_id] = trip
                    curr_id += 1
        
        ## Define plug states
        self.state_to_id["plug"] = {}
        for region in self.regions:
            for rate in range(self.num_charging_rates):
                plug = Plug(id = curr_id, region, rate)
                ## Set the number of available plugs
                plug_num = self.region_rate_plug_num[(region, rate)]
                plug.set(plug_num)
                self.state_counts[curr_id] = plug_num
                self.state_to_id["plug"][(region, rate)] = curr_id
                self.state_dict[curr_id] = plug
                curr_id += 1
        
        ## Define charging load states
        self.state_to_id["charging_load"] = {}
        for type in ["current", "peak"]:
            load = ChargingLoad(id = curr_id, type = type, 0)
            self.state_to_id["charging_load"][type] = curr_id
            self.state_dict[curr_id] = load
            curr_id += 1
    
    ## Atomic state transitions within a timestamp
    ##  The action, given by a solver, is performed on each individual car
    ## Options:
    ##      - Picking up a passenger request
    ##          Feasibility:
    ##              Time to arrival <= pickup_patience
    ##              remaining battery >= battery required for travel
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
    def transit_within_timestamp(self, action):
        if action.get_type() == "pickup":
            return self.transit_travel_within_timestamp(action, "pickup")
        elif action.get_type() == "rerouting":
            return self.transit_travel_within_timestamp(action, "rerouting")
        elif action.get_type() == "idling":
            return self.transit_travel_within_timestamp(action, "idling")
        return self.transit_charged_within_timestamp(action)
    
    ## Atomic state transitions for pickup action within timestamp
    def transit_travel_within_timestamp(self, action, type, car_id = None):
        origin, dest = action.get_origin(), action.get_dest()
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
            if type == "pickup":
                time_to_arrival = self.map.steps_to_location(car.get_curr_region(), car.get_dest())
                close_to_dest = time_to_arrival <= self.pickup_patience
            else:
                close_to_dest = True
            ## Compute the battery required for travel
            if type != "idling":
                battery_required = car.battery_per_step() * time_to_arrival
                enough_battery = battery_required <= car.get_battery() + self.battery_offset
            else:
                enough_battery = True
            ## Check if car can head to the pickup origin
            can_reach_origin = car.get_dest() == origin or not car.is_filled()
            ## Check if car is filled
            if type == "pickup":
                car_not_filled = True
            else:
                car_not_filled = not car.is_filled()
            ## Check all feasibility
            if close_to_dest and enough_battery and can_reach_origin and car_not_filled:
                self.state_counts[id] -= 1
                ## Compute target car type
                if type != "idling":
                    car_query_type = "general"
                else:
                    car_query_type = "idling"
                target_car_state = (dest, car.get_curr_region(), car.get_battery(), car.is_filled(), type = car_query_type)
                target_car_id = self.state_to_id["car"][target_car_state]
                self.state_counts[target_car_id] += 1
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
                target_car_state = (region, region, car.get_battery(), 0, type = "charged")
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
                return True
            car_type_idx += 1
        return False
    
    ## State transitions on multiple cars within the timestamp
    ##  More likely used by matching-based solvers
    def transit_group_within_timestamp(self, car_type_id_lst, action_lst):
        assert len(car_type_id_lst) == len(action_lst)
        ## Iterate through both lists
        for car_type_id, action in zip(car_type_id_lst, action_lst):
            if action.get_type() == "charged":
                atomic_action_success = self.transit_charged_within_timestamp(action, car_id = car_type_id)
            else:
                atomic_action_success = self.transit_travel_within_timestamp(action, type = action.get_type(), car_id = car_type_id)
            if not atomic_action_success:
                return False
        return True
    
    ## State transitions across timestamps
    ##  No actions given by solvers. Just refreshing state values
    ## Procedures:
    ##      - Update passenger requests: new requests + carry-over requests - abandoned requests
    ##      - Occupied cars make a movement by getting closer and consuming power
    ##      - Empty cars (empty, idling, charged) make a movement by getting closer and consuming power
    ##      - Zero out the idling and charged cars
    ##      - Set occupied cars to empty if they have arrived at destinations
    ##      - Reset all charging plug numbers to the given ones (i.e. all plugs are available)
    ##      - Zero out the current charging load
    def transit_across_timestamp(self):
        assert self.curr_ts < self.time_horizon
        ## Update passenger requests
        for origin in self.regions:
            for dest in self.regions:
                for stag_time in range(self.connection_patience + 1, 0, -1):
                    trip_id_curr = self.state_to_id["trip"][(origin, dest, stag_time)]
                    trip_id_prev = self.state_to_id["trip"][(origin, dest, stag_time - 1)]
                    self.state_counts[trip_id_curr] = self.state_counts[trip_id_prev]
                ## Load new passenger requests
                trip_id_new = self.state_to_id["trip"][(origin, dest, 0)]
                self.state_counts[trip_id_new] = self.trip_arrivals[(self.trip_arrivals["T"] == self.curr_ts) & (self.trip_arrivals["Origin"] == origin) & (self.trip_arrivals["Destination"] == dest)].iloc[0]["Count"]
        ## Make movements for occupied cars
        state_counts_new = torch.zeros(self.num_total_states)
        for dest in self.regions:
            for curr_region in self.regions:
                for battery in range(self.num_battery_levels):
                    for is_filled in [0, 1]:
                        car = Car(id = curr_id, dest, curr_region, battery, is_filled, type = "general")
                        car_id_curr = self.state_to_id["car"][(dest, curr_region, battery, is_filled, "general")]
                        car_curr = self.state_dict[car_id_curr]
                        next_region = self.map.next_cell_to_move(curr_region, dest)
                        car_id_next = self.state_to_id["car"][(dest, next_region, battery - car_curr.battery_per_step(), is_filled, "general")]
                        state_counts_new[car_id_next] += self.state_counts[car_id_curr]
        ## Drop-off passengers
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                car_id_filled = self.state_to_id["car"][(region, region, battery, 1, "general")]
                car_id_empty = self.state_to_id["car"][(region, region, battery, 1, "general")]
                state_counts_new[car_id_empty] += state_counts_new[car_id_filled]
                state_counts_new[car_id_filled] = 0
        ## Gather idling and charged cars
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                for type in ["idling", "charged"]:
                    car_id_curr = self.state_to_id["car"][(region, region, battery, 0, type)]
                    car_id_general = self.state_to_id["car"][(region, region, battery, 0, "general")]
                    state_counts_new[car_id_general] += self.state_counts[car_id_curr]
        ## Reset charging plugs
        for region in self.regions:
            for rate in range(self.num_charging_rates):
                plug_id = self.state_to_id["plug"][(region, rate)]
                state_counts_new[plug_id] = self.region_rate_plug_num[(region, rate)]
        ## Set peak charging load
        peak_load_id = self.state_to_id["charging_load"]["peak"]
        state_counts_new[peak_load_id] = self.state_counts[peak_load_id]
        ## Update state counts
        self.state_counts = state_counts_new
        ## Increment timestamp by 1
        self.curr_ts += 1
    
    ## Check if a given car is available
    def is_available_car(self, car):
        car_is_filled = car.is_filled()
        curr_region = car.get_curr_region()
        dest = car.get_dest()
        close_to_dest = self.map.steps_to_location(curr_region, dest) <= self.pickup_patience
        return not car_is_filled or (car_is_filled and close_to_dest)

    ## Get a list of the ids of all available car types
    def get_all_available_car_types(self):
        ret = []
        for car_type in self.state_to_id["car"]:
            id = self.state_to_id["car"][car_type]
            car = self.state_dict[id]
            if car_type[4] == "general" and self.is_available_car(car):
                ret.append(id)
        return ret
    
    ## Get a list of the ids of all available car types that are non-empty
    def get_all_available_existing_car_types(self):
        ret = []
        for id in self.available_car_types:
            if self.state_counts[id] > 0:
                ret.append(id)
        self.available_existing_car_types = ret

    ## Get a count of all available cars
    def get_available_car_counts(self):
        cnt = 0
        for id in self.available_car_types:
            cnt += int(self.state_counts[id])
        return cnt

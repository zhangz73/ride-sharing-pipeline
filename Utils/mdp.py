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
        assert type in ["general", "idling", "charged", "assigned"]
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
#        if not self.filled:
#            msg = f"Car: current region {self.curr_region}, destination {self.dest}, battery {self.battery}, filled {self.filled}, type {self.type}"
#        else:
        msg = f"Car: time to destination {self.time_to_dest}, destination {self.dest}, battery {self.battery}, filled {self.filled}, type {self.type}"
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
#        return f"{self.get_type().title()} from {self.get_origin()} to {self.get_dest()}"
        return f"Travel from {self.get_origin()} to {self.get_dest()}"

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
#        self.reward_df = self.reward_df[(self.reward_df["Type"] == "Charge") | ((self.reward_df["Type"] == "Travel") & (self.reward_df["Pickup"] == 1))]
        self.curr_ts = 0
        self.reward_cache = self.reward_df[self.reward_df["T"] == 0]
    
    ## Return payoff given action and timestamp
    ## Deprecated!!!
    def _query(self, action, timestamp):
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
    
    def get_travel_reward(self, origin, dest, ts):
        ## Update cache
        if self.curr_ts != ts:
            self.curr_ts = ts
            self.reward_cache = self.reward_df[self.reward_df["T"] == self.curr_ts]
        tmp_df = self.reward_cache[(self.reward_cache["Origin"] == origin) & (self.reward_cache["Destination"] == dest) & (self.reward_cache["Type"] == "Travel") & (self.reward_cache["Pickup"] == 1)]
        if tmp_df.shape[0] > 0:
            payoff = tmp_df.iloc[0]["Payoff"]
        else:
            payoff = 0
        ## For now, we only care about matching rate
        return payoff #1
    
    def get_charging_reward(self, region, rate, ts):
#        if ts >= 960 and ts < 1260:
#            pay_rate = 0.058
#        elif ts >= 540 and ts < 840:
#            pay_rate = 0.025
#        else:
#            pay_rate = 0.029
        ## Update cache
        if self.curr_ts != ts:
            self.curr_ts = ts
            self.reward_cache = self.reward_df[self.reward_df["T"] == self.curr_ts]
        tmp_df = self.reward_cache[(self.reward_cache["Region"] == region) & (self.reward_cache["Rate"] == rate) & (self.reward_cache["Type"] == "Charge")]
        if tmp_df.shape[0] > 0:
            payoff = tmp_df.iloc[0]["Payoff"]
        else:
            payoff = 0
        ## For now, charging rate assumes to be 0
#        pay_rate = 0
        return payoff #-0.5 #-pay_rate * rate

### This module implements the MDP process that does not allow interruptions of actions
class MarkovDecisionProcess:
    def __init__(self, map, trip_demands, reward_query, time_horizon, connection_patience, pickup_patience, num_battery_levels, battery_jump, charging_rates, battery_per_step = 1, battery_offset = 1, region_battery_car_fname = "region_battery_car.tsv", region_rate_plug_fname = "region_rate_plug.tsv", normalize_by_tripnums = False):
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
        self.battery_per_step = battery_per_step
        self.region_battery_car_num = None
        self.region_rate_plug_num = None
        self.battery_offset = battery_offset
        self.region_battery_car_fname = region_battery_car_fname
        self.region_rate_plug_fname = region_rate_plug_fname
        self.normalize_by_tripnums = normalize_by_tripnums
        self.load_initial_data()
        ## Auxiliary variables
        self.regions = self.map.get_regions()
        self.max_travel_time = self.map.get_max_travel_time()
        self.num_charging_rates = len(self.charging_rates)
        self.num_car_states = len(self.regions) * (2 * self.pickup_patience + self.max_travel_time + 2) * self.num_battery_levels + len(self.regions) * self.num_battery_levels * self.num_charging_rates
        self.num_trip_states = len(self.regions) * len(self.regions) * (self.connection_patience + 1)
        self.num_trip_reduced_states = len(self.regions) * 2 * (self.connection_patience + 1)
        self.num_plug_states = len(self.regions) * self.num_charging_rates
        self.num_total_states = self.num_car_states + self.num_trip_states + self.num_plug_states
        self.num_car_reduced_states = len(self.regions) * (self.pickup_patience + 1) * self.num_battery_levels * 2 + len(self.regions) * self.num_battery_levels * self.num_charging_rates
        self.num_total_reduced_states = self.num_car_states + self.num_trip_reduced_states + self.num_plug_states
        ## TODO: Fix it!
        self.num_total_local_states = len(self.regions) * (self.connection_patience + 1) + 3
        ## Variables keeping track of states
        self.state_dict = {}
        self.state_to_id = {}
        self.state_counts = torch.zeros(self.num_total_states)
        self.reduced_state_to_id = {}
        self.reduced_state_counts = torch.zeros(self.num_total_reduced_states)
        self.local_order_map = {}
        ## Variables keeping track of current timestamp
        self.curr_ts = 0
        self.reset_timestamp()
        ## Populate state variables
        self.define_all_states()
        self.define_all_reduced_states()
        ## Variables keeping track of available car types
        self.available_car_types, self.state_is_available_car = self.get_all_available_car_types()
        self.available_existing_car_types = self.get_all_available_existing_car_types()
        ## Variables keeping track of all actions
        self.all_actions = {}
        self.action_to_id = {}
        self.action_desc_to_id = {}
        self.construct_all_actions()
        self.all_reduced_actions = {}
        self.reduced_action_to_id = {}
        self.reduced_action_desc_to_id = {}
        self.construct_all_reduced_actions()
        self.action_lst = [self.all_actions[k] for k in self.all_actions.keys()]
        self.reduced_action_lst = [self.all_reduced_actions[k] for k in self.all_reduced_actions.keys()]
        ## Store a copy of initial states for resetting
        self.state_counts_init = self.state_counts.clone()
        self.reduced_state_counts_init = self.reduced_state_counts.clone()
        ## Variables keeping track of available car types
        self.available_car_types, self.state_is_available_car = self.get_all_available_car_types()
        self.available_existing_car_types = self.get_all_available_existing_car_types()
        ## Map state transitions in a specific format so that it can be vectorized
        self.transit_across_timestamp_prepare()
    
    ## Get the length of states
    def get_state_len(self, state_reduction = False, model = "policy"):
        if not state_reduction:
            return self.num_total_states
        if model == "policy":
            return self.num_total_reduced_states + self.num_total_local_states
        return self.num_total_reduced_states
    
    ## Reset all states to the initial one
    def reset_states(self):
        self.state_counts = self.state_counts_init.clone()
        self.reduced_state_counts = self.reduced_state_counts_init.clone()
        self.available_existing_car_types = self.get_all_available_existing_car_types()
        self.reset_timestamp()
        self.payoff_curr_ts = torch.tensor(0.)
    
    ## Set the states and payoff_schedule_dct according to the given ones
    def set_states(self, state_counts, ts, payoff_curr_ts = None):
        self.state_counts = state_counts.clone()
#        self.available_existing_car_types = self.get_all_available_existing_car_types()
        if payoff_curr_ts is None:
            payoff_curr_ts = torch.tensor(0.)
        self.payoff_curr_ts = payoff_curr_ts
        self.curr_ts = ts
    
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
    
    ## Get the state counts (cloned version)
    ## TODO: Implement it!
    def get_state_counts(self, state_reduction = False, car_id = None):
        if not state_reduction:
            return self.state_counts.clone()
        assert car_id is not None
        reduced_state_counts = self.reduced_state_counts.clone()
        car = self.state_dict[car_id]
        local_state_counts = self.get_local_state(car)
        return torch.cat([reduced_state_counts, local_state_counts])
    
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
                action = Travel(id = curr_id, origin = origin, dest = dest, is_filled = None)
                self.all_actions[curr_id] = action
                self.action_to_id[action] = curr_id
                self.action_desc_to_id[("travel", origin, dest)] = curr_id
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
    
    ## Construct the list of all reduced actions: |R| + |charging rates|
    ##  Travel Action: Go to dest d
    ##  Charge Action: Plug in with rate \delta
    ##  Do-Nothing Action: Keep traveling or idling
    def construct_all_reduced_actions(self):
        curr_id = 0
        ## Construct all travel actions
        for dest in self.regions:
            action = Travel(id = curr_id, origin = None, dest = dest, is_filled = None)
            self.all_reduced_actions[curr_id] = action
            self.reduced_action_to_id[action] = curr_id
            self.reduced_action_desc_to_id[("travel", dest)] = curr_id
            curr_id += 1
        ## Construct all charge actions
        for rate in self.charging_rates:
            action = Charge(id = curr_id, region = None, rate = rate)
            self.all_reduced_actions[curr_id] = action
            self.reduced_action_to_id[action] = curr_id
            self.reduced_action_desc_to_id[("charge", rate)] = curr_id
            curr_id += 1
        ## Construct the nothing action
        action = Nothing(id = curr_id)
        self.all_reduced_actions[curr_id] = action
        self.reduced_action_to_id[action] = curr_id
        self.reduced_action_desc_to_id[("nothing")] = curr_id
    
    ## Return the list of all possible actions
    def get_all_actions(self, state_reduction = False):
        if not state_reduction:
            return self.all_actions
        return self.all_reduced_actions
    
    ## Return the payoff at the current timestamp
    def get_payoff_curr_ts(self):
        ret = self.payoff_curr_ts.clone()
        if self.normalize_by_tripnums:
            ret = ret / self.total_arrivals
        return ret
    
    ## Zero out the payoff at the current timestamp
    def reset_payoff_curr_ts(self):
        self.payoff_curr_ts = torch.tensor(0.)
    
    ## Reset timestamp to 0 and regenerate trip arrivals
    def reset_timestamp(self):
        self.curr_ts = 0
        self.trip_arrivals = self.trip_demands.generate_arrivals()
        self.ts_cache = 0
        self.origin_cache = 0
        self.trip_arrivals_cache = self.trip_arrivals[(self.trip_arrivals["Origin"] == self.origin_cache) & (self.trip_arrivals["T"] == self.ts_cache)]
        self.payoff_curr_ts = torch.tensor(0.)
        self.total_arrivals = self.trip_arrivals["Count"].sum()
    
    ## Query the trip arrivals
    def query_trip_arrival(self, origin, dest, t):
        ## Update cache if not applicable
        if self.ts_cache != t:
            self.ts_cache = t
            self.trip_arrivals_cache = self.trip_arrivals[self.trip_arrivals["T"] == self.ts_cache]
        ## Query
        tmp_df = self.trip_arrivals_cache[(self.trip_arrivals_cache["Destination"] == dest) & (self.trip_arrivals_cache["Origin"] == origin)]
        if tmp_df.shape[0] > 0:
            cnt = tmp_df.iloc[0]["Count"]
        else:
            cnt = 0
        return cnt
    
    ## Construct all states at the beginning
    def define_all_states(self):
        curr_id = 0
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
        ## Define car states -- General type
        self.state_to_id["car"] = {}
        for dest in self.regions:
            for battery in range(self.num_battery_levels):
                for eta in range(self.pickup_patience + self.max_travel_time + 1):
                    car = Car(curr_id, dest, None, battery, None, type = "general", time_to_dest = eta)
                    self.state_to_id["car"][("general", dest, eta, battery)] = curr_id
                    self.state_dict[curr_id] = car
                    curr_id += 1
        ## Define car states -- Assigned type
        for dest in self.regions:
            for battery in range(self.num_battery_levels):
                for eta in range(self.pickup_patience + 1):
                    car = Car(curr_id, dest, None, battery, None, type = "assigned", time_to_dest = eta)
                    self.state_to_id["car"][("assigned", dest, eta, battery)] = curr_id
                    self.state_dict[curr_id] = car
                    curr_id += 1
        ## Define car states -- Charged type
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                for rate in self.charging_rates:
                    car = Car(curr_id, region, region, battery, None, type = "charged", charged_rate = rate)
                    self.state_to_id["car"][("charged", region, battery, rate)] = curr_id
                    self.state_dict[curr_id] = car
                    curr_id += 1
        ## Populate state counts for initial car deployment
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                if (region, battery) in self.region_battery_car_num:
                    cnt = self.region_battery_car_num[(region, battery)]
                else:
                    cnt = 0
                id = self.state_to_id["car"][("general", region, 0, battery)]
                self.state_counts[id] = cnt
        ## Define trip states
        self.state_to_id["trip"] = {}
        for origin in self.regions:
            curr_id_begin = curr_id
            for dest in self.regions:
                for stag_time in range(self.connection_patience + 1):
                    trip = Trip(curr_id, origin, dest, stag_time)
                    self.state_to_id["trip"][(origin, dest, stag_time)] = curr_id
                    self.state_dict[curr_id] = trip
                    curr_id += 1
                ## Load new passenger requests
                trip_id_new = self.state_to_id["trip"][(origin, dest, 0)]
                self.state_counts[trip_id_new] = self.query_trip_arrival(origin, dest, 0)
            self.local_order_map[origin] = (curr_id_begin, curr_id)
    
    ## Construct all reduced states at the beginning
    def define_all_reduced_states(self):
        self.reduced_state_to_id["plug"] = self.state_to_id["plug"]
        curr_id = len(self.reduced_state_to_id["plug"])
        ## Define car states -- General type
        self.reduced_state_to_id["car"] = {}
        for dest in self.regions:
            for battery in range(self.num_battery_levels):
                for eta in range(self.pickup_patience + 1):
                    self.reduced_state_to_id["car"][("general", dest, eta, battery)] = curr_id
                    curr_id += 1
        ## Define car states -- Assigned type
        for dest in self.regions:
            for battery in range(self.num_battery_levels):
                for eta in range(self.pickup_patience + 1):
                    self.reduced_state_to_id["car"][("assigned", dest, eta, battery)] = curr_id
                    curr_id += 1
        ## Define car states -- Charged type
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                for rate in self.charging_rates:
                    self.reduced_state_to_id["car"][("charged", region, battery, rate)] = curr_id
                    curr_id += 1
        ## Populate state counts for initial car deployment
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                if (region, battery) in self.region_battery_car_num:
                    cnt = self.region_battery_car_num[(region, battery)]
                else:
                    cnt = 0
                id = self.reduced_state_to_id["car"][("general", region, 0, battery)]
                self.reduced_state_counts[id] = cnt
        self.reduced_state_to_id["trip"] = {}
        for region in self.regions:
            for stag_time in range(self.connection_patience + 1):
                self.reduced_state_to_id["trip"][("origin", region, stag_time)] = curr_id
                curr_id += 1
                self.reduced_state_to_id["trip"][("dest", region, stag_time)] = curr_id
                curr_id += 1
        ## Load new passenger requests
        for origin in self.regions:
            for dest in self.regions:
                trip_id_origin = self.reduced_state_to_id["trip"][("origin", origin, 0)]
                trip_id_dest = self.reduced_state_to_id["trip"][("dest", dest, 0)]
                trip_id_new = self.state_to_id["trip"][(origin, dest, 0)]
                cnt = self.state_counts[trip_id_new]
                self.reduced_state_counts[trip_id_origin] += cnt
                self.reduced_state_counts[trip_id_dest] += cnt
    
    ## Atomic state transitions within a timestamp
    ## Return if the action has been successfully processed. False if the action is not feasible
    ## If and only if action_id is NOT None, the reduced scheme will be adopted
    def transit_within_timestamp(self, action, reduced = False, car_id = None):
        if reduced:
            assert car_id is not None
#            action = self.all_reduced_actions[action_id]
        else:
#            action = self.all_actions[action_id]
            car_id = None
        if action.get_type() == "charged":
            return self.transit_charged_within_timestamp(action, car_id = car_id)
        elif action.get_type() == "nothing":
            return self.transit_nothing_within_timestamp(car_id = car_id)
        return self.transit_travel_within_timestamp(action, car_id = car_id)
    
    ## Atomic state transitions for nothing action within timestamp
    def transit_nothing_within_timestamp(self, car_id = None):
        if car_id is None:
            id = self.select_feasible_car(None, None, "nothing")
        else:
            id = car_id
        car = self.state_dict[id]
        origin = car.get_dest()
        eta = car.get_time_to_dest()
        battery = car.get_battery()
        target_car_state = ("assigned", origin, eta, battery)
        target_car_id = self.state_to_id["car"][target_car_state]
        ## Update states
        self.state_counts[id] -= 1
        self.reduced_state_counts[id] -= 1
        self.state_counts[target_car_id] += 1
        self.reduced_state_counts[target_car_id] += 1
        return True
    
    ## Atomic state transitions for travel action within timestamp
    def transit_travel_within_timestamp(self, action, car_id = None):
        origin, dest = action.get_origin(), action.get_dest()
        car_type_idx = 0
        if origin is None:
            assert car_id is not None
        if car_id is None:
            id = self.select_feasible_car(origin, dest, "travel")
        else:
            id = car_id
        if id is None:
            return False
        car = self.state_dict[id]
        origin = car.get_dest()
        eta = car.get_time_to_dest()
        battery = car.get_battery()
        trip_time = self.map.time_to_location(origin, dest, self.curr_ts)
        trip_distance = self.map.distance(origin, dest)
        if battery < self.battery_per_step * trip_distance:
            return False
        target_car_state = ("general", dest, eta + trip_time, battery - self.battery_per_step * trip_distance)
        target_car_id = self.state_to_id["car"][target_car_state]
        ## Update states
        self.state_counts[id] -= 1
        self.reduced_state_counts[id] -= 1
        self.state_counts[target_car_id] += 1
        self.reduced_state_counts[target_car_id] += 1
        ## Update active trip requests if pickup is performed
        stag_time = self.connection_patience
        action_fulfilled = False
        atomic_payoff = 0
        while stag_time >= 0 and not action_fulfilled:
            trip_id = self.state_to_id["trip"][(origin, dest, stag_time)]
            if self.state_counts[trip_id] > 0:
                self.state_counts[trip_id] -= 1
                action_fulfilled = True
                trip_id_origin = self.reduced_state_to_id["trip"][("origin", origin, stag_time)]
                trip_id_dest = self.reduced_state_to_id["trip"][("dest", dest, stag_time)]
                self.reduced_state_counts[trip_id_origin] -= 1
                self.reduced_state_counts[trip_id_dest] -= 1
            stag_time -= 1
        if action_fulfilled:
            atomic_payoff = self.reward_query.get_travel_reward(origin, dest, self.curr_ts)
        self.payoff_curr_ts += atomic_payoff
        return True
    
    ## Atomic state transitions for charged action within timestamp
    def transit_charged_within_timestamp(self, action, car_id = None):
        region, rate = action.get_region(), action.get_rate()
        if region is None:
            assert car_id is not None
        if car_id is None:
            id = self.select_feasible_car(region, region, "charge")
        else:
            id = car_id
        if id is None:
            return False
        car = self.state_dict[id]
        region = car.get_dest()
        plug_id = self.state_to_id["plug"][(region, rate)]
        if self.state_counts[plug_id] == 0:
            return False
        battery = car.get_battery()
        target_car_state = ("charged", region, battery, rate)
        target_car_id = self.state_to_id["car"][target_car_state]
        ## Update states
        self.state_counts[id] -= 1
        self.reduced_state_counts[id] -= 1
        self.state_counts[target_car_id] += 1
        self.reduced_state_counts[target_car_id] += 1
        atomic_payoff = self.reward_query.get_charging_reward(region, rate, self.curr_ts)
        self.payoff_curr_ts += atomic_payoff
        return True
    
    def select_feasible_car(self, origin, dest, type):
        car_ret = None
        if type == "travel":
            ## Car within max pickup patience time to origin
            ## Car remaining battery can make the trip from origin to dest
            ## Find the closest car then with the largest battery
            trip_distance = self.map.distance(origin, dest)
            for eta in range(self.pickup_patience + 1):
                for battery in range(self.num_battery_levels - 1, self.battery_per_step * trip_distance - 1, -1):
                    if car_ret is None:
                        target_car_state = ("general", origin, eta, battery)
                        target_car_id = self.state_to_id["car"][target_car_state]
                        if self.state_counts[target_car_id] > 0:
                            car_ret = target_car_id
                            break
                if car_ret is not None:
                    break
        elif type == "nothing":
            car_ret = self.get_all_available_existing_car_types()
            if len(car_ret) > 0:
                car_ret = car_ret[0]
            else:
                car_ret = None
        else:
            ## Car already at origin
            ## Find the car with lowest battery
            for battery in range(self.num_battery_levels):
                if car_ret is None:
                    target_car_state = ("general", origin, 0, battery)
                    target_car_id = self.state_to_id["car"][target_car_state]
                    if self.state_counts[target_car_id] > 0:
                        car_ret = target_car_id
                        break
        return car_ret
        
    ## Vectorize it!!!
    ##  1. Pad 0 to the end for null mappings
    ##  2. Create a 2D tensor y corresponding to indices per slot
    ##  3. t = t.gather(1, y)
    ##  4. torch.sum(t, dim = 1)
    def transit_across_timestamp_prepare(self):
        self.state_counts_map = [[] for _ in range(self.num_total_states + 1)]
        self.reduced_state_counts_map = [[] for _ in range(self.num_total_reduced_states + 1)]
        self.plug_tracking_map = torch.zeros(self.num_total_states)
        self.plug_tracking_reduced_map = torch.zeros(self.num_total_reduced_states)
        self.trip_arrivals_map = torch.zeros((self.time_horizon, self.num_total_states))
        self.trip_arrivals_reduced_map = torch.zeros((self.time_horizon, self.num_total_reduced_states))
        ## Update passenger requests
        for origin in self.regions:
            for dest in self.regions:
                for stag_time in range(self.connection_patience, 0, -1):
                    trip_id_curr = self.state_to_id["trip"][(origin, dest, stag_time)]
                    trip_id_prev = self.state_to_id["trip"][(origin, dest, stag_time - 1)]
                    self.state_counts_map[trip_id_curr].append(trip_id_prev)
                    ## Update origins
                    trip_id_curr_reduced = self.reduced_state_to_id["trip"][("origin", origin, stag_time)]
                    trip_id_prev_reduced = self.reduced_state_to_id["trip"][("origin", origin, stag_time - 1)]
                    self.reduced_state_counts_map[trip_id_curr_reduced].append(trip_id_prev_reduced)
                    ## Update dests
                    trip_id_curr_reduced = self.reduced_state_to_id["trip"][("dest", dest, stag_time)]
                    trip_id_prev_reduced = self.reduced_state_to_id["trip"][("dest", dest, stag_time - 1)]
                    self.reduced_state_counts_map[trip_id_curr_reduced].append(trip_id_prev_reduced)
        ## Load new passenger requests
        for t in range(self.time_horizon - 1):
            for origin in self.regions:
                for dest in self.regions:
                    trip_id_new = self.state_to_id["trip"][(origin, dest, 0)]
                    trip_id_origin = self.reduced_state_to_id["trip"][("origin", origin, 0)]
                    trip_id_dest = self.reduced_state_to_id["trip"][("dest", dest, 0)]
                    cnt = self.query_trip_arrival(origin, dest, t + 1)
                    self.trip_arrivals_map[t, trip_id_new] = cnt
                    self.trip_arrivals_reduced_map[t, trip_id_origin] += cnt
                    self.trip_arrivals_reduced_map[t, trip_id_dest] += cnt
        ## Make movements for traveling cars
        for dest in self.regions:
            for eta in range(1, self.pickup_patience + self.max_travel_time + 1):
                for battery in range(self.num_battery_levels):
                    car_id_curr = self.state_to_id["car"][("general", dest, eta, battery)]
                    if eta <= self.pickup_patience:
                        car_id_assigned = self.state_to_id["car"][("assigned", dest, eta, battery)]
                    car_curr = self.state_dict[car_id_curr]
                    next_state = ("general", dest, eta - 1, battery)
#                    if next_state in self.state_to_id["car"]:
                    car_id_next = self.state_to_id["car"][next_state]
                    if eta <= self.pickup_patience:
                        self.state_counts_map[car_id_next] += [car_id_curr, car_id_assigned]
                        self.reduced_state_counts_map[car_id_next] += [car_id_curr, car_id_assigned]
                    else:
                        self.state_counts_map[car_id_next] += [car_id_curr]
                        self.reduced_state_counts_map[car_id_next] += [car_id_curr]
        ## Gather idling cars
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                car_id_curr = self.state_to_id["car"][("assigned", region, 0, battery)]
                car_id_general = self.state_to_id["car"][("general", region, 0, battery)]
                self.state_counts_map[car_id_general] += [car_id_curr, car_id_general]
                self.reduced_state_counts_map[car_id_general] += [car_id_curr, car_id_general]
        ## Gather charged cars
        for region in self.regions:
            for battery in range(self.num_battery_levels):
                for rate in self.charging_rates:
                    car_id_curr = self.state_to_id["car"][("charged", region, battery, rate)]
                    next_battery = battery
                    ## charged cars gain power
                    car_curr = self.state_dict[car_id_curr]
                    next_battery += rate
                    next_battery = min(next_battery, self.num_battery_levels - 1)
                    ## Update payoff
                    curr_action_id = self.action_desc_to_id[("charge", region, rate)]
                    car_id_general = self.state_to_id["car"][("general", region, 0, next_battery)]
                    self.state_counts_map[car_id_general] += [car_id_curr]
                    self.reduced_state_counts_map[car_id_general] += [car_id_curr]
        ## Reset charging plugs
        for region in self.regions:
            for rate in self.charging_rates:
                plug_id = self.state_to_id["plug"][(region, rate)]
                if (region, rate) in self.region_rate_plug_num:
                    plug_num = self.region_rate_plug_num[(region, rate)]
                else:
                    plug_num = 0
                self.plug_tracking_map[plug_id] = plug_num
                self.plug_tracking_reduced_map[plug_id] = plug_num
        ## Clean state counts map
        max_len = 0
        for i in range(len(self.state_counts_map)):
            self.state_counts_map[i] = list(set(self.state_counts_map[i]))
            max_len = max(max_len, len(self.state_counts_map[i]))
        for i in range(len(self.state_counts_map)):
            curr_len = len(self.state_counts_map[i])
            for _ in range(max_len - curr_len):
                self.state_counts_map[i].append(self.num_total_states)
        self.state_counts_map = torch.tensor(self.state_counts_map)
        ## Clean reduced state counts map
        max_len = 0
        for i in range(len(self.reduced_state_counts_map)):
            self.reduced_state_counts_map[i] = list(set(self.reduced_state_counts_map[i]))
            max_len = max(max_len, len(self.reduced_state_counts_map[i]))
        for i in range(len(self.reduced_state_counts_map)):
            curr_len = len(self.reduced_state_counts_map[i])
            for _ in range(max_len - curr_len):
                self.reduced_state_counts_map[i].append(self.num_total_reduced_states)
        self.reduced_state_counts_map = torch.tensor(self.reduced_state_counts_map)
    
    def transit_across_timestamp(self):
        assert self.curr_ts < self.time_horizon
        state_counts_new = torch.zeros(self.num_total_states)
        state_counts_aug = torch.cat((self.state_counts, torch.tensor([0])))
        state_counts_new = self.plug_tracking_map + self.trip_arrivals_map[self.curr_ts,:] + torch.sum(state_counts_aug[self.state_counts_map], dim = 1).reshape((-1,))[:-1]
        ## Update state counts
        self.state_counts = state_counts_new.clone()

        reduced_state_counts_new = torch.zeros(self.num_total_reduced_states)
#        reduced_state_counts_aug = torch.cat((self.reduced_state_counts, torch.tensor([0])))
        reduced_state_counts_new = self.plug_tracking_reduced_map + self.trip_arrivals_reduced_map[self.curr_ts,:] + torch.sum(state_counts_aug[self.reduced_state_counts_map], dim = 1).reshape((-1,))[:-1]
        ## Update reduced state counts
        self.reduced_state_counts = reduced_state_counts_new.clone()
        
        ## Recompute existing available cars
#        self.available_existing_car_types = self.get_all_available_existing_car_types()
        ## Increment timestamp by 1
        self.curr_ts += 1
    
    ## Get the local state information given a car
    def get_local_state(self, car):
        dest, eta, battery = car.get_dest(), car.get_time_to_dest(), car.get_battery()
        local_state_counts = torch.zeros(self.num_total_local_states)
        local_state_counts[0] = dest
        local_state_counts[1] = eta
        local_state_counts[2] = battery
        begin, end = self.local_order_map[dest]
        local_state_counts[3:] = self.state_counts[begin:end]
        return local_state_counts
    
    ## Check if the action has the potential to be feasible
    ##  If False, then the action is certainly infeasible
    ##  If True, then the action might still be infeasible
    def action_is_potentially_feasible(self, action_id, reduced, car_id = None):
        if reduced:
            if car_id is None:
                return True
            action = self.all_actions[action_id]
            if action.get_type() == "nothing":
                return True
            car = self.state_dict[car_id]
            dest, eta, battery = car.get_dest(), car.get_time_to_dest(), car.get_battery()
            if action.get_type() == "charged":
                rate = action.get_rate()
                plug_id = self.state_to_id["plug"][(dest, rate)]
                return eta == 0 and self.state_counts[plug_id] > 0
            new_dest = action.get_dest()
            trip_distance = self.map.distance(dest, new_dest)
            min_battery_needed = self.battery_per_step * trip_distance
            return eta <= self.pickup_patience and battery >= min_battery_needed
        ## If not in the state reduction scheme
        action = self.all_actions[action_id]
        if action.get_type() == "nothing":
            ## Always feasible
            return True
        elif action.get_type() == "charged":
            ## At least 1 car in the region (i.e. eta = 0)
            ## At least 1 plug with the rate available
            region, rate = action.get_region(), action.get_rate()
            plug_id = self.state_to_id["plug"][(region, rate)]
            if self.state_counts[plug_id] == 0:
                return False
            for battery in range(self.num_battery_levels):
                target_car_state = ("general", region, 0, battery)
                target_car_id = self.state_to_id["car"][target_car_state]
                if self.state_counts[target_car_id] > 0:
                    return True
        else:
            ## At least 1 car within eta of pickup patience
            ## The car has battery level larger than trip distance
            origin, dest = action.get_origin(), action.get_dest()
            trip_distance = self.map.distance(origin, dest)
            min_battery_needed = self.battery_per_step * trip_distance
            for eta in range(self.pickup_patience + 1):
                for battery in range(min_battery_needed, self.num_battery_levels):
                    target_car_state = ("general", origin, eta, battery)
                    target_car_id = self.state_to_id["car"][target_car_state]
                    if self.state_counts[target_car_id] > 0:
                        return True
        return False
    
    def state_counts_to_potential_feasible_actions(self, reduced, state_counts = None):
        if state_counts is None:
            state_counts = self.state_counts
        if reduced:
            total_actions = len(self.all_reduced_actions)
        else:
            total_actions = len(self.all_actions)
        mask = torch.zeros(total_actions)
        has_feasible_action = False
        for action_id in range(total_actions):
            if self.action_is_potentially_feasible(action_id, reduced):
                mask[action_id] = 1
                has_feasible_action = True
        return mask
    
    ## Return a list of feasible actions
    def all_feasible_actions(self, reduced):
        feasible_actions = []
        if reduced:
            total_actions = len(self.all_reduced_actions)
        else:
            total_actions = len(self.all_actions)
        for action_id in range(total_actions):
            if self.action_is_potentially_feasible(action_id, reduced):
                feasible_actions.append(action_id)
        return feasible_actions
    
    def get_available_car_counts(self):
        cnt = torch.sum(self.state_counts[self.available_car_types])
        return int(cnt.data)
    
    def get_available_car_ids(self, state_reduction):
        cnt = self.get_available_car_counts()
        if not state_reduction:
            return [None] * cnt
        return self.get_all_available_existing_car_ids()

    def get_all_available_existing_car_types(self):
        ret = (self.state_counts * self.state_is_available_car > 0).nonzero(as_tuple = True)[0]
        return list(ret.numpy())
    
    def get_all_available_existing_car_ids(self):
        available_car_types = self.get_all_available_existing_car_types()
        counts = self.state_counts[available_car_types]
        return np.repeat(available_car_types, counts)
    
    ## Get a list of the ids of all available car types
    ## Cars within pickup_patience to their dests
    def get_all_available_car_types(self):
        ret = []
        state_is_available_car = torch.zeros(len(self.state_counts))
        for car_type in self.state_to_id["car"]:
            id = self.state_to_id["car"][car_type]
            car = self.state_dict[id]
            if car_type[0] == "general" and car_type[2] <= self.pickup_patience:
                ret.append(id)
                state_is_available_car[id] = 1
        return ret, state_is_available_car

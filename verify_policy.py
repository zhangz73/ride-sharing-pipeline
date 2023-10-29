import numpy as np
import pandas as pd

## (dest, eta, battery): cnt
CAR_INIT = {(0, 0, 0):200, (1, 0, 0):200, (2, 0, 0):200, (3, 0, 0):200, (4, 0, 0):200}
TIME_HORIZON = 360
NUM_REGIONS = 5
BATTERY_PER_DISTANCE = 0
PACK_SIZE = 1
## t car_dest car_eta car_battery action_type action_info
df_policy = pd.read_csv("PolicyLogs/policy_log.csv")
## T Origin Destination Distance TripTime
df_triptime = pd.read_csv("Data/Map/map_1000car5region.tsv", sep = "\t")
## 0 ... 24
trip_requests_matrix = pd.read_csv("PolicyLogs/trip_requests_realized.csv").to_numpy()

total_trips = np.sum(trip_requests_matrix)

def verify_car_balancedness():
    car_dict = CAR_INIT
    fulfilled_trip_num = 0
    for t in range(TIME_HORIZON):
        curr_car_dict = {}
        df_policy_tmp = df_policy[df_policy["t"] == t]
        for i in range(df_policy_tmp.shape[0]):
            car_dest = df_policy_tmp.iloc[i]["car_dest"]
            car_eta = df_policy_tmp.iloc[i]["car_eta"]
            car_battery = df_policy_tmp.iloc[i]["car_battery"]
            action_type = df_policy_tmp.iloc[i]["action_type"]
            action_info = df_policy_tmp.iloc[i]["action_info"]
            car_key = (car_dest, car_eta, car_battery)
            if car_key not in car_dict or car_dict[car_key] <= 0:
                print(t, car_key)
                print(car_dict)
            assert car_key in car_dict and car_dict[car_key] > 0
            if action_type in ["pickup", "rerouting"]:
                origin, dest = car_dest, int(action_info)
                trip_idx = origin * NUM_REGIONS + dest
                trip_time = df_triptime[(df_triptime["T"] == t) & (df_triptime["Origin"] == origin) & (df_triptime["Destination"] == dest)].iloc[0]["TripTime"]
                distance = df_triptime[(df_triptime["T"] == t) & (df_triptime["Origin"] == origin) & (df_triptime["Destination"] == dest)].iloc[0]["Distance"]
                if trip_requests_matrix[t, trip_idx] > 0:
                    fulfilled_trip_num += 1
                    trip_requests_matrix[t, trip_idx] -= 1
                elif trip_requests_matrix[t, trip_idx] == 0 and origin == dest:
                    assert car_eta == 0
                    trip_time = 1
                    distance = 0
                next_dest = dest
                next_eta = car_eta + trip_time - 1
                next_battery = car_battery - BATTERY_PER_DISTANCE * distance
                assert next_battery >= 0
            elif action_type == "charged":
                assert car_eta == 0
                region = car_dest
                rate = int(action_info)
                next_dest = region
                next_eta = car_eta + 1 - 1
                next_battery = min(car_battery + rate, PACK_SIZE - 1)
            else:
                next_dest = car_dest
                next_eta = max(car_eta - 1, 0)
                next_battery = car_battery
            next_car_key = (next_dest, next_eta, next_battery)
            if next_car_key not in curr_car_dict:
                curr_car_dict[next_car_key] = 0
            curr_car_dict[next_car_key] += 1
            car_dict[car_key] -= 1
        for car_key in car_dict:
            if car_dict[car_key] > 0:
                next_car_key = (car_key[0], max(car_key[1] - 1, 0), car_key[2])
                if next_car_key not in curr_car_dict:
                    curr_car_dict[next_car_key] = 0
                curr_car_dict[next_car_key] += car_dict[car_key]
        car_dict = curr_car_dict
    return fulfilled_trip_num

fulfilled_trip_num = verify_car_balancedness()
print(fulfilled_trip_num / total_trips)

import numpy as np
import os
import csv

# file_base_path = "runs/lift/panda/osc_pose/online/v8_reduced_ep_len_500/run_1"
file_base_path = "demonstrations/"
npy_file_name = "robosuite_door_mirror_demonstration_v2.npy"
csv_file_name = "robosuite_data_test.csv"

npy_data = np.load(os.path.join(file_base_path,npy_file_name), allow_pickle=True)
#data_mean = npy_data.mean(axis=1)

for i in range(len(npy_data)):
    if npy_data[i]["rewards"][-1] != 1:
        print(f"index: {i}, reward: {npy_data[i]['rewards'][-1]}")
        break

expert_demos = npy_data[:i]

print(f"length of npy_data: {len(npy_data)}")
print(f"length of expert_demos: {len(expert_demos)}")

np.save("robosuite_door_mirror_demonstration_v2_expert.npy", expert_demos)
# print(f"lengt of actions: {npy_data[0]['actions'].shape}")
# for action in npy_data[0]['actions']:
#     print(f"action: {action}")

# with open(os.path.join(file_base_path,csv_file_name), mode='w') as csv_file:
#     csvwriter = csv.writer(csv_file)
#     csvwriter.writerows(npy_data)


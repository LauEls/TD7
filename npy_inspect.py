import numpy as np
import os
import csv

# file_base_path = "runs/lift/panda/osc_pose/online/v8_reduced_ep_len_500/run_1"
file_base_path = "demonstrations/"
npy_file_name = "robosuite_data_test.npy"
csv_file_name = "robosuite_data_test.csv"

npy_data = np.load(os.path.join(file_base_path,npy_file_name), allow_pickle=True)
#data_mean = npy_data.mean(axis=1)

print(f"length of npy_data: {len(npy_data)}")
print(f"lengt of actions: {npy_data[0]['actions'].shape}")
for action in npy_data[0]['actions']:
    print(f"action: {action}")

# with open(os.path.join(file_base_path,csv_file_name), mode='w') as csv_file:
#     csvwriter = csv.writer(csv_file)
#     csvwriter.writerows(npy_data)


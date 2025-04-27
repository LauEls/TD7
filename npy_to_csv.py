import numpy as np
import os
import csv

for i in range(5):

    # file_base_path = "runs/door_mirror/gh360/osc_pose/v3_ep_length_50/run_"+str(i)
    file_base_path = "runs/door/real_gh360/eef_vel/online/v12_corl_with_demos_3/run_"+str(i)
    # file_base_path = "runs/lift/panda/osc_pose/offline/v5_medium_expert_2/run_0"
    npy_file_name = "results.npy"
    csv_file_name = "results.csv"

    npy_data = np.load(os.path.join(file_base_path,npy_file_name), allow_pickle=True)
    #data_mean = npy_data.mean(axis=1)

    with open(os.path.join(file_base_path,csv_file_name), mode='w') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerows(npy_data)


import numpy as np
import os
import csv

file_base_path = "runs/lift/panda/osc_pose/online/v2_alg_comp/"
npy_file_name = "TD7_Lift_7.npy"
csv_file_name = "TD7_Lift_7.csv"

npy_data = np.load(os.path.join(file_base_path,npy_file_name), allow_pickle=True)
#data_mean = npy_data.mean(axis=1)

with open(os.path.join(file_base_path,csv_file_name), mode='w') as csv_file:
    csvwriter = csv.writer(csv_file)
    csvwriter.writerows(npy_data)


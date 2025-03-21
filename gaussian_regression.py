import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
#from mujoco_py import MjSim
import sys
import robosuite as suite
from robosuite.utils.input_utils import *

from robosuite.wrappers import GymWrapper
sys.path.insert(0, '/home/laurenz/phd_project/sac/sac_2')
from wrappers import NormalizedBoxEnv

import time


options = {}

# recorded_actions = np.loadtxt("/home/laurenz/phd_project/sac/scripts/test_data/v6/delta_action.csv", delimiter=",", dtype=float)
file_name = "/home/laurenz/phd_project/TD7/demonstrations/robosuite_data_test.npy"
# npy_file_name = "robosuite_data_test.npy"
# csv_file_name = "robosuite_data_test.csv"

recorded_data = np.load(file_name, allow_pickle=True)

print("Welcome to robosuite v{}!".format(suite.__version__))
print(suite.__logo__)

options["env_name"] = 'DoorMirror'
# options["env_name"] = 'Door'
options["robots"] = 'GH360'
options["gripper_types"] = 'HookGripper'
controller_name = 'JOINT_VELOCITY'
options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)
options["table_offset"] = (-0.43, 0.412, 0.81)
# options["table_offset"] = (-0.35, 0.5, 0.75)

env = suite.make(
    **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_object_obs=True,
        # ignore_done=True,
        use_camera_obs=False,
        hard_reset=False,
        ignore_done=True,
        obs_optimization=True,
        # control_freq=20,
        reward_shaping=True,
        render_camera="agentview",
)

env = NormalizedBoxEnv(GymWrapper(env))
obs = env.reset()
    

plot_data = False

demo_data = np.load("demonstrations/robosuite_door_mirror_demonstration_v1.npy", allow_pickle=True)

# print(f"demo_data shape: {demo_data[0]}")
X_train = []
y_train = []
new_X_train = []
new_y_train = []

for j in range(20):
    for i, action in enumerate(demo_data[j]["actions"]):
        # print(f"action: {action}")
        new_X_train.append(demo_data[j]["observations"][i])
        new_y_train.append(action)
        # X_train = np.append(X_train, action)
        # y_train = np.append(y_train, i)
    #     print(f"demo shape: {demo.shape}")

# X_train.append(new_X_train)
# y_train.append(new_y_train)

# X_train = np.array(X_train).transpose()
X_train = np.array(new_X_train)
y_train = np.array(new_y_train)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process = GaussianProcessRegressor()
gaussian_process.fit(X_train, y_train)
# gaussian_process.kernel_

# print(f"min X_train: {len(np.min(X_train, axis=0))}")
# X_test = np.linspace(start=np.min(X_train, axis=0), stop=np.max(X_train, axis=0), num=499).reshape(-1, 1)

# X_test = []
# for i in range(X_train.shape[1]):
#     X_test.append(np.linspace(start=np.min(X_train[i], axis=0), stop=np.max(X_train[i], axis=0), num=500))

# X_test = np.array(X_test).transpose()

# mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)

if plot_data:
    disp_action = 4
    # plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.scatter(X_train, y_train[:,disp_action], label="Observations")
    plt.plot(X_test, mean_prediction[:,disp_action], label="Mean prediction", color="orange")
    plt.fill_between(
        X_test.ravel(),
        mean_prediction[:,disp_action] - 1.96 * std_prediction[:,disp_action],
        mean_prediction[:,disp_action] + 1.96 * std_prediction[:,disp_action],
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("Gaussian process regression on noise-free dataset")
    plt.show()

obs_list = []

for i in range(500):
    obs_list.append(obs)
    npy_obs = np.array(obs_list)
    p_obs = npy_obs[i].reshape(1, -1)
    # print(f"obs_list shape: {npy_obs.shape}")
    # print(f"p_obs shape: {p_obs.shape}")
    action = gaussian_process.predict(p_obs)
    # print(f"action: {action}")
    obs, reward, done, _ = env.step(np.squeeze(action))
    # obs_list.append(obs)
    env.render()

if reward == 1.0:
    print("Success")
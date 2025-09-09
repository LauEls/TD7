import pandas
import numpy as np

panda_goal_eef_vel = pandas.read_csv('door/real_gh360/eef_vel/online/v8_corl_with_demos/run_2/goal_eef_vel_test.csv')
goal_eef_vel = panda_goal_eef_vel.to_numpy()
panda_joint_states = pandas.read_csv('door/real_gh360/eef_vel/online/v8_corl_with_demos/run_2/joint_states_test.csv')
joint_states = panda_joint_states.to_numpy()
panda_motor_states = pandas.read_csv('door/real_gh360/eef_vel/online/v8_corl_with_demos/run_2/motor_states_test.csv')
motor_states = panda_motor_states.to_numpy()


episode_start_times = []

for i in range(1, goal_eef_vel.shape[0]):
    ep_start = True
    for j in range(1, goal_eef_vel.shape[1]):
        if goal_eef_vel[i, j] == 0.0 or goal_eef_vel[i-1, j] != 0.0:
            ep_start = False
            break

    if ep_start:
        episode_start_times.append(goal_eef_vel[i-1, 0])

print(len(episode_start_times))

episode_start_joint_states = []
episode_start_motor_states = []

for ep_start_time in episode_start_times:
    closest_idx = np.argmin(np.abs(joint_states[:, 0] - ep_start_time))
    episode_start_joint_states.append(joint_states[closest_idx, 1:])

    closest_idx = np.argmin(np.abs(motor_states[:, 0] - ep_start_time))
    episode_start_motor_states.append(motor_states[closest_idx, 1:])

np.set_printoptions(precision=6, suppress=True)
print(episode_start_joint_states[0:10])


print("Average joint states at episode start:")
print(np.mean(episode_start_joint_states, axis=0))
print(np.var(episode_start_joint_states, axis=0))

test_mean = 0
for js in episode_start_joint_states:
    test_mean += js[0]
test_mean /= len(episode_start_joint_states)
print("Manual mean calculation:")
print(test_mean)

test_variance = 0
for js in episode_start_joint_states:
    test_variance += (js[0] - test_mean) ** 2
test_variance /= len(episode_start_joint_states)
print("Manual variance calculation:")
print(test_variance)

print("Average motor states at episode start:")
print(np.mean(episode_start_motor_states, axis=0))
print(np.var(episode_start_motor_states, axis=0))

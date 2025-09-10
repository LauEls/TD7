import pandas
import numpy as np
import matplotlib.pyplot as plt

def find_start_states(goal_eef_vels, joint_states, motor_states):
    episode_start_times = []

    for i in range(1, goal_eef_vels.shape[0]):
        ep_start = True
        for j in range(1, goal_eef_vels.shape[1]):
            if goal_eef_vels[i, j] == 0.0 or goal_eef_vels[i-1, j] != 0.0:
                ep_start = False
                break

        if ep_start:
            episode_start_times.append(goal_eef_vels[i-1, 0])

    episode_start_joint_states = []
    episode_start_motor_states = []

    for ep_start_time in episode_start_times:
        closest_idx = np.argmin(np.abs(joint_states[:, 0] - ep_start_time))
        episode_start_joint_states.append(joint_states[closest_idx, 1:])

        closest_idx = np.argmin(np.abs(motor_states[:, 0] - ep_start_time))
        episode_start_motor_states.append(motor_states[closest_idx, 1:])

    return episode_start_joint_states, episode_start_motor_states

def calc_variance(joint_mean, motor_mean, ep_start_joint_states, ep_start_motor_states):

    joint_state_variance = 0
    for js in ep_start_joint_states:
        joint_state_variance += (js - joint_mean) ** 2
    joint_state_variance /= len(ep_start_joint_states)

    motor_state_variance = 0
    for ms in ep_start_motor_states:
        motor_state_variance += (ms - motor_mean) ** 2
    motor_state_variance /= len(ep_start_motor_states)

    return joint_state_variance, motor_state_variance

    
    

# panda_goal_eef_vel = pandas.read_csv('door/real_gh360/eef_vel/online/v8_corl_with_demos/run_2/goal_eef_vel_test.csv')
# panda_joint_states = pandas.read_csv('door/real_gh360/eef_vel/online/v8_corl_with_demos/run_2/joint_states_test.csv')
# panda_motor_states = pandas.read_csv('door/real_gh360/eef_vel/online/v8_corl_with_demos/run_2/motor_states_test.csv')
panda_goal_eef_vel = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/goal_eef_vel.csv')
panda_joint_states = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/joint_states.csv')
panda_motor_states = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/motor_states.csv')
panda_final_eval_goal_eef_vel = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/final_eval_goal_eef_vel.csv')
panda_final_eval_joint_states = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/final_eval_joint_states.csv')
panda_final_eval_motor_states = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/final_eval_motor_states.csv')

goal_eef_vel = panda_goal_eef_vel.to_numpy()
joint_states = panda_joint_states.to_numpy()
motor_states = panda_motor_states.to_numpy()

final_eval_goal_eef_vel = panda_final_eval_goal_eef_vel.to_numpy()
final_eval_joint_states = panda_final_eval_joint_states.to_numpy()
final_eval_motor_states = panda_final_eval_motor_states.to_numpy()

ep_start_joint_states, ep_start_motor_states = find_start_states(goal_eef_vel, joint_states, motor_states)
final_eval_ep_start_joint_states, final_eval_ep_start_motor_states = find_start_states(final_eval_goal_eef_vel, final_eval_joint_states, final_eval_motor_states)

# ep_start_joint_states = np.concatenate((ep_start_joint_states, final_eval_ep_start_joint_states), axis=0)
# ep_start_motor_states = np.concatenate((ep_start_motor_states, final_eval_ep_start_motor_states), axis=0)
ep_start_joint_states = np.array(ep_start_joint_states)
ep_start_motor_states = np.array(ep_start_motor_states)
ep_start_motor_state_reduced = np.array([(ep_start_motor_states[:,0]+ep_start_motor_states[:,1])/2, (ep_start_motor_states[:,2]+ep_start_motor_states[:,3])/2, (ep_start_motor_states[:,4]+ep_start_motor_states[:,5])/2, (ep_start_motor_states[:,6]+ep_start_motor_states[:,7])/2, (ep_start_motor_states[:,8]+ep_start_motor_states[:,9])/2, ep_start_motor_states[:,10], (ep_start_motor_states[:,11]+ep_start_motor_states[:,12])/2]).T
final_eval_ep_start_joint_states = np.array(final_eval_ep_start_joint_states)
final_eval_ep_start_motor_states = np.array(final_eval_ep_start_motor_states)
final_eval_ep_start_motor_state_reduced = np.array([(final_eval_ep_start_motor_states[:,0]+final_eval_ep_start_motor_states[:,1])/2, (final_eval_ep_start_motor_states[:,2]+final_eval_ep_start_motor_states[:,3])/2, (final_eval_ep_start_motor_states[:,4]+final_eval_ep_start_motor_states[:,5])/2, (final_eval_ep_start_motor_states[:,6]+final_eval_ep_start_motor_states[:,7])/2, (final_eval_ep_start_motor_states[:,8]+final_eval_ep_start_motor_states[:,9])/2, final_eval_ep_start_motor_states[:,10], (final_eval_ep_start_motor_states[:,11]+final_eval_ep_start_motor_states[:,12])/2]).T



print("Original motor states:")
print(np.array(ep_start_motor_states).shape)
print("Reduced motor states:")
print(ep_start_motor_state_reduced.shape)


joint_state_mean = np.mean(ep_start_joint_states, axis=0)
motor_state_mean = np.mean(ep_start_motor_state_reduced, axis=0)

joint_variance, motor_variance = calc_variance(joint_state_mean, motor_state_mean, ep_start_joint_states, ep_start_motor_state_reduced)
final_eval_joint_variance, final_eval_motor_variance = calc_variance(joint_state_mean, motor_state_mean, final_eval_ep_start_joint_states, final_eval_ep_start_motor_state_reduced)


motor_variance *= 10**3
final_eval_motor_variance *= 10**3
joint_variance *= 10**3
final_eval_joint_variance *= 10**3

np.set_printoptions(precision=6, suppress=True)

print("Joint mean:")
print(joint_state_mean)
print("Motor mean:")
print(motor_state_mean)

print("Joint variances:")
print(joint_variance)
print(final_eval_joint_variance)
print("Motor variances:")
print(motor_variance)
print(final_eval_motor_variance)

metrics = ["Shoulder Yaw", "Shoulder Roll", "Shoulder Pitch", "Uppperarm Roll", "Elbow", "Forearm Roll", "Wrist Pitch"]
theta = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
theta = np.concatenate((theta, [theta[0]]))
fig, ax = plt.subplots(figsize=(13, 12), subplot_kw={'projection': 'polar'})
# Title
ax.set_title("Motor Position [rad*10^3] Variance in Reset Position at Episode Start",y=1.1, fontsize=20)
# Direction of the zero angle to the north (upwards)
ax.set_theta_zero_location("N")
# Direction of the angles to be counterclockwise
ax.set_theta_direction(-1)
# Radial label position (position of values on the radial axes)
ax.set_rlabel_position(90)
# Make radial gridlines appear behind other elements
ax.spines['polar'].set_zorder(1)
# Color of radial girdlines
ax.spines['polar'].set_color('lightgrey')

color_palette = ['#339F00', '#0500FF', '#9CDADB', '#FF00DE', '#FF9900', '#FFFFFF']

values = np.concatenate((final_eval_motor_variance, [final_eval_motor_variance[0]]))
ax.plot(theta, values, linewidth=1.75, linestyle='solid', label="Motor Variance during Final Evaluation", marker='o', markersize=10, color=color_palette[1])
ax.fill(theta, values, alpha=0.50, color=color_palette[1])  
values = np.concatenate((motor_variance, [motor_variance[0]]))
ax.plot(theta, values, linewidth=1.75, linestyle='solid', label="Motor Variance during Training", marker='o', markersize=10, color=color_palette[0])
ax.fill(theta, values, alpha=0.50, color=color_palette[0])
values = np.concatenate((final_eval_joint_variance, [final_eval_joint_variance[0]]))
ax.plot(theta, values, linewidth=1.75, linestyle='solid', label="Joint Variance during Final Evaluation", marker='o', markersize=10, color=color_palette[2])
ax.fill(theta, values, alpha=0.50, color=color_palette[2])
values = np.concatenate((joint_variance, [joint_variance[0]]))
ax.plot(theta, values, linewidth=1.75, linestyle='solid', label="Joint Variance during Training", marker='o', markersize=10, color=color_palette[4])
ax.fill(theta, values, alpha=0.50, color=color_palette[4])

plt.yticks([0, 5, 10, 15, 20, 25], ["0", "5", "10", "15", "20", "25"], color="black", size=12)
plt.xticks(theta, metrics + [metrics[0]], color="black", size=12)
plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.05))
plt.show()

# print(final_eval_joint_mean)
# print("Joint variances:")
# print(joint_variance)
# print(final_eval_joint_variance)
# print("Motor means:")
# print(motor_mean)
# print(final_eval_motor_mean)
# print("Motor variances:")
# print(motor_variance)
# print(final_eval_motor_variance)


# episode_start_times = []

# for i in range(1, goal_eef_vel.shape[0]):
#     ep_start = True
#     for j in range(1, goal_eef_vel.shape[1]):
#         if goal_eef_vel[i, j] == 0.0 or goal_eef_vel[i-1, j] != 0.0:
#             ep_start = False
#             break

#     if ep_start:
#         episode_start_times.append(goal_eef_vel[i-1, 0])

# print(len(episode_start_times))

# episode_start_joint_states = []
# episode_start_motor_states = []

# for ep_start_time in episode_start_times:
#     closest_idx = np.argmin(np.abs(joint_states[:, 0] - ep_start_time))
#     episode_start_joint_states.append(joint_states[closest_idx, 1:])

#     closest_idx = np.argmin(np.abs(motor_states[:, 0] - ep_start_time))
#     episode_start_motor_states.append(motor_states[closest_idx, 1:])

# np.set_printoptions(precision=6, suppress=True)
# print(episode_start_joint_states[0:10])


# print("Average joint states at episode start:")
# print(np.mean(episode_start_joint_states, axis=0))
# print(np.var(episode_start_joint_states, axis=0))

# test_mean = 0
# for js in episode_start_joint_states:
#     test_mean += js[0]
# test_mean /= len(episode_start_joint_states)
# print("Manual mean calculation:")
# print(test_mean)

# test_variance = 0
# for js in episode_start_joint_states:
#     test_variance += (js[0] - test_mean) ** 2
# test_variance /= len(episode_start_joint_states)
# print("Manual variance calculation:")
# print(test_variance)

# print("Average motor states at episode start:")
# print(np.mean(episode_start_motor_states, axis=0)*10**-3)
# print(np.var(episode_start_motor_states, axis=0)*10**-3)

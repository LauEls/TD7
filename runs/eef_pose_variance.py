import pandas
import numpy as np
import matplotlib.pyplot as plt

# def end_time(goal_eef_vel):
#     if goal_eef_vel != 0.0 or goal_eef_vel == 0.0:
#         return True
#     return False

# def end_

def find_episode_success(env_observations):
    success_indices = []
    for i in range(1, env_observations.shape[0]):
        if env_observations[i, 4] >= 1.57 and env_observations[i-1, 4] < 1.57:
            success_indices.append(i)
    return success_indices

def find_start_end_times(goal_eef_vels, end_evaluation):
    episode_start_times = []
    episode_end_times = []

    for i in range(1, goal_eef_vels.shape[0]):
        ep_start = True
        ep_end = True
        for j in range(1, goal_eef_vels.shape[1]):
            if goal_eef_vels[i, j] == 0.0 or goal_eef_vels[i-1, j] != 0.0:
                ep_start = False
            if goal_eef_vels[i, j] != 0.0 or goal_eef_vels[i-1, j] == 0.0:
            # if end_evaluation:
                ep_end = False


        if ep_start:
            episode_start_times.append(goal_eef_vels[i, 0])
        if ep_end:
            episode_end_times.append(goal_eef_vels[i-1, 0])


            
    if len(episode_start_times) > len(episode_end_times):
        episode_start_times = episode_start_times[:-1]

        
    return episode_start_times, episode_end_times

def filter_eef_poses(eef_poses, start_times, end_times):
    filtered_poses = []
    new_poses = []
    time_idx = 0
    for i in range(eef_poses.shape[0]):
        if eef_poses[i, 0] >= start_times[time_idx]:
                new_poses.append(eef_poses[i, 1:])
                if eef_poses[i, 0] >= end_times[time_idx]:
                    time_idx += 1
                    filtered_poses.append(np.array(new_poses))
                    new_poses = []
                    if time_idx >= len(start_times) or time_idx >= len(end_times):
                        break

        

    return np.array(filtered_poses)

panda_goal_eef_vel = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/goal_eef_vel.csv')
panda_eef_poses = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/eef_pose.csv')

goal_eef_vel = panda_goal_eef_vel.to_numpy()
eef_poses = panda_eef_poses.to_numpy()

print(f"Goal EEF velocities shape: {goal_eef_vel.shape}")
print(f"EEF poses shape: {eef_poses.shape}")

ep_start_times, ep_end_times = find_start_end_times(goal_eef_vel)


print(f"Episode start times: {len(ep_start_times)}")
print(f"Episode end times: {len(ep_end_times)}")


ep_durations = []
for i in range(len(ep_start_times)):
    new_duration = ep_end_times[i] - ep_start_times[i]
    if new_duration < 0.0:
        print(f"Episode {i} has negative duration!")
    ep_durations.append(new_duration)
    if new_duration < 15e9:
        print(f"Episode {i} has short duration: {new_duration*1e-9}s")

print(f"Episode mean duration: {np.mean(ep_durations)*1e-9}s")
print(f"Episode std duration: {np.std(ep_durations)*1e-9}s")
print(f"Episode min duration: {np.min(ep_durations)*1e-9}s")
print(f"Episode max duration: {np.max(ep_durations)*1e-9}s")

eef_poses_filtered = filter_eef_poses(eef_poses, ep_start_times, ep_end_times)
print(f"Filtered EEF poses: {eef_poses_filtered.shape}")
print(f"Example episode length: {eef_poses_filtered[0].shape}")
print(f"Example episode first pose: {eef_poses_filtered[0][0]}")
print(f"Max episode length: {max([ep.shape[0] for ep in eef_poses_filtered])}")
print(f"Min episode length: {min([ep.shape[0] for ep in eef_poses_filtered])}")

min_len = min([ep.shape[0] for ep in eef_poses_filtered])
mean_start_pos = np.mean([ep[0] for ep in eef_poses_filtered], axis=0)
print(f"Mean start position: {mean_start_pos}")
eef_positions_resampled = []

for i in range(len(eef_poses_filtered)):
    t = np.linspace(0, eef_poses_filtered[i].shape[0], eef_poses_filtered[i].shape[0])
    tt = np.linspace(0, eef_poses_filtered[i].shape[0], min_len)
    xx = np.interp(tt, t, eef_poses_filtered[i][:, 0])
    yy = np.interp(tt, t, eef_poses_filtered[i][:, 1])
    zz = np.interp(tt, t, eef_poses_filtered[i][:, 2])
    eef_positions_resampled.append(np.vstack((xx, yy, zz)).T)

eef_positions_resampled = np.array(eef_positions_resampled)
print(f"Resampled EEF positions: {eef_positions_resampled.shape}")
start_pos = np.mean(eef_positions_resampled[:, 0, :], axis=0)
start_pos_2 = np.array([0.09667707, 0.33416114, 0.26332604])
print(f"Start position: {start_pos}")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(start_pos_2[0], start_pos_2[1], start_pos_2[2], color='red', s=50, label='Start')
door_handle_pos = np.array([0.12883546,  0.34898176,  0.16951997])
ax.scatter3D(door_handle_pos[0], door_handle_pos[1], door_handle_pos[2], color='green', s=50, label='Door Handle')

colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'teal', 'navy', 'maroon', 'lime', 'coral', 'gold', 'indigo', 'violet']
for i in range(20):
    ax.plot3D(eef_positions_resampled[-i][:, 0], eef_positions_resampled[-i][:, 1], eef_positions_resampled[-i][:, 2], colors[i%len(colors)])
    # ax.plot3D(eef_poses_filtered[-i-1][:, 0], eef_poses_filtered[-i-1][:, 1], eef_poses_filtered[-i-1][:, 2], colors[i])
# t = np.linspace(0, eef_poses_filtered[-1].shape[0], eef_poses_filtered[-1].shape[0])
# tt = np.linspace(0, eef_poses_filtered[-1].shape[0], 260)
# xx = np.interp(tt, t, eef_poses_filtered[-1][:, 1])
# yy = np.interp(tt, t, eef_poses_filtered[-1][:, 2])
# zz = np.interp(tt, t, eef_poses_filtered[-1][:, 3])

# ax.plot3D(xx, yy, zz, 'red')



ax.set_title('3D EEF Pose Trajectory for last Episode')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)') 
plt.show()



# fig2 = plt.figure()
# ax2 = plt.axes(projection='3d')

# ax2.set_title('3D EEF Pose Trajectory for last Episode')
# ax2.set_xlabel('X Position (m)')
# ax2.set_ylabel('Y Position (m)')
# ax2.set_zlabel('Z Position (m)') 
# plt.show()
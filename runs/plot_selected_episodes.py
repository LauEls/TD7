import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def filter_eef_poses(eef_poses, event_times):
    free_space_poses = []
    handle_move_poses = []
    hinge_move_poses = []
    episode_success_poses = []
    free_space_subsequences = []
    handle_move_subsequences = []
    hinge_move_subsequences = []
    episode_success_subsequences = []
    new_free_space_poses = []
    new_handle_move_poses = []
    new_hinge_move_poses = []
    new_episode_success_poses = []
    time_idx = 0
    state = 0

    for i in range(eef_poses.shape[0]):
        if eef_poses[i, 0] >= event_times[time_idx][0]:
            if state == 1 or state == 5:  # After last episode end
                new_free_space_poses.append(eef_poses[i])
                free_space_subsequences.append(np.array(new_free_space_poses))
                new_free_space_poses = []
            elif state == 2:
                new_handle_move_poses.append(eef_poses[i])
                handle_move_subsequences.append(np.array(new_handle_move_poses))
                new_handle_move_poses = []
            elif state == 3:
                new_hinge_move_poses.append(eef_poses[i])
                hinge_move_subsequences.append(np.array(new_hinge_move_poses))
                new_hinge_move_poses = []
            elif state == 4:
                new_episode_success_poses.append(eef_poses[i])
                episode_success_subsequences.append(np.array(new_episode_success_poses))
                new_episode_success_poses = []
                
            state = event_times[time_idx][1]
            if state == 0:  # Episode end
                if len(new_free_space_poses) > 0: free_space_subsequences.append(np.array(new_free_space_poses))
                if len(new_handle_move_poses) > 0: handle_move_subsequences.append(np.array(new_handle_move_poses))
                if len(new_hinge_move_poses) > 0: hinge_move_subsequences.append(np.array(new_hinge_move_poses))
                if len(new_episode_success_poses) > 0: episode_success_subsequences.append(np.array(new_episode_success_poses))
                free_space_poses.append(free_space_subsequences)
                handle_move_poses.append(handle_move_subsequences)
                hinge_move_poses.append(hinge_move_subsequences)
                episode_success_poses.append(episode_success_subsequences)
                free_space_subsequences = []
                handle_move_subsequences = []
                hinge_move_subsequences = []
                episode_success_subsequences = []
                new_free_space_poses = []
                new_handle_move_poses = []
                new_hinge_move_poses = []
                new_episode_success_poses = []
            if time_idx < len(event_times)-1: time_idx += 1
        # if state == 0:  # Before first episode start
        #     continue
        if state == 1 or state == 5:  # After last episode end
            new_free_space_poses.append(eef_poses[i])
        elif state == 2:
            new_handle_move_poses.append(eef_poses[i])
        elif state == 3:
            new_hinge_move_poses.append(eef_poses[i])
        elif state == 4:
            new_episode_success_poses.append(eef_poses[i])

            
        

        # if eef_poses[i, 0] >= event_times[time_idx][0]:
        #         if event_times[time_idx][1] == 0:  # Episode end
        #             time_idx += 1
        #             free_space_poses.append(np.array(new_free_space_poses))
        #             handle_move_poses.append(np.array(new_handle_move_poses))
        #             hinge_move_poses.append(np.array(new_hinge_move_poses))
        #             episode_success_poses.append(np.array(new_episode_success_poses))
        #             new_free_space_poses = []
        #             new_handle_move_poses = []
        #             new_hinge_move_poses = []
        #             new_episode_success_poses = []
        #             if time_idx >= len(event_times):
        #                 break
        #         elif event_times[time_idx][1] == 4:  # Episode success
        #             new_episode_success_poses.append(eef_poses[i])
        #             time_idx += 1
        #         elif event_times[time_idx][1] == 3:  # Hinge move
        #             new_hinge_move_poses.append(eef_poses[i])
        #             time_idx += 1
        #         elif event_times[time_idx][1] == 2:  # Handle move
        #             new_handle_move_poses.append(eef_poses[i])
        #             time_idx += 1
        #         else:  # Free space
        #             new_free_space_poses.append(eef_poses[i])

    filtered_eef_poses = {"free_space": free_space_poses, 
                          "handle_move": handle_move_poses, 
                          "hinge_move": hinge_move_poses, 
                          "episode_success": episode_success_poses}
    return filtered_eef_poses


def filter_eef_poses_old(eef_poses, start_times, end_times, handle_move_start_times, hinge_move_start_times, episode_success_times):
    # filtered_poses = []
    free_space_poses = []
    handle_move_poses = []
    hinge_move_poses = []
    episode_success_poses = []
    new_free_space_poses = []
    new_handle_move_poses = []
    new_hinge_move_poses = []
    new_episode_success_poses = []
    time_idx = 0
    handle_idx = 0
    hinge_idx = 0
    success_idx = 0
    success = False
    hinge_move = False
    handle_move = False

    # while handle_move_start_times[handle_idx] < start_times[time_idx]:
    #     handle_idx += 1
    #     if handle_idx >= len(handle_move_start_times):
    #         print("Handle idx exceeded")
    #         break
    # while hinge_move_start_times[hinge_idx] < start_times[time_idx]:
    #     hinge_idx += 1
    #     if hinge_idx >= len(hinge_move_start_times):
    #         print("Hinge idx exceeded")
    #         break
    # while episode_success_times[success_idx] < start_times[time_idx]:
    #     success_idx += 1
    #     if success_idx >= len(episode_success_times):
    #         print("Success idx exceeded")
    #         break
    for i in range(eef_poses.shape[0]):
        if eef_poses[i, 0] >= start_times[time_idx]:
                if eef_poses[i, 0] >= end_times[time_idx]:
                    time_idx += 1
                    # filtered_poses.append(np.array(new_poses))
                    free_space_poses.append(np.array(new_free_space_poses))
                    handle_move_poses.append(np.array(new_handle_move_poses))
                    hinge_move_poses.append(np.array(new_hinge_move_poses))
                    episode_success_poses.append(np.array(new_episode_success_poses))
                    new_free_space_poses = []
                    new_handle_move_poses = []
                    new_hinge_move_poses = []
                    new_episode_success_poses = []
                    # new_poses = []
                    if time_idx >= len(start_times) or time_idx >= len(end_times):
                        break

                    if success:
                        success = False
                        success_idx += 1
                        if success_idx >= len(episode_success_times):
                            success_idx -= 1
                    if hinge_move:
                        hinge_move = False
                        hinge_idx += 1
                        if hinge_idx >= len(hinge_move_start_times):
                            hinge_idx -= 1
                    if handle_move:
                        handle_move = False
                        handle_idx += 1
                        if handle_idx >= len(handle_move_start_times):
                            handle_idx -= 1
                elif eef_poses[i, 0] >= episode_success_times[success_idx]:
                    new_episode_success_poses.append(eef_poses[i])
                    success = True
                elif eef_poses[i, 0] >= hinge_move_start_times[hinge_idx]:
                    new_hinge_move_poses.append(eef_poses[i])
                    hinge_move = True
                elif eef_poses[i, 0] >= handle_move_start_times[handle_idx]:
                    new_handle_move_poses.append(eef_poses[i])
                    handle_move = True
                else:
                    new_free_space_poses.append(eef_poses[i])

        

    filtered_eef_poses = {"free_space": free_space_poses, 
                          "handle_move": handle_move_poses, 
                          "hinge_move": hinge_move_poses, 
                          "episode_success": episode_success_poses}
    return filtered_eef_poses

def state_evaluation(handle_angle, hinge_angle):
    if hinge_angle >= 0.4 and handle_angle <= 0.1:
        return 4  # Success
    elif hinge_angle > 0.1:
        return 3  # Hinge moved
    elif handle_angle > 0.1:
        return 2  # Handle moved
    else:
        return 5  # Free space

def find_start_end_times(goal_eef_vels):
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
            episode_start_times.append([goal_eef_vels[i, 0], 1])
        if ep_end:
            episode_end_times.append([goal_eef_vels[i-1, 0], 0])


    if len(episode_start_times) > len(episode_end_times):
        episode_start_times = episode_start_times[:-1]

        
    return episode_start_times, episode_end_times

def check_success(hinge_qpos, handle_qpos):
    return hinge_qpos >= 0.4 and handle_qpos <= 0.1

def state_transistion_times(env_observations):
    state_transistion_times = []
    for i in range(1, env_observations.shape[0]):
        prev_state = state_evaluation(env_observations[i-1, 4], env_observations[i-1, 5])
        curr_state = state_evaluation(env_observations[i, 4], env_observations[i, 5])
        if curr_state != prev_state:
            state_transistion_times.append([env_observations[i, 0], curr_state])

    return state_transistion_times

def episode_success_times(env_observations):
    episode_success_times = []
    for i in range(1, env_observations.shape[0]):
        if check_success(env_observations[i, 5], env_observations[i, 4]) and not check_success(env_observations[i-1, 5], env_observations[i-1, 4]):
            episode_success_times.append([env_observations[i, 0], 4])
            
    return episode_success_times


def handle_angle_reached_times(env_observations, target_angle=0.6):
    handle_angle_reached_times = []
    for i in range(1, env_observations.shape[0]):
        if env_observations[i, 5] < 0.1:
            if env_observations[i, 4] >= target_angle and env_observations[i-1, 4] < target_angle:
                handle_angle_reached_times.append([env_observations[i, 0], 2])
    return handle_angle_reached_times

def hinge_angle_reached_times(env_observations, target_angle=0.4):
    hinge_angle_reached_times = []
    for i in range(1, env_observations.shape[0]):
        if env_observations[i, 5] >= target_angle and env_observations[i-1, 5] < target_angle:
            hinge_angle_reached_times.append([env_observations[i, 0], 3])
    return hinge_angle_reached_times

def plot_trajectories(start_idx, stop_idx, step_size, filtered_poses, x_lims, y_lims, z_lims, trajectory_line_width=5, alpha=0.8):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    axis_fontsize = 18
    # trajectory_line_width = 5

    start_pos_2 = np.array([0.09667707, 0.33416114, 0.26332604])
    door_handle_pos = np.array([0.12883546,  0.34898176,  0.16951997])

    start_text = ax.text(start_pos_2[0], start_pos_2[1], start_pos_2[2]+0.01, "Start Position", color='red', size=14, weight='bold')
    start_text.set_path_effects([
        pe.Stroke(linewidth=3, foreground='white'), # Border color and width
        pe.Normal() # Ensures the text is drawn on top of the stroke
    ])
    
    door_handle_text = ax.text(door_handle_pos[0], door_handle_pos[1], door_handle_pos[2]+0.01, "Door Handle", color='orange', size=14, weight='bold')
    door_handle_text.set_path_effects([
        pe.Stroke(linewidth=3, foreground='white'), # Border color and width
        pe.Normal() # Ensures the text is drawn on top of the stroke
    ])
    
    start_point = ax.scatter3D(start_pos_2[0], start_pos_2[1], start_pos_2[2], color='red', s=100, label='Start')
    start_point.set_path_effects([
        pe.Stroke(linewidth=5, foreground='white'), # Border color and width
        pe.Normal() # Ensures the text is drawn on top of the stroke
    ])
    
    door_handle_point = ax.scatter3D(door_handle_pos[0], door_handle_pos[1], door_handle_pos[2], color='orange', s=100, label='Door Handle')
    door_handle_point.set_path_effects([
        pe.Stroke(linewidth=5, foreground='white'), # Border color and width
        pe.Normal() # Ensures the text is drawn on top of the stroke
    ])

    # start_idx = 0
    # stop_idx = 44
    # start_idx = 55
    # stop_idx = 99
    # # start_idx = 88
    # # stop_idx = 132
    # # start_idx = 110
    # # stop_idx = 154
    # # start_idx = 165
    # # stop_idx = 220
    # step_size = 11

    # # start_idx = 1
    # # stop_idx = 5
    # start_idx = 100
    # stop_idx = 109
    # # start_idx = 110
    # # stop_idx = 154
    # # start_idx = 165
    # # stop_idx = 220
    # step_size = 1

    for i in range(start_idx, stop_idx+1, step_size):
        if len(filtered_poses['free_space'][i]) != 0:
            for j in range(len(filtered_poses['free_space'][i])):
                ax.plot3D(filtered_poses['free_space'][i][j][:,1], filtered_poses['free_space'][i][j][:,2], filtered_poses['free_space'][i][j][:,3], color=(0.8500, 0.3250, 0.0980), alpha=alpha, label='Free Space', linewidth=trajectory_line_width)
        if len(filtered_poses['handle_move'][i]) != 0:
            for j in range(len(filtered_poses['handle_move'][i])):
                ax.plot3D(filtered_poses['handle_move'][i][j][:,1], filtered_poses['handle_move'][i][j][:,2], filtered_poses['handle_move'][i][j][:, 3], color=(0.9290, 0.6940, 0.1250), alpha=alpha, label='Handle Move', linewidth=trajectory_line_width)
        if len(filtered_poses['hinge_move'][i]) != 0:
            for j in range(len(filtered_poses['hinge_move'][i])):
                ax.plot3D(filtered_poses['hinge_move'][i][j][:,1], filtered_poses['hinge_move'][i][j][:,2], filtered_poses['hinge_move'][i][j][:,3], color=(0.4660, 0.6740, 0.1880), alpha=alpha, label='Hinge Move', linewidth=trajectory_line_width)
        # if len(filtered_poses['episode_success'][i]) != 0:
        #     for j in range(len(filtered_poses['episode_success'][i])):
        #         ax.plot3D(filtered_poses['episode_success'][i][j][:,1], filtered_poses['episode_success'][i][j][:,2], filtered_poses['episode_success'][i][j][:,3], 'green', label='Episode Success', linewidth=trajectory_line_width)
    # ax.legend()

    # ax.set_title('3D EEF Pose Trajectory for last Episode')
    ax.set_xlabel('X Position (m)', fontsize=axis_fontsize, labelpad=30)
    ax.set_ylabel('Y Position (m)', fontsize=axis_fontsize, labelpad=30)
    ax.set_zlabel('Z Position (m)', fontsize=axis_fontsize, labelpad=30)

    # ax.xaxis.set_tick_params(labelsize=axis_fontsize)
    # ax.yaxis.set_tick_params(labelsize=axis_fontsize)
    # ax.zaxis.set_tick_params(labelsize=axis_fontsize)
    

    # 1. Define your specific limits
    # x_lims = [0.05, 0.15]
    # y_lims = [0.15, 0.35]
    # z_lims = [0.09, 0.275]

    # ax.set_xticks(np.linspace(x_lims[0], x_lims[1], num=3))
    # ax.set_yticks(np.linspace(y_lims[0], y_lims[1], num=5))
    # ax.set_zticks(np.linspace(z_lims[0], z_lims[1], num=6))

    ax.autoscale(True)

    # ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.yaxis.set_major_locator(MultipleLocator(5))

    # # Change minor ticks to show every 5. (20/4 = 5)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_zlim(z_lims)

    # 2. Calculate the ranges
    x_range = x_lims[1] - x_lims[0]
    y_range = y_lims[1] - y_lims[0]
    z_range = z_lims[1] - z_lims[0]

    # 3. Set the box aspect ratio to match the data range ratio
    # This ensures 1 unit is physically equal on all axes
    ax.set_box_aspect((x_range, y_range, z_range))


    # plt.axis('equal')
    # ax.set_xlim([0.1, 0.45])
    # ax.set_ylim([-0.275, 0.075])
    # ax.set_zlim([-0.05, 0.3])

    ax.view_init(elev=22, azim=45, roll=0)
    


if __name__ == "__main__":
    panda_goal_eef_vel = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/goal_eef_vel.csv')
    panda_eef_poses = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/eef_pose.csv')
    panda_env_obs = pandas.read_csv('door/real_gh360/eef_vel/online/v14_video_recording/run_0/environment_observations.csv')

    goal_eef_vel = panda_goal_eef_vel.to_numpy()
    eef_poses = panda_eef_poses.to_numpy()
    env_observations = panda_env_obs.to_numpy()

    print(f"Goal EEF velocities shape: {goal_eef_vel.shape}")
    print(f"EEF poses shape: {eef_poses.shape}")
    print(f"Environment observations shape: {env_observations.shape}")

    ep_start_times, ep_end_times = find_start_end_times(goal_eef_vel)
    print(f"Episode start times: {len(ep_start_times)}")
    print(f"Episode end times: {len(ep_end_times)}")


    # ep_durations = []
    # for i in range(len(ep_start_times)):
    #     new_duration = ep_end_times[i] - ep_start_times[i]
    #     if new_duration < 0.0:
    #         print(f"Episode {i} has negative duration!")
    #     ep_durations.append(new_duration)
    #     if new_duration < 15e9:
    #         print(f"Episode {i} has short duration: {new_duration*1e-9}s")

    # print(f"Episode mean duration: {np.mean(ep_durations)*1e-9}s")
    # print(f"Episode std duration: {np.std(ep_durations)*1e-9}s")
    # print(f"Episode min duration: {np.min(ep_durations)*1e-9}s")
    # print(f"Episode max duration: {np.max(ep_durations)*1e-9}s")

    # ep_success_times = episode_success_times(env_observations)
    # print(f"Episode success times: {len(ep_success_times)}")
    # handle_move_start_times = handle_angle_reached_times(env_observations, target_angle=0.1)
    # print(f"Handle move start times: {len(handle_move_start_times)}")
    # hinge_move_start_times = hinge_angle_reached_times(env_observations, target_angle=0.1)
    # print(f"Hinge move start times: {len(hinge_move_start_times)}")

    ep_state_trans_times = state_transistion_times(env_observations)

    all_event_times = ep_start_times + ep_end_times + ep_state_trans_times
    all_event_times.sort(key=lambda x: x[0])
    print("All event times (first 10):")
    for i, t in enumerate(all_event_times):
        if i-1 >= 0:
            if t[1] == 5 and all_event_times[i-1][1] == 1:
                all_event_times.remove(t)
            # if t[1] != 1 and all_event_times[i-1][1] == 0:
            #     all_event_times.remove(t)
        if i+1 < len(all_event_times):
            if t[1] == 5 and all_event_times[i+1][1] == 1:
                all_event_times.remove(t)

    # between_eps = False
    # for t in all_event_times:
    #     if t[1] == 0:
    #         between_eps = True
    #     elif t[1] == 1 and between_eps:
    #         between_eps = False

    #     if (t[1] != 1 and t[1] != 0) and between_eps:
    #         all_event_times.remove(t)

    between_eps = False
    episode_success = False
    filtered_event_times = []
    for t in all_event_times:
        if t[1] == 4: t[1] = 0

        if t[1] == 0 and not between_eps:
            between_eps = True 
            filtered_event_times.append(t)
            continue
            # if episode_success:
            #     episode_success = False
        elif t[1] == 1 and between_eps:
            between_eps = False
        # elif t[1] == 4:
        #     episode_success = True
        

        # if (between_eps and t[1] != 0) or episode_success:
        if between_eps:
            continue

        filtered_event_times.append(t)

    for t in filtered_event_times[-20:]:
        print(f"  Time: {t[0]}, Event Type: {t[1]}")

    filtered_poses = filter_eef_poses(eef_poses, filtered_event_times)

    print(f"Filtered EEF poses:")
    print(f"  Free space poses: {filtered_poses['free_space'][218][0].shape}")
    print(f"  Handle move poses: {filtered_poses['handle_move'][218][0].shape}")
    print(f"  Hinge move poses: {filtered_poses['hinge_move'][218][0].shape}")
    # print(f"  Episode success poses: {filtered_poses['episode_success'][218][0].shape}") 

    plot_trajectories(
        start_idx=0, 
        stop_idx=44, 
        step_size=11, 
        filtered_poses=filtered_poses, 
        x_lims=[0.0, 0.35], 
        y_lims=[-0.1, 0.45], 
        z_lims=[0.09, 0.35], 
        trajectory_line_width=5
    )

    plot_trajectories(
        start_idx=55, 
        stop_idx=99, 
        step_size=11, 
        filtered_poses=filtered_poses, 
        x_lims=[0.05, 0.15], 
        y_lims=[0.15, 0.35], 
        z_lims=[0.09, 0.275], 
        trajectory_line_width=5
    )

    plot_trajectories(
        start_idx=165, 
        stop_idx=220, 
        step_size=11, 
        filtered_poses=filtered_poses, 
        x_lims=[0.05, 0.12], 
        y_lims=[0.15, 0.35], 
        z_lims=[0.09, 0.275], 
        trajectory_line_width=5
    )

    plt.show()
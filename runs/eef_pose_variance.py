import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# def end_time(goal_eef_vel):
#     if goal_eef_vel != 0.0 or goal_eef_vel == 0.0:
#         return True
#     return False

def calculate_tangent_vector(p1, p2):
    tangent = p2 - p1
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return tangent / tangent_norm

def calculate_perpendicular_vectors(normal):
    # Choose an arbitrary vector that is not parallel to the tangent
    arbitrary_vector = np.array([0, 0, 1])
    if np.isclose(np.abs(np.dot(normal, arbitrary_vector)), 1): # If parallel
        arbitrary_vector = np.array([1, 0, 0]) # Use another

    x_axis = np.cross(normal, arbitrary_vector)
    x_axis = x_axis / np.linalg.norm(x_axis) # Normalize X

    # 3. Find the second perpendicular vector (Y-axis)
    y_axis = np.cross(normal, x_axis)

    return x_axis, y_axis

# def calculate_perpendicular_vectors(points, normal):
 

def generate_circle_points(center, normal, radius, num_points=100):
    x_axis, y_axis = calculate_perpendicular_vectors(normal)
    angles = np.linspace(0, 2 * np.pi, num_points)
    points = []
    for angle in angles:
        point = center + radius * (np.cos(angle) * x_axis + np.sin(angle) * y_axis)
        points.append(point)
    return np.array(points)

def generate_elipsis_points(center, normal, radius_x, radius_y, num_points=100):
    x_axis, y_axis = calculate_perpendicular_vectors(normal)
    angles = np.linspace(0, 2 * np.pi, num_points)
    points = []
    for angle in angles:
        point = center + radius_x * np.cos(angle) * x_axis + radius_y * np.sin(angle) * y_axis
        points.append(point)
    return np.array(points)

def fit_ellipsoid_pca(points):
    #Center the data
    offset = np.mean(points, axis=0)
    xc = points - offset

    # Calculate the covariance matrix
    cov_matrix = np.cov(xc, rowvar=False)

    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    # The eigenvectors (Vh.T) give the orientation
    # The eigenvalues relate to the axis lengths
    U, S, Vh = np.linalg.svd(cov_matrix)
    
    # S contains the eigenvalues. The square roots of these are proportional to 
    # the standard deviations along the principal axes. For a bounding ellipsoid, 
    # you can use a scaling factor (e.g., 2 or 3 for 2 or 3 standard deviations).
    # The axis lengths (radii) are derived from S.
    # Note: A simple heuristic for axis lengths might be needed depending on data distribution.
    radii = np.sqrt(S) * 1. # Example scaling

    # The orientation matrix is the transpose of Vh (or just Vh if you use the U, S, Vh output of numpy's svd)
    orientation = Vh.T
    rotation = Vh

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + offset

    return x, y, z

def surface_of_ellipsoid(mean, variance):
    rx, ry, rz = np.sqrt(variance)

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    x += mean[0]
    y += mean[1]
    z += mean[2]

    return x, y, z

def find_episode_end_times(episode_start_times, episode_end_times, episode_success_times):
    episode_ends = []
    success_time_idx = 0
    cntr = 0
    print(f"Episode success times: {len(episode_success_times)}")
    for i in range(len(episode_start_times)):
        ep_end_time = episode_end_times[i]
        if episode_end_times[i] < episode_start_times[i]:
            print(f"Episode {i} has end time before start time!")
        while episode_success_times[success_time_idx] < episode_start_times[i]:
            if success_time_idx < len(episode_success_times): success_time_idx += 1
        if episode_success_times[success_time_idx] > episode_start_times[i] and episode_success_times[success_time_idx] < episode_end_times[i]:
            ep_end_time = episode_success_times[success_time_idx]
            cntr += 1
            success_time_idx += 1
            if success_time_idx >= len(episode_success_times): success_time_idx -= 1

        episode_ends.append(ep_end_time)

    print(f"Episodes that ended in success: {cntr} out of {len(episode_start_times)}")
    print(f"Success times used up to index: {success_time_idx}")
    return episode_ends

def check_success(hinge_qpos, handle_qpos):
    return hinge_qpos >= 0.4 and handle_qpos <= 0.1

def episode_success_times(env_observations):
    episode_success_times = []
    for i in range(1, env_observations.shape[0]):
        if check_success(env_observations[i, 5], env_observations[i, 4]) and not check_success(env_observations[i-1, 5], env_observations[i-1, 4]):
            episode_success_times.append(env_observations[i, 0])
            
    return episode_success_times

def handle_angle_reached_times(env_observations, target_angle=0.6):
    handle_angle_reached_times = []
    for i in range(1, env_observations.shape[0]):
        if env_observations[i, 3] > target_angle and env_observations[i-1, 3] <= target_angle:
            handle_angle_reached_times.append(i)
    return handle_angle_reached_times

def hinge_angle_reached_times(env_observations, target_angle=0.4):
    hinge_angle_reached_times = []
    for i in range(1, env_observations.shape[0]):
        if env_observations[i, 5] > target_angle and env_observations[i-1, 5] <= target_angle:
            hinge_angle_reached_times.append(i)
    return hinge_angle_reached_times

def calculate_trajectory_length(poses):
    length = 0.0
    for i in range(1, poses.shape[0]):
        length += np.linalg.norm(poses[i] - poses[i-1])
    return length

def interpolate_poses(eef_poses, target_length):
    resampled_poses = []
    for i in range(len(eef_poses)):
        trajectory_length = calculate_trajectory_length(eef_poses[i])
        step_size = trajectory_length / target_length - 0.1

        t = np.linspace(0, eef_poses[i].shape[0], eef_poses[i].shape[0])
        tt = np.linspace(0, eef_poses[i].shape[0], target_length*100)
        xx = np.interp(tt, t, eef_poses[i][:, 0])
        yy = np.interp(tt, t, eef_poses[i][:, 1])
        zz = np.interp(tt, t, eef_poses[i][:, 2])

        

        downsampled_xx = []
        downsampled_yy = []
        downsampled_zz = []
        downsampled_xx.append(xx[0])
        downsampled_yy.append(yy[0])
        downsampled_zz.append(zz[0])

        downsampled_cntr = 0

        for j in range(len(xx)):
            if np.linalg.norm(np.array([xx[j], yy[j], zz[j]]) - np.array([downsampled_xx[downsampled_cntr], downsampled_yy[downsampled_cntr], downsampled_zz[downsampled_cntr]])) < step_size:
                continue

            downsampled_xx.append(xx[j])
            downsampled_yy.append(yy[j])
            downsampled_zz.append(zz[j])
            downsampled_cntr += 1
            # if downsampled_cntr >= target_length - 1:
            #     break

        downsampled_xx = np.array(downsampled_xx)
        downsampled_yy = np.array(downsampled_yy)
        downsampled_zz = np.array(downsampled_zz)
        resampled_poses.append(np.vstack((downsampled_xx, downsampled_yy, downsampled_zz)).T)
    return np.array(resampled_poses)

# def find_episode_success(env_observations):
#     success_indices = []
#     for i in range(1, env_observations.shape[0]):
#         if env_observations[i, 4] >= 1.57 and env_observations[i-1, 4] < 1.57:
#             success_indices.append(i)
#     return success_indices

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

        

    return filtered_poses

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

ep_success_times = episode_success_times(env_observations)
print(f"Time differences between episode start and success: {(ep_success_times[5] - ep_start_times[0])*1e-9}s")
print(f"Time difference between episode end and success: {(ep_end_times[len(ep_end_times)-1] - ep_success_times[5])*1e-9}s")
ep_end_times = find_episode_end_times(ep_start_times, ep_end_times, ep_success_times)

eef_poses_filtered = filter_eef_poses(eef_poses, ep_start_times, ep_end_times)
# print(f"Filtered EEF poses: {eef_poses_filtered.shape}")
print(f"Example episode length: {eef_poses_filtered[0].shape}")
print(f"Example episode first pose: {eef_poses_filtered[0][0]}")
print(f"Max episode length: {max([ep.shape[0] for ep in eef_poses_filtered])}")
print(f"Min episode length: {min([ep.shape[0] for ep in eef_poses_filtered])}")

min_len = min([ep.shape[0] for ep in eef_poses_filtered])
mean_start_pos = np.mean([ep[0] for ep in eef_poses_filtered], axis=0)
print(f"Mean start position: {mean_start_pos}")
eef_positions_resampled = interpolate_poses(eef_poses_filtered, 250)
eef_positions_resampled = eef_positions_resampled[:-1]

# for i in range(len(eef_poses_filtered)):
#     t = np.linspace(0, eef_poses_filtered[i].shape[0], eef_poses_filtered[i].shape[0])
#     tt = np.linspace(0, eef_poses_filtered[i].shape[0], min_len)
#     xx = np.interp(tt, t, eef_poses_filtered[i][:, 0])
#     yy = np.interp(tt, t, eef_poses_filtered[i][:, 1])
#     zz = np.interp(tt, t, eef_poses_filtered[i][:, 2])
#     eef_positions_resampled.append(np.vstack((xx, yy, zz)).T)

# eef_positions_resampled = np.array(eef_positions_resampled)
eef_pos_x_max = np.max(eef_positions_resampled[:, :, 0])
eef_pos_y_max = np.max(eef_positions_resampled[:, :, 1])
eef_pos_z_max = np.max(eef_positions_resampled[:, :, 2])
eef_pos_x_min = np.min(eef_positions_resampled[:, :, 0])
eef_pos_y_min = np.min(eef_positions_resampled[:, :, 1])
eef_pos_z_min = np.min(eef_positions_resampled[:, :, 2])

print(f"EEF position x min: {eef_pos_x_min}, max: {eef_pos_x_max}")
print(f"EEF position y min: {eef_pos_y_min}, max: {eef_pos_y_max}")
print(f"EEF position z min: {eef_pos_z_min}, max: {eef_pos_z_max}")

x_range = eef_pos_x_max - eef_pos_x_min
y_range = eef_pos_y_max - eef_pos_y_min
z_range = eef_pos_z_max - eef_pos_z_min

print(f"EEF position x range: {x_range}")
print(f"EEF position y range: {y_range}")
print(f"EEF position z range: {z_range}")

print(f"Resampled EEF positions: {eef_positions_resampled.shape}")
start_pos = np.mean(eef_positions_resampled[:, 0, :], axis=0)
start_pos_2 = np.array([0.09667707, 0.33416114, 0.26332604])

# cov_matrix = np.cov(eef_positions_resampled[-20:], rowvar=False)
# U, S, Vh = np.linalg.svd(cov_matrix)


mean_trajectory = np.mean(eef_positions_resampled[-20:], axis=0)
trajectory_variance = np.var(eef_positions_resampled[-20:], axis=0)

# mean_trajectory = np.mean(eef_positions_resampled, axis=0)
# trajectory_variance = np.var(eef_positions_resampled, axis=0)

print(f"Trajectory variance shape: {trajectory_variance.shape}")
print(f"Mean trajectory shape: {mean_trajectory.shape}")
print(f"Start position: {start_pos}")

tangent_vectors = []
tangent = calculate_tangent_vector(mean_trajectory[0], mean_trajectory[1])
tangent_vectors.append(tangent)
for i in range(1, mean_trajectory.shape[0]):
    tangent = calculate_tangent_vector(mean_trajectory[i-1], mean_trajectory[i])
    tangent_vectors.append(tangent)
tangent_vectors = np.array(tangent_vectors)
print(f"Tangent vectors shape: {tangent_vectors.shape}")


################################


###############################

x = 1
y = 0
z = 2




surface_x, surface_y, surface_z = fit_ellipsoid_pca(eef_positions_resampled[-1,-20:,:])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(start_pos_2[x], -start_pos_2[y], start_pos_2[z], color='red', s=50, label='Start')
door_handle_pos = np.array([0.12883546,  0.34898176,  0.16951997])
ax.scatter3D(door_handle_pos[x], -door_handle_pos[y], door_handle_pos[z], color='green', s=50, label='Door Handle')
ax.scatter3D(mean_trajectory[-1, x], -mean_trajectory[-1, y], mean_trajectory[-1, z], color='blue', s=50, label='End of Mean Trajectory')

print(f"Door handle - start position distance: {door_handle_pos - start_pos_2}")

colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'teal', 'navy', 'maroon', 'lime', 'coral', 'gold', 'indigo', 'violet']

ax.plot3D(mean_trajectory[:, x], -mean_trajectory[:, y], mean_trajectory[:, z], 'red', linewidth=3, label='Mean Trajectory')
ax.plot3D(mean_trajectory[:, x] + np.sqrt(trajectory_variance[:, x]), -mean_trajectory[:, y]- np.sqrt(trajectory_variance[:, y]), mean_trajectory[:, z]+ np.sqrt(trajectory_variance[:, z]), 'blue', linestyle='dashed', label='Mean + 1 Std Dev (X)')

# ax.fill_between(mean_trajectory[:, x], -mean_trajectory[:,y], mean_trajectory[:,z], mean_trajectory[:, x] + np.sqrt(trajectory_variance[:, x]), -(mean_trajectory[:, y] + np.sqrt(trajectory_variance[:,y])), mean_trajectory[:,z] + np.sqrt(trajectory_variance[:,z]), color='green', alpha=0.2, label='Trajectory Variance')
# ax.fill_between(mean_trajectory[:, x] - np.sqrt(trajectory_variance[:, x]), 
#                 -(mean_trajectory[:, y] - np.sqrt(trajectory_variance[:, y])),
#                 mean_trajectory[:, z] - np.sqrt(trajectory_variance[:, z]),
#                 mean_trajectory[:, x] + np.sqrt(trajectory_variance[:, x]),
#                 -(mean_trajectory[:, y] + np.sqrt(trajectory_variance[:, y])),
#                 mean_trajectory[:, z] + np.sqrt(trajectory_variance[:, z]),
#                 color='green', alpha=0.2, label='Trajectory Variance')
# ax.fill_between(mean_trajectory[:, x] + np.sqrt(trajectory_variance[:, x]), 
#                 -(mean_trajectory[:, y] - np.sqrt(trajectory_variance[:, y])),
#                 mean_trajectory[:, z] - np.sqrt(trajectory_variance[:, z]),
#                 mean_trajectory[:, x] - np.sqrt(trajectory_variance[:, x]),
#                 -(mean_trajectory[:, y] + np.sqrt(trajectory_variance[:, y])),
#                 mean_trajectory[:, z] + np.sqrt(trajectory_variance[:, z]),
#                 color='green', alpha=0.2)
# ax.fill_between(mean_trajectory[:, x] - np.sqrt(trajectory_variance[:, x]), 
#                 -(mean_trajectory[:, y] - np.sqrt(trajectory_variance[:, y])),
#                 mean_trajectory[:, z] + np.sqrt(trajectory_variance[:, z]),
#                 mean_trajectory[:, x] + np.sqrt(trajectory_variance[:, x]),
#                 -(mean_trajectory[:, y] + np.sqrt(trajectory_variance[:, y])),
#                 mean_trajectory[:, z] - np.sqrt(trajectory_variance[:, z]),
#                 color='green', alpha=0.2)
# ax.fill_between(mean_trajectory[:, x] - np.sqrt(trajectory_variance[:, x]), 
#                 -(mean_trajectory[:, y] + np.sqrt(trajectory_variance[:, y])),
#                 mean_trajectory[:, z] - np.sqrt(trajectory_variance[:, z]),
#                 mean_trajectory[:, x] + np.sqrt(trajectory_variance[:, x]),
#                 -(mean_trajectory[:, y] - np.sqrt(trajectory_variance[:, y])),
#                 mean_trajectory[:, z] + np.sqrt(trajectory_variance[:, z]),
#                 color='green', alpha=0.2)

circle_points_list = []
radii = []
for i in range(len(mean_trajectory)):
    new_radii = []
    radius = np.mean(np.sqrt(trajectory_variance[i]))
    circle_points = generate_circle_points(mean_trajectory[i], tangent_vectors[i-1] if i > 0 else tangent_vectors[0], radius, num_points=20)
    circle_points_list.append(circle_points)
    for _ in range(len(circle_points)):
        new_radii.append(radius)

    radii.append(new_radii)
#concatenate all circle points into a single array for surface plotting
circle_points_array = np.array(circle_points_list)

# combine first and second dimension of circle_points_array
# circle_points_array = circle_points_array.reshape(-1, circle_points_array.shape[2])
# x_mesh, y_mesh, z_mesh = np.meshgrid(circle_points_array[:, x], circle_points_array[:, y], circle_points_array[:, z])

radii_array = np.array(radii)

print(f"Circle points array shape: {circle_points_array.shape}")
print(f"Radii array shape: {radii_array.shape}")
# print(f"mesh grid shapes: x: {x_mesh.shape}, y: {y_mesh.shape}, z: {z_mesh.shape}")

my_col = cm.jet((radii_array - np.min(radii_array)) / (np.max(radii_array) - np.min(radii_array)))

# plot surface with colormap
ax.plot_surface(circle_points_array[:, :, x], -circle_points_array[:, :, y], circle_points_array[:, :, z], facecolors=my_col, linewidth=0, alpha=0.5, label='Trajectory Variance Ellipses')
# ax.plot_surface(x_mesh, -y_mesh, z_mesh, cmap=cm.jet, alpha=0.2, label='Trajectory Variance Surface')
    # if i > 0:
    #     ax.fill_between(prev_circle_points[:, x], -prev_circle_points[:, y], prev_circle_points[:, z],
    #                     circle_points[:, x], -circle_points[:, y], circle_points[:, z],
    #                     color='green', alpha=0.2)
    
    # prev_circle_points = circle_points
    # ax.plot3D(circle_points[:, x], -circle_points[:, y], circle_points[:, z], color='green', alpha=0.5)


    # surface_x_2, surface_y_2, surface_z_2 = surface_of_ellipsoid(mean_trajectory[i], trajectory_variance[i])
    # ax.plot_surface(surface_y_2, -surface_x_2, surface_z_2, color='cyan', alpha=0.3, label='PCA Ellipsoid Fit')

# ax.plot_surface(mean_trajectory[:, x], -mean_trajectory[:, y], mean_trajectory[:, z], color='red', alpha=0.3)

# ax.fill_between(mean_trajectory[:, x] - np.sqrt(trajectory_variance[:, x]), 
#                 -mean_trajectory[:, y],
#                 -mean_trajectory[:, y] + np.sqrt(trajectory_variance[:, x]),
#                 color='red', alpha=0.2, label='Trajectory Variance (X)')
# ax.fill_between(mean_trajectory[:, x], -mean_trajectory[:, y],
#                 -mean_trajectory[:, y] + np.sqrt(trajectory_variance[:, z]),
#                 color='red', alpha=0.2, label='Trajectory Variance (Z)')

# for i in range(20):
#     ax.plot3D(eef_positions_resampled[-i][:, x], -eef_positions_resampled[-i][:, y], eef_positions_resampled[-i][:, z], colors[i%len(colors)])
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
ax.set_xlim([0.1, 0.45])
ax.set_ylim([-0.275, 0.075])
ax.set_zlim([-0.05, 0.3])

ax.view_init(elev=0, azim=-90)
plt.show()


# fig2 = go.Figure(data=go.Streamtube(
#     x=mean_trajectory[:, x],
#     y=-mean_trajectory[:, y],
#     z=mean_trajectory[:, z],
#     u=np.gradient(trajectory_variance[:, x]),
#     v=-np.gradient(trajectory_variance[:, y]),
#     w=np.gradient(trajectory_variance[:, z]),
# ))

# fig2.show()

# fig2 = plt.figure()
# ax2 = plt.axes(projection='3d')

# ax2.set_title('3D EEF Pose Trajectory for last Episode')
# ax2.set_xlabel('X Position (m)')
# ax2.set_ylabel('Y Position (m)')
# ax2.set_zlabel('Z Position (m)') 
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def tubeplot_py(curve, r, n=8, ct=None):
#     """
#     Constructs a tube, or warped cylinder, along a 3D curve.

#     This function is a direct translation of the original MATLAB 'tubeplot' 
#     function by Janus H. Wesenberg.

#     Args:
#         curve (np.ndarray): [3, N] vector of curve data (x, y, z coordinates).
#         r (float): The radius of the tube.
#         n (int, optional): Number of points to use on circumference. Defaults to 8.
#         ct (float, optional): Threshold for collapsing close points. Defaults to 0.5 * r.

#     Returns:
#         tuple: (X, Y, Z) meshgrid arrays for the tube surface (if requested).
#                If no output arguments are requested, the tube is plotted.
    
#     Notes:
#         The algorithm fails if you have bends beyond 90 degrees.
#     """
    
#     # --- Input Validation and Default Setup ---
    
#     # Check minimum arguments (r is required)
#     if r is None:
#         raise ValueError('Give at least curve and radius (r).')

#     # Check curve shape
#     if curve.shape[0] != 3:
#         raise ValueError('Malformed curve: should be a numpy array of shape [3, N].')
        
#     # Default threshold for collapsing points
#     if ct is None:
#         ct = 0.5 * r

#     # --- 1. Collapse Close Points ---
    
#     # Initialize the cleaned curve and point counter
#     n_original_points = curve.shape[1]
#     cleaned_curve = np.zeros_like(curve)
#     npoints = 0
    
#     # Always include the first point
#     npoints += 1
#     cleaned_curve[:, npoints - 1] = curve[:, 0]
    
#     # Collapse points within ct distance of the last added point
#     for k in range(1, n_original_points - 1):
#         if np.linalg.norm(curve[:, k] - cleaned_curve[:, npoints - 1]) > ct:
#             npoints += 1
#             cleaned_curve[:, npoints - 1] = curve[:, k]
            
#     # Always include the endpoint
#     if np.linalg.norm(curve[:, n_original_points - 1] - cleaned_curve[:, npoints - 1]) > 0:
#         npoints += 1
#         cleaned_curve[:, npoints - 1] = curve[:, n_original_points - 1]

#     # Trim the cleaned_curve to the actual number of points
#     curve = cleaned_curve[:, :npoints]
    
#     # --- 2. Calculate Direction Vectors (dv) ---
    
#     # dv: average for internal points. first stretch for endpoints.
#     # MATLAB: curve(:,[2:end,end])-curve(:,[1,1:end-1])
#     dv = curve[:, np.r_[1:npoints, npoints - 1]] - curve[:, np.r_[0, 0:npoints - 1]]
    
#     # --- 3. Initialize Normal Vector (nvec) ---
    
#     # make nvec not parallel to dv[:, 0]
#     nvec = np.zeros(3)
#     # Get index of minimum absolute value in the first direction vector
#     idx = np.argmin(np.abs(dv[:, 0])) 
#     nvec[idx] = 1 # Set the corresponding component to 1
    
#     # --- 4. Precalculate Circumference Factors ---
    
#     # Angles for the circle circumference
#     angles = np.linspace(0, 2 * np.pi, n + 1)
    
#     # cfact and sfact are [3, n+1] arrays (replicated for easy vector ops)
#     cfact = np.tile(np.cos(angles), (3, 1))
#     sfact = np.tile(np.sin(angles), (3, 1))

#     # Initialize the result array: [3, n+1 (circumference), npoints + 2 (caps)]
#     xyz = np.zeros((3, n + 1, npoints + 2))
    
#     # --- 5. Main Loop: Propagate the Local Frame (nvec, convec) ---
    
#     # Start at k=1 (index 0 in Python)
#     for k in range(npoints):
#         # convec = cross(nvec, dv[:, k])
#         convec = np.cross(nvec, dv[:, k])
        
#         # Normalize convec
#         norm_convec = np.linalg.norm(convec)
#         if norm_convec == 0:
#              # Handle collinear case (straight segment, nvec || dv) by skipping update
#              # or by choosing a new, arbitrary, perpendicular nvec.
#              # In a direct translation, we propagate the old frame:
#              if k > 0:
#                 convec = xyz[:, :, k][:, 0] - curve[:, k-1] # Use previous convec if possible
#                 nvec = xyz[:, :, k][:, 1] - curve[:, k-1] # Use previous nvec if possible
#              else:
#                 # If it happens at the start, just use the arbitrary perpendicular vector
#                 convec = np.cross(dv[:, k], nvec)
             
#         else:
#             convec = convec / norm_convec

#         # nvec = cross(dv[:, k], convec) - This updates nvec to be orthonormal to dv and convec
#         nvec = np.cross(dv[:, k], convec)
        
#         # Normalize nvec
#         norm_nvec = np.linalg.norm(nvec)
#         if norm_nvec == 0:
#             # Should not happen if dv and convec are orthogonal and non-zero.
#             # Use a safe fallback if the previous block failed.
#             nvec = np.array([1, 0, 0]) # Arbitrary vector
#         else:
#             nvec = nvec / norm_nvec
            
#         # Update xyz at the current point k (stored at k+1 in the array due to caps)
#         # xyz(:,:,k+1) = repmat(curve(:,k),[1,n+1]) + cfact.*repmat(r*nvec,[1,n+1]) + sfact.*repmat(r*convec,[1,n+1]);
        
#         # curve[:,k] is the center point
#         center = np.tile(curve[:, k].reshape(3, 1), (1, n + 1))
        
#         # r*nvec is the major axis component (x-axis of the circle)
#         n_comp = cfact * np.tile(r * nvec.reshape(3, 1), (1, n + 1))
        
#         # r*convec is the minor axis component (y-axis of the circle)
#         c_comp = sfact * np.tile(r * convec.reshape(3, 1), (1, n + 1))
        
#         xyz[:, :, k + 1] = center + n_comp + c_comp

#     # --- 6. Cap the Ends ---
    
#     # xyz(:,:,1) = repmat(curve(:,1),[1,n+1]); (First cap)
#     xyz[:, :, 0] = np.tile(curve[:, 0].reshape(3, 1), (1, n + 1))
    
#     # xyz(:,:,end) = repmat(curve(:,end),[1,n+1]); (Last cap)
#     xyz[:, :, -1] = np.tile(curve[:, -1].reshape(3, 1), (1, n + 1))

#     # --- 7. Extract Results ---
    
#     # Squeeze to remove the leading '3' dimension from the first axis, 
#     # resulting in arrays of shape (n+1, npoints+2).
#     X = np.squeeze(xyz[0, :, :])
#     Y = np.squeeze(xyz[1, :, :])
#     Z = np.squeeze(xyz[2, :, :])
    
#     # --- 8. Plot or Return ---
    
#     if len(plt.get_fignums()) == 0 or plt.gcf().canvas.manager.window.isVisible() == False:
#         # Create a new figure if none is active or visible
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#     else:
#         # Use existing figure
#         fig = plt.gcf()
#         # Find the 3D axes, or create one if not found (Matplotlib's behavior is complex here)
#         ax = fig.gca(projection='3d')
#         if ax.name != '3d':
#              ax = fig.add_subplot(111, projection='3d') # If gca is 2D, create 3D

#     ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, rstride=1, cstride=1, linewidth=0.1, antialiased=False)
#     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#     ax.set_title('Tubeplot Visualization')
    
#     # Return X, Y, Z only if output arguments are explicitly requested
#     if len(plt.gcf().canvas.manager.callstack) > 0 and 'plot_surface' not in str(plt.gcf().canvas.manager.callstack[-1]):
#         return X, Y, Z
#     else:
#         plt.show() # Show the plot if no explicit return was requested
#         return None # Explicitly return None if plotting was the action

# # --- Example Usage (Same as MATLAB's example) ---

# # 1. Generate curve data
# t = np.linspace(0, 2 * np.pi, 50)
# curve_data = np.array([
#     np.cos(t),
#     np.sin(t),
#     0.2 * (t - np.pi)**2
# ])

# # 2. Call the function (will plot by default)
# tubeplot_py(curve_data, r=0.1)

# 3. Example of requesting output data
# X, Y, Z = tubeplot_py(curve_data, r=0.1, n=16, ct=0.01)
# print(f"Surface mesh data arrays returned: X shape={X.shape}, Y shape={Y.shape}, Z shape={Z.shape}")
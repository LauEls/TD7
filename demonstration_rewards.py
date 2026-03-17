import numpy as np
import os

# epsiode_length = 130
# demo_file_name = "gh360_door_demonstration_v8.npy"
epsiode_length = 500
demo_file_name = "gh360_sim_door_demonstration_with_variance_v1.npy"

demo_paths = np.load(os.path.join("demonstrations/",demo_file_name), allow_pickle=True)

episode_rewards = []

for path in demo_paths:
    episode_reward = path["rewards"].sum()
    episode_rewards.append(episode_reward)
    print(episode_reward)

episode_rewards = np.array(episode_rewards[0:20])
print(f"Size of episode rewards: {episode_rewards.shape}")
print("Mean episode reward: ", episode_rewards.mean()/epsiode_length)
print("Standard deviation of episode rewards: ", episode_rewards.std()/epsiode_length)
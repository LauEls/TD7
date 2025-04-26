import json
import gym
import numpy as np
import gh360_gym
import random

if __name__ == "__main__":
    config_file = 'runs/door/real_gh360/eef_vel/online/v8_corl_with_demos/variant.json'
    demo_file_path = "demonstrations/gh360_door_demonstration_v8.npy"

    try:
        with open(config_file) as f:
            variant = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
            "Please check filepath and try again.".format(config_file))
        
    env_config = variant["environment_kwargs"]
    env_name = variant["environment_kwargs"].pop("env_name")
    # variant["environment_kwargs"].pop("input_max")
    # variant["environment_kwargs"].pop("input_min")
    #variant["environment_kwargs"].pop("max_joint_pos")
    #variant["environment_kwargs"].pop("min_joint_pos")
    # variant["environment_kwargs"].pop("max_current")

    env = gym.make('gh360_gym/'+env_name, **env_config)
    env.reset()

    ep_length = variant["episode_length"]

    demos = np.load(demo_file_path, allow_pickle=True)

    # for demo in demos:

    #     for action in demo["actions"]:
    #         env.step(action)

    #     env.reset()

    
    obs, info = env.reset()
    
    total_reward = np.zeros(10)
    for j in range(10):
        obs, info = env.reset()
        obs, info = env.special_reset(j)
        demo = random.choice(demos)

        # total_reward = 0
        for action in demo["actions"]:
            
            # obs, reward, done, _ = env.step(action)
            obs, reward, done, _ = env.step(action)
            total_reward[j] += reward

        print(f"Episode {j} reward: {total_reward[j]:.3f}")
        if reward == 1:
            print(f"Episode {j} successful")

        # print("Total Reward: ", total_reward)
    env.reset()

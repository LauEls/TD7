import json
import gym
import numpy as np
import gh360_gym

if __name__ == "__main__":
    config_file = '/home/gh360/TD7/runs/door/real_gh360/eef_vel/online/v2_constraint_demo/variant.json'
    demo_file_path = "/home/gh360/ros2_gh360_ws/src/gh360/gh360_demonstration/data/spacemouse_demonstrations/door/gh360_door_demonstration_v6.npy"

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

    for demo in demos:
        for action in demo["actions"]:
            env.step(action)

        env.reset()

import json
import gym
import numpy as np
import random
import robosuite as suite
from robosuite.utils.input_utils import *
from robosuite.wrappers import GymWrapper
from util import NormalizedBoxEnv

if __name__ == "__main__":
    config_file = "runs/door_mirror/gh360/osc_pose/online/20_demos/variant.json"
    demo_file_path = "demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy"

    try:
        with open(config_file) as f:
            variant = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
            "Please check filepath and try again.".format(config_file))
        
    env_config = variant["environment_kwargs"]

    controller = env_config.pop("controller")
    if controller in set(suite.ALL_CONTROLLERS):
        print("Controller: "+controller)
        # This is a default controller
        controller_config = suite.load_controller_config(default_controller=controller)
        
        if "controller_config" in env_config.keys():
            controller_settings = env_config.pop("controller_config")
            for config in controller_settings:
                controller_config[config] = controller_settings[config]
    else:
        # This is a string to the custom controller
        controller_config = suite.load_controller_config(custom_fpath=controller)

    suite_env = suite.make(**env_config,
                    #  has_renderer=variant["render"],
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_object_obs=True,
                    use_camera_obs=False,
                    controller_configs=controller_config,
                    )
    env = NormalizedBoxEnv(GymWrapper(suite_env))
    env.reset()
    env.render()

    demos = np.load(demo_file_path, allow_pickle=True)
    demos = demos[0:20]

    
    obs = env.reset()
    
    total_reward = np.zeros(10)
    for j in range(10):
        obs = env.reset()
        demo = random.choice(demos)

        for action in demo["actions"]:
            obs, reward, done, _ = env.step(action)
            total_reward[j] += reward
            env.render()

        print(f"Episode {j} reward: {total_reward[j]:.3f}")
        if reward == 1:
            print(f"Episode {j} successful")

    env.reset()
    env.render()

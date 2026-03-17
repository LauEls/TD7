import TD7
import os
import json
import gym

import numpy as np
import argparse
import torch
import time
import signal
import subprocess

import robosuite as suite
from robosuite.wrappers import GymWrapper
from util import NormalizedBoxEnv

def rollout(eval_env, RL_agent, ep_length=500):
    try:
        while True:
            state, info = eval_env.reset()
            
            cntr = 0
            total_reward = 0
            while cntr < ep_length:
                action = RL_agent.select_action(np.array(state), use_exploration=False)
                state, reward, done, _ = eval_env.step(action)
                # eval_env.render()
                total_reward += reward
                cntr += 1
            
            print(f"Episode reward: {total_reward:.3f}")
            if reward == 1:
                print(f"Episode successful")
        
    except KeyboardInterrupt:
        eval_env.reset()
    


if __name__ == "__main__":
    load_dir = "runs/door/real_gh360/eef_vel/online/v16_erf"

    kwargs_fpath = os.path.join(load_dir, "variant.json")
    try:
        with open(kwargs_fpath) as f:
            variant = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
            "Please check filepath and try again.".format(kwargs_fpath))
        
    env_config = variant["environment_kwargs"]
    env_name = variant["environment_kwargs"].pop("env_name")
    ep_length = variant["episode_length"]

    env = gym.make('gh360_gym/'+env_name, **env_config)

    hp = TD7.Hyperparameters(**variant["hyperparameters"])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    result_path = os.path.join(load_dir, "run_0")
    hp.dir_path = result_path
    hp.continue_learning = True
    RL_agent = TD7.Agent(state_dim, action_dim, max_action, hp=hp)

    rollout(env, RL_agent, ep_length)
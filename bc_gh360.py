import numpy as np
import random
import gym
import gh360_gym
import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

#from mujoco_py import MjSim
import robosuite as suite
from robosuite.utils.input_utils import *
import sys
from robosuite.wrappers import GymWrapper
# sys.path.insert(0, '/home/laurenz/phd_project/sac/sac_2')
# from wrappers import NormalizedBoxEnv

def relu_weights_normal_init(tensor):
    size = tensor.size()[0]
    std = np.sqrt(2/size)
    return tensor.data.normal_(0.0,std)


class policy_network(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        hidden_activation = F.relu,
        init_w=1e-3,
        reparam_noise=1e-6,
    ):
        super(policy_network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.reparam_noise = reparam_noise
        self.dist = None
        # self.deterministic = deterministic

        self.fcs = nn.ModuleList()
        layer_input_size = self.input_size
        for layer_output_size in hidden_sizes:
            fc = nn.Linear(layer_input_size, layer_output_size)
            relu_weights_normal_init(fc.weight)
            fc.bias.data.fill_(0)
            self.fcs.append(fc)
            layer_input_size = layer_output_size

        self.fc_mean = nn.Linear(hidden_sizes[-1],output_size)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.fill_(0)

        self.fc_log_std = nn.Linear(hidden_sizes[-1], output_size)
        # self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        # self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        self.fc_log_std.weight.data.fill_(-10.0)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)
        self.to(self.device)

        self.cntr = 0
        

    def forward(self, obs):
        o = obs
        for fc in self.fcs:
            o = self.hidden_activation(fc(o))
        mean = self.fc_mean(o)

        return mean

    def rsample_and_logprob(self, obs, reparam=True, deterministic=False):
        return self.forward(obs)

    def get_action(self, obs_np, deterministic=False):
        obs = obs_np[None]
        obs = torch.from_numpy(obs).float().to(self.device)
        # actions, _ = self.rsample_and_logprob(obs, reparam=False, deterministic=deterministic)
        actions = self.rsample_and_logprob(obs, reparam=False, deterministic=deterministic)
        # actions_np = actions.cpu().detach().numpy()[0]
        return actions[0]
    
    def save_checkpoint(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path):
        self.load_state_dict(torch.load(file_path))
    


if __name__ == "__main__":
    seed=random.randint(0, 100000)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    training = True

    # load_dir = "/home/laurenz/phd_project/TD7/runs/door_mirror/gh360/joint_velocity/online/v6_cont_after_offline/"
    config_file = '/home/gh360/TD7/runs/door/real_gh360/eef_vel/online/v2_constraint_demo/variant.json'
    # load_dir = "../../sac_2/runs/data/stack/panda/osc_pose/with_demo/v4_obs_opt/"

    # demo_file_path = "/home/laurenz/phd_project/TD7/demonstrations/robosuite_door_mirror_demonstration_v3_expert.npy"
    demo_file_path = "/home/gh360/ros2_gh360_ws/src/gh360/gh360_demonstration/data/spacemouse_demonstrations/door/gh360_door_demonstration_v6.npy"
    # demo_file_path = "/home/laurenz/phd_project/sac/scripts/robosuite_env_solved/demonstrations/stack/stack_random_no_noise_test_v2.npy"
    

    try:
        with open(config_file) as f:
            variant = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
            "Please check filepath and try again.".format(config_file))
        
    env_config = variant["environment_kwargs"]
    env_name = variant["environment_kwargs"].pop("env_name")
    # variant["environment_kwargs"].pop("max_joint_pos")
    # variant["environment_kwargs"].pop("min_joint_pos")
    # variant["environment_kwargs"].pop("input_max")
    # variant["environment_kwargs"].pop("input_min")
    # variant["environment_kwargs"].pop("max_motor_current")
    # variant["environment_kwargs"].pop("min_motor_current")

    env = gym.make('gh360_gym/'+env_name, **env_config)
    # env = NormalizedBoxEnv(raw_env)
    env.reset()

    ep_length = variant["episode_length"]

    demos = np.load(demo_file_path, allow_pickle=True)

    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    h_layer = []
    # for i in variant["policy_kwargs"]["hidden_sizes"]:
    #     h_layer.append(i)

    for i in range(2):
        h_layer.append(256)

    policy = policy_network(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=h_layer,
    )

    t_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    losses = []
    
    if training:
        cntr = 0
        for t in range(80000):
            i = int(np.random.uniform(0, len(demos)))
            # print("i: ",i)
            obs = demos[i]["observations"]
            # print(f"obs: {obs}")
            actions = demos[i]["actions"]
            actions = torch.from_numpy(actions).float().to(t_device)

            action_pred = policy.get_action(obs)

            loss = criterion(action_pred, actions)
            losses.append(loss.item())
            if loss.item() < 0.01:
                cntr += 1
                if cntr > 500:
                    break
            else:
                cntr = 0
            if t % 2000 == 1999:
                print(t, np.mean(losses))
                losses = []

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        policy.save_checkpoint("policy")
    else:
        policy.load_checkpoint("policy")

    obs, info = env.reset()
    total_reward = 0
    for i in range(ep_length):
        action = policy.get_action(obs)
        action = action.cpu().detach().numpy()
        print("Action: ", action)
        print("Action Shape: ", action.shape)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print("Total Reward: ", total_reward)
    env.reset()
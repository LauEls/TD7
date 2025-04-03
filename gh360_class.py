import TD7
import os
import json
import gym
import gh360_gym
import numpy as np
import argparse
import torch
import time

class RL_GH360:
    def __init__(self, env_name, exp_runs, config_file_path):
        load_dir = config_file_path
        self.exp_run = 0

        kwargs_fpath = os.path.join(load_dir, "variant.json")
        try:
            with open(kwargs_fpath) as f:
                variant = json.load(f)
        except FileNotFoundError:
            print("Error opening default controller filepath at: {}. "
                "Please check filepath and try again.".format(kwargs_fpath))
            
        state_file = os.path.join(load_dir, "state.json")
        try:
            with open(state_file) as f:
                state = json.load(f)
        except FileNotFoundError:
            print("Error opening default controller filepath at: {}. "
                "Please check filepath and try again.".format(state_file))
            
        env_config = variant["environment_kwargs"]
        env_name = variant["environment_kwargs"].pop("env_name")
        seed = variant["seed"]
        self.offline = variant["offline"]

        self.env = gym.make('gh360_gym/'+env_name, **env_config)
        self.eval_env = self.env

        self.use_checkpoints = True
        self.ep_length = variant["episode_length"]
        self.timesteps_before_training = self.ep_length*50
        self.eval_freq = self.ep_length*10
        self.max_timesteps = self.ep_length*10000
        self.init_buffer_paths = variant["init_buffer_paths"]
        self.eval_eps = 1

        if self.offline:
            expert_paths = np.load(os.path.join("demonstrations/",variant["demo_file_name"]), allow_pickle=True)
            print("Expert paths shape: ", expert_paths.shape)
            random_paths_file = os.path.join("demonstrations/",variant["demo_file_name"])
            random_paths_file = random_paths_file[0:-4] + "_random_paths" + random_paths_file[-4:]
            print(random_paths_file)
            random_paths = np.load(random_paths_file, allow_pickle=True)
            print("Random paths shape: ", random_paths.shape)
            paths = np.concatenate((expert_paths, random_paths))
            self.use_checkpoints = False
            self.eval_during_training = variant["eval_during_training"]


        self.result_path = os.path.join(load_dir, "run_"+str(self.exp_run))

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        

        

        print("---------------------------------------")
        print(f"Algorithm: TD7, Env: {env_name}, Seed: {seed}")
        print("---------------------------------------")

        self.env.seed(seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.offline:
            # print("Expert paths: ", expert_paths[0]['actions'].shape)
            # print("Random paths: ", random_paths[0]['actions'].shape)
            # print("Paths: ", paths[0]['observations'].shape)
            # print("Paths: ", paths[30]['observations'].shape)
            state_dim = paths[0]['observations'].shape[1]
            action_dim = paths[0]['actions'].shape[1]
            max_action = 0.0
            for path in paths:
                new_max_action = np.max(np.abs(path['actions']))
                if new_max_action > max_action: max_action = new_max_action
            # print("Max action: ", max_action)
        else:
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.shape[0] 
            max_action = float(self.env.action_space.high[0])

        print("State dim: ", state_dim)
        print("Action dim: ", action_dim)
        print("Max action: ", max_action)
        hp = TD7.Hyperparameters(**variant["hyperparameters"])
        hp.dir_path = self.result_path
        
        self.RL_agent = TD7.Agent(state_dim, action_dim, max_action, offline=self.offline, hp=hp)

    def start_learning(self):
        if self.offline:
            self.train_offline()
        else:
            self.train_online()


    def train_online(self, stop_event=None):
        t = 0
        if self.RL_agent.continue_learning:
            buffer_paths = np.load(self.result_path+"/buffer_paths.npy", allow_pickle=True)
            self.RL_agent.replay_buffer.load_paths(buffer_paths)
        elif self.init_buffer_paths:
            buffer_paths = np.load(self.result_path+"/init_buffer_paths.npy", allow_pickle=True)
            self.RL_agent.replay_buffer.load_paths(buffer_paths)
            print(f"Loaded initial buffer paths of length: {len(buffer_paths[0]['observations'])}")
            print(f"Buffer size: {self.RL_agent.replay_buffer.size}")
            t = self.RL_agent.replay_buffer.size-1
            

        self.evals = []
        start_time = time.time()
        allow_train = False

        state, ep_finished = self.env.reset(), False
        ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

        # for i in range(int(args.max_timesteps+1)):
        while t < int(self.max_timesteps+1):
            self.maybe_evaluate_and_print(t, start_time)
            
            if allow_train:
                action = self.RL_agent.select_action(np.array(state))
            else:
                action = self.env.action_space.sample()

            next_state, reward, ep_finished, _ = self.env.step(action) 
            
            
            ep_total_reward += reward
            ep_timesteps += 1

            if ep_timesteps >= self.ep_length: 
                ep_finished = 1
                self.env.step(np.zeros(self.RL_agent.action_dim))
            # done = float(ep_finished) if ep_timesteps < 500 else 0
            done = ep_finished
            
            self.RL_agent.replay_buffer.add(state, action, next_state, reward, done)

            state = next_state

            if allow_train and not self.use_checkpoints:
                self.RL_agent.train()

            if ep_finished: 
                print(f"Reward: {ep_total_reward}")
                print(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")

                if allow_train and self.use_checkpoints:
                    self.RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

                if t >= self.timesteps_before_training:
                    allow_train = True
                    if t >= self.timesteps_before_training and t <= (self.timesteps_before_training+ep_timesteps):
                        print("Saving initial buffer paths")
                        self.RL_agent.replay_buffer.save_paths(self.result_path+"/init_buffer_paths.npy")

                state, done = self.env.reset(), False
                ep_total_reward, ep_timesteps = 0, 0
                ep_num += 1 
                
                if stop_event.is_set():
                    self.save_training_state()
                    return

            t += 1

        # Save final model
        self.RL_agent.save_model(self.result_path)
        self.RL_agent.replay_buffer.save_paths(os.path.join(self.result_path,"buffer_paths.npy"))
        self.RL_agent.replay_buffer.save_priority(os.path.join(self.result_path, "priority.npy"))
        self.RL_agent.save_class_variables(self.result_path)
    
    def train_offline(self):
        pass

    def maybe_evaluate_and_print(self, t, start_time):
        if t % self.eval_freq == 0 and t > 0:
            print("---------------------------------------")
            print(f"Evaluation at {t} time steps")
            print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

            total_reward = np.zeros(self.eval_eps)
            for ep in range(self.eval_eps):
                state, done = self.eval_env.reset(), False
                cntr = 0
                while not done and cntr < self.ep_length:
                    action = self.RL_agent.select_action(np.array(state), self.use_checkpoints, use_exploration=False)
                    state, reward, done, _ = self.eval_env.step(action)
                    total_reward[ep] += reward
                    cntr += 1
                            
            print(f"Average total reward over {self.eval_eps} episodes: {total_reward.mean():.3f}")

            print("---------------------------------------")

            self.evals.append(total_reward)
            # np.save(f"./results/{args.file_name}", evals)
            np.save(os.path.join(self.result_path,"results.npy"), self.evals)

            self.eval_env.reset()

    def save_training_state(self):
        self.RL_agent.save_model(self.result_path)
        self.RL_agent.replay_buffer.save_paths(os.path.join(self.result_path,"buffer_paths.npy"))
        self.RL_agent.replay_buffer.save_priority(os.path.join(self.result_path, "priority.npy"))
        self.RL_agent.save_class_variables(self.result_path)
        pass
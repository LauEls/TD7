import TD7
import os
import json
import gym
import gh360_gym
import numpy as np
import argparse
import torch
import time
import signal
import subprocess

class RL_GH360:
    def __init__(self, config_file_path):
        self.path_ros2_ws = "~/ros2_gh360_ws"
        self.load_dir = config_file_path
        self.exp_run = 0

        kwargs_fpath = os.path.join(self.load_dir, "variant.json")
        try:
            with open(kwargs_fpath) as f:
                variant = json.load(f)
        except FileNotFoundError:
            print("Error opening default controller filepath at: {}. "
                "Please check filepath and try again.".format(kwargs_fpath))
            
        # state_file = os.path.join(self.load_dir, "state.json")
        # try:
        #     with open(state_file) as f:
        #         state = json.load(f)
        # except FileNotFoundError:
        #     print("Error opening default controller filepath at: {}. "
        #         "Please check filepath and try again.".format(state_file))
            
        env_config = variant["environment_kwargs"]
        self.env_name = variant["environment_kwargs"].pop("env_name")
        self.original_seed = variant["seed"]
        self.offline = variant["offline"]
        self.demo_buffer = variant["demo_buffer"]
        self.max_experiment_runs = variant["max_experiment_runs"]
        if self.demo_buffer:
            self.demo_file_name = variant["demo_file_name"]
            self.demo_episodes = variant["demo_episodes"]

        self.env = gym.make('gh360_gym/'+self.env_name, **env_config)
        self.eval_env = self.env

        self.use_checkpoints = False
        self.ep_length = variant["episode_length"]
        
        self.eval_freq = self.ep_length*10
        self.max_timesteps = self.ep_length*variant["max_episodes"] #10000
        self.init_buffer_paths = variant["init_buffer_paths"]
        self.eval_eps = variant["eval_episodes"]

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

        self.hp = TD7.Hyperparameters(**variant["hyperparameters"])

        


    def init_run(self, exp_run=-1):
        print(f"exp_run: {exp_run}")
        if exp_run == -1:
            self.continue_training = self.load_training_state()
        else:
            self.continue_training = True 
            self.exp_run = exp_run
            self.t = 130
        self.result_path = os.path.join(self.load_dir, "run_"+str(self.exp_run))
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not self.continue_training: 
            self.t = 0
            self.evals = []
        else:
            if self.t > 0:
                self.evals = np.load(os.path.join(self.result_path,"results.npy"), allow_pickle=True).tolist()
            else:
                self.evals = []
        
        
        self.timesteps_before_training = self.ep_length*50
        
        seed = self.original_seed + self.exp_run
        print("---------------------------------------")
        print(f"Algorithm: TD7, Env: {self.env_name}, Seed: {seed}")
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
        
        self.hp.dir_path = self.result_path
        if self.continue_training and self.t > 0:
            self.hp.continue_learning = True
        else: self.hp.continue_learning = False
        
        self.RL_agent = TD7.Agent(state_dim, action_dim, max_action, demo_buffer=self.demo_buffer, offline=self.offline, hp=self.hp)
        if self.demo_buffer:
            paths = np.load(os.path.join("demonstrations/",self.demo_file_name), allow_pickle=True)
            demo_episodes = self.demo_episodes
            if demo_episodes >= len(paths): demo_episodes = len(paths)
            else:
                i_plus = int(len(paths)/demo_episodes)
                reduced_paths = []
                i = 0
                while i < len(paths):
                    reduced_paths.append(paths[i])
                    i += i_plus
                    print("i: ", i)
                paths = reduced_paths
                
            print(f"demonstraition episodes: {len(paths)}")
            self.RL_agent.demo_buffer.load_paths(paths)
            print(f"demo buffer size: {self.RL_agent.demo_buffer.size}")
            # self.timesteps_before_training -= self.RL_agent.demo_buffer.size
            # if self.timesteps_before_training < 256: self.timesteps_before_training = 256
            self.timesteps_before_training = 256

    def start_learning(self, stop_event=None):
        self.init_run()

        if self.offline:
            self.train_offline()
        else:
            experiment_finished = False
            while not experiment_finished:
                print(f"exp_run: {self.exp_run}")
                print(f"max_experiment_runs: {self.max_experiment_runs}")
                if not self.train_online(stop_event=stop_event):
                    self.stop_record_rosbag()
                    return
                self.stop_record_rosbag()
                if self.exp_run < self.max_experiment_runs:
                    self.init_run()
                else:
                    experiment_finished = True

    def start_rollout(self, stop_event=None, exp_run=0):
        self.init_run(exp_run=exp_run)
        self.rollout(stop_event=stop_event)
        self.stop_record_rosbag()

    def train_online(self, stop_event=None):
        # t = 0
        # if self.t > 0:
        #     buffer_paths = np.load(self.result_path+"/buffer_paths.npy", allow_pickle=True)
        #     self.RL_agent.replay_buffer.load_paths(buffer_paths)
        # elif self.init_buffer_paths:
        #     buffer_paths = np.load(self.result_path+"/init_buffer_paths.npy", allow_pickle=True)
        #     self.RL_agent.replay_buffer.load_paths(buffer_paths)
        #     print(f"Loaded initial buffer paths of length: {len(buffer_paths[0]['observations'])}")
        #     print(f"Buffer size: {self.RL_agent.replay_buffer.size}")
        #     t = self.RL_agent.replay_buffer.size-1
            

        # self.evals = []
        self.record_process = self.start_record_rosbag("rosbag_"+str(int(time.time())))
        start_time = time.time()
        if self.t >= self.timesteps_before_training: allow_train = True
        else: allow_train = False

        state, info = self.env.reset()
        ep_total_reward, ep_timesteps,= 0, 0
        ep_num = int(self.t/self.ep_length) + 1
        self.evaluated = False

        # for i in range(int(args.max_timesteps+1)):
        while self.t < int(self.max_timesteps+1):
            self.maybe_evaluate_and_print(self.t, start_time)
            
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

            # if allow_train and not self.use_checkpoints:
            #     self.RL_agent.train()

            if ep_finished: 
                print(f"Reward: {ep_total_reward}")
                print(f"Total T: {self.t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")
                

                if allow_train:
                    if self.use_checkpoints:
                        self.RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)
                    else:
                        for _ in range(ep_timesteps):
                            self.RL_agent.train()

                if self.t >= self.timesteps_before_training:
                    allow_train = True
                    if self.t >= self.timesteps_before_training and self.t <= (self.timesteps_before_training+ep_timesteps):
                        print("Saving initial buffer paths")
                        self.RL_agent.replay_buffer.save_paths(self.result_path+"/init_buffer_paths.npy")

                state, info = self.env.reset()
                ep_total_reward, ep_timesteps = 0, 0
                ep_num += 1 
                
                if stop_event.is_set() or not info["reset_success"]:
                    self.t += 1
                    self.save_training_state()
                    return False

            self.t += 1

            if ep_finished and self.evaluated:
                self.evaluated = False
                self.save_training_state()

        # Save final model
        self.exp_run += 1
        self.t = 0
        self.save_training_state()
        return True
        # self.RL_agent.save_model(self.result_path)
        # self.RL_agent.replay_buffer.save_paths(os.path.join(self.result_path,"buffer_paths.npy"))
        # self.RL_agent.replay_buffer.save_priority(os.path.join(self.result_path, "priority.npy"))
        # self.RL_agent.save_class_variables(self.result_path)
    
    def train_offline(self):
        pass

    def rollout(self, stop_event=None):
        self.record_process = self.start_record_rosbag("final_eval_rosbag_"+str(int(time.time())))
        total_reward = np.zeros(10)
        eval_eps = 10
        print("---------------------------------------")
        print(f"rollout for experiment: {self.exp_run}")
        for ep in range(eval_eps):
            print("---------------------------------------")
            print(f"Evaluation {ep}")
            state, info = self.eval_env.reset()
            state, info = self.eval_env.special_reset(ep)

            if stop_event.is_set() or not info["reset_success"]:
                    return False
            
            cntr = 0
            while cntr < self.ep_length:
                action = self.RL_agent.select_action(np.array(state), self.use_checkpoints, use_exploration=False)
                state, reward, done, _ = self.eval_env.step(action)
                # eval_env.render()
                total_reward[ep] += reward
                cntr += 1
            
            print(f"Episode {ep} reward: {total_reward[ep]:.3f}")
            if reward == 1:
                print(f"Episode {ep} successful")

            

        print(f"Average total reward over {eval_eps} episodes: {total_reward.mean():.3f}")
        print("---------------------------------------")
        self.eval_env.reset()

    def maybe_evaluate_and_print(self, t, start_time):
        if t % self.eval_freq == 0:
            print("---------------------------------------")
            print(f"Evaluation at {t} time steps")
            print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

            ep_succ = False

            total_reward = np.zeros(self.eval_eps)
            for ep in range(self.eval_eps):
                state, info = self.eval_env.reset()
                done = False
                cntr = 0
                while not done and cntr < self.ep_length:
                    action = self.RL_agent.select_action(np.array(state), self.use_checkpoints, use_exploration=False)
                    state, reward, done, _ = self.eval_env.step(action)
                    total_reward[ep] += reward
                    cntr += 1

                if reward == 1:
                    ep_succ = True
                            
            print(f"Average total reward over {self.eval_eps} episodes: {total_reward.mean():.3f}")

            print("---------------------------------------")

            self.evals.append(total_reward)
            # np.save(f"./results/{args.file_name}", evals)
            np.save(os.path.join(self.result_path,"results.npy"), self.evals)
            
            

            # ep_cntr = self.eval_eps
            # while ep_succ and ep_cntr < 10:
            #     state, info = self.eval_env.reset()
            #     done = False
            #     cntr = 0
            #     while not done and cntr < self.ep_length:
            #         action = self.RL_agent.select_action(np.array(state), self.use_checkpoints, use_exploration=False)
            #         state, reward, done, _ = self.eval_env.step(action)
            #         total_reward[ep] += reward
            #         cntr += 1

            #     if reward == 1:
            #         ep_succ = True

            #     ep_cntr += 1

            # print(f"Total amount of successful evaluation episodes: {ep_cntr}")
            self.eval_env.reset()
            self.evaluated = True
        

        

    def save_training_state(self):
        self.RL_agent.save_model(self.result_path)
        self.RL_agent.replay_buffer.save_paths(os.path.join(self.result_path,"buffer_paths.npy"))
        self.RL_agent.replay_buffer.save_priority(os.path.join(self.result_path, "priority.npy"))
        self.RL_agent.save_class_variables(self.result_path)

        data = dict()
        data['experiment_run'] = self.exp_run
        data['t'] = self.t
        print("saving data: ", data)

        json.dump(data, open(os.path.join(self.load_dir,'training_state.json'), 'w'))
        

    def load_training_state(self):
        try:
            with open(os.path.join(self.load_dir,'training_state.json')) as f:
                data = json.load(f)
        except FileNotFoundError:
            return False
        # with open(os.path.join(self.result_path,'training_state.json')) as f:
        #     data = json.load(f)
        self.exp_run = data['experiment_run']
        self.t = data['t']
        print("loaded data: ", data)
        
        return True
    
    def start_record_rosbag(self, rosbag_name):
        topic_list = ""
        topic_list += " /gh360/motor_states_sorted"
        topic_list += " /gh360/joint_states"
        topic_list += " /gh360/eef_pose"
        topic_list += " /gh360/motor_goal_velocity"
        topic_list += " /door/environment_observations"
        topic_list += " /gh360_control/cmd_eef_vel"
        topic_list += " /gh360_control/cmd_joint_vel"
        topic_list += " /gh360_control/cmd_joint_pos"
        topic_list += " /gh360_control/cmd_motor_pos"

        pre_command = f'source {self.path_ros2_ws}/install/setup.bash;'
        
        process_command = f'{pre_command} ros2 bag record -o {self.result_path}/{rosbag_name}{topic_list}'
        # process_command += ' -a'
        record_process = subprocess.Popen(process_command, shell=True, executable="/bin/bash", preexec_fn=os.setsid)

        return record_process
    
    def stop_record_rosbag(self):
        print("Stop Recording")
        if self.record_process.poll() is None:
            os.killpg(os.getpgid(self.record_process.pid), signal.SIGINT)
            self.record_process.wait()
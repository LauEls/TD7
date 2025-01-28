import argparse
import os
import time

import gym
import numpy as np
import torch
import json
import robosuite as suite
from robosuite.wrappers import GymWrapper

import TD7
from util import NormalizedBoxEnv


def rollout(RL_agent, eval_env, args):
	
	# print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

	total_reward = np.zeros(10)

	for ep in range(10):
		print("---------------------------------------")
		print(f"Evaluation {ep}")
		state, done = eval_env.reset(), False
		eval_env.render()
		cntr = 0
		while not done and cntr < args.ep_length:
			action = RL_agent.select_action(np.array(state), args.use_checkpoints, use_exploration=False)
			state, reward, done, _ = eval_env.step(action)
			eval_env.render()
			total_reward[ep] += reward
			cntr += 1
		print(f"Episode {ep} reward: {total_reward[ep]:.3f}")

	print(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f}")
	print("---------------------------------------")


def train_online(RL_agent, env, eval_env, args):
	allow_train = False
	evals = []
	
	if RL_agent.continue_learning:
		buffer_paths = np.load(args.load_dir+"/buffer_paths.npy", allow_pickle=True)
		RL_agent.replay_buffer.load_paths(buffer_paths)
		allow_train = True
		args.timesteps_before_training = 0
		print("Continue Learning")

	
	start_time = time.time()

	state, ep_finished = env.reset(), False
	# for i in range(600):
	# 	env.render()
	# 	time.sleep(0.1)
	ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

	for t in range(int(args.max_timesteps+1)):
		maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
		
		if allow_train:
			action = RL_agent.select_action(np.array(state))
		else:
			action = env.action_space.sample()

		next_state, reward, ep_finished, _ = env.step(action) 

		if args.render: env.render()
		
		ep_total_reward += reward
		ep_timesteps += 1

		if ep_timesteps >= args.ep_length: ep_finished = 1
		# done = float(ep_finished) if ep_timesteps < 500 else 0
		done = ep_finished
		
		RL_agent.replay_buffer.add(state, action, next_state, reward, done)

		state = next_state

		if allow_train and not args.use_checkpoints:
			RL_agent.train()

		if ep_finished: 
			print(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")

			if allow_train and args.use_checkpoints:
				RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

			if t >= args.timesteps_before_training:
				allow_train = True

			state, done = env.reset(), False
			ep_total_reward, ep_timesteps = 0, 0
			ep_num += 1 

	# Save final model
	RL_agent.save_model(args.result_path)


def train_offline(RL_agent, env, eval_env, paths, args):
	# RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))
	RL_agent.replay_buffer.load_paths(paths)

	evals = []
	start_time = time.time()

	for t in range(int(args.max_timesteps+1)):
		maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=False)
		RL_agent.train()

	RL_agent.save_model(args.result_path)


def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=False):
	if t % args.eval_freq == 0:
		print("---------------------------------------")
		print(f"Evaluation at {t} time steps")
		print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

		total_reward = np.zeros(args.eval_eps)
		for ep in range(args.eval_eps):
			state, done = eval_env.reset(), False
			cntr = 0
			while not done and cntr < args.ep_length:
				action = RL_agent.select_action(np.array(state), args.use_checkpoints, use_exploration=False)
				state, reward, done, _ = eval_env.step(action)
				total_reward[ep] += reward
				cntr += 1

		print(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f}")
		if d4rl:
			total_reward = eval_env.get_normalized_score(total_reward) * 100
			print(f"D4RL score: {total_reward.mean():.3f}")
		
		print("---------------------------------------")

		evals.append(total_reward)
		# np.save(f"./results/{args.file_name}", evals)
		np.save(os.path.join(args.result_path,"results.npy"), evals)


if __name__ == "__main__":
	# experimental_runs = 1
	# for i in range(experimental_runs):
	if True:
		i = 0
		load_dir = "runs/lift/panda/osc_pose/offline/v5_medium_expert_2"
		# load_dir = "runs/lift/panda/osc_pose/online/v8_reduced_ep_len_500"
		# load_dir = "runs/stack/panda/osc_pose/online/v1"
		# load_dir = "runs/trajectory_following/gh360t/eq_soft/v5_motor_vel"
		# load_dir = "runs/trajectory_following/gh360t/eq_vs/v1"
		# load_dir = "runs/door_mirror/gh360/osc_pose/v1_old_reward_system"
		# load_dir = "runs/door_mirror/gh360/osc_pose/v2_new_reward_system"
		# load_dir = "runs/door_mirror/gh360/joint_velocity/v4_test_joint_limit_2"
		# load_dir = "runs/door_mirror/gh360/joint_velocity/v2_new_reward_system"
		# load_dir = "runs/door_mirror/gh360t/eq_soft/v5_new_door_pos_no_motor_obs"
		# load_dir = "runs/door_mirror/gh360t/eq_soft/v4_old_rewards_motor_obs"

		kwargs_fpath = os.path.join(load_dir, "variant.json")
		try:
			with open(kwargs_fpath) as f:
				variant = json.load(f)
		except FileNotFoundError:
			print("Error opening default controller filepath at: {}. "
				"Please check filepath and try again.".format(kwargs_fpath))
			
		env_config = variant["environment_kwargs"]
		env_name = variant["environment_kwargs"]["env_name"]
		# seed = variant["seed"]
		seed = i
		offline = variant["offline"]
		demo_buffer = variant["demo_buffer"]
		# use_checkpoints = True

		# Load controller
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
		
		suite_eval_env = suite.make(**env_config,
								#  has_renderer=variant["render"],
								has_renderer=True,
								has_offscreen_renderer=False,
								use_object_obs=True,
								use_camera_obs=False,
								controller_configs=controller_config,
								)
			
		env = NormalizedBoxEnv(GymWrapper(suite_env))
		eval_env = NormalizedBoxEnv(GymWrapper(suite_eval_env))

		parser = argparse.ArgumentParser()
		# RL
		# parser.add_argument("--env", default="HalfCheetah-v4", type=str)
		# parser.add_argument("--seed", default=0, type=int)
		# parser.add_argument("--offline", default=False, action=argparse.BooleanOptionalAction)
		parser.add_argument('--use_checkpoints', default=True)
		# Evaluation
		parser.add_argument("--timesteps_before_training", default=25e3, type=int)
		parser.add_argument("--eval_freq", default=5e3, type=int)
		parser.add_argument("--eval_eps", default=10, type=int)
		parser.add_argument("--max_timesteps", default=5e6, type=int)
		parser.add_argument("--ep_length", default=500, type=int)
		# File
		parser.add_argument('--file_name', default=None)
		parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)
		parser.add_argument('--result_path', default="./results", type=str)
		parser.add_argument('--load_dir', default="", type=str)
		parser.add_argument('--rollout', default=False, type=bool)
		parser.add_argument('--render', default=False, type=bool)
		args = parser.parse_args()

		if True:
			args.ep_length = variant["episode_length"]
			args.timesteps_before_training = args.ep_length*50
			args.eval_freq = args.ep_length*10
			args.max_timesteps = args.ep_length*10000

		args.render = variant["render"]

		if offline:
			# import d4rl
			# d4rl.set_dataset_path(args.d4rl_path)

			paths = np.load(os.path.join("demonstrations/",variant["demo_file_name"]), allow_pickle=True)
			# print("Loaded paths length: ", len(paths))
			args.use_checkpoints = False

		if args.file_name is None:
			args.file_name = f"TD7_{env_name}_{seed}"

		result_path = os.path.join(load_dir, "run_"+str(i))

		if not os.path.exists(result_path):
			os.makedirs(result_path)
		args.result_path = result_path
		args.load_dir = load_dir
		
		# if not os.path.exists("./results"):
		# 	os.makedirs("./results")

		# env = gym.make(args.env)
		# eval_env = gym.make(args.env)

		print("---------------------------------------")
		print(f"Algorithm: TD7, Env: {env_name}, Seed: {seed}")
		print("---------------------------------------")

		env.seed(seed)
		env.action_space.seed(seed)
		eval_env.seed(seed+100)
		torch.manual_seed(seed)
		np.random.seed(seed)

		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0] 
		max_action = float(env.action_space.high[0])

		hp = TD7.Hyperparameters(**variant["hyperparameters"])
		hp.dir_path = load_dir
		
		RL_agent = TD7.Agent(state_dim, action_dim, max_action, demo_buffer=demo_buffer, offline=offline, hp=hp)

		if demo_buffer:
			paths = np.load(os.path.join("demonstrations/",variant["demo_file_name"]), allow_pickle=True)
			RL_agent.demo_buffer.load_paths(paths)

		if not args.rollout:
			if offline:
				train_offline(RL_agent, env, eval_env, paths, args)
			else:
				train_online(RL_agent, env, eval_env, args)
		else:
			RL_agent.load_model(args.result_path)
			rollout(RL_agent, eval_env, args)
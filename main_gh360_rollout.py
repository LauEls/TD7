import gh360_gym
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



# def train_online(RL_agent, env, eval_env, args):
# 	if RL_agent.continue_learning:
# 		buffer_paths = np.load(args.result_path+"/buffer_paths.npy", allow_pickle=True)
# 		RL_agent.replay_buffer.load_paths(buffer_paths)

# 	evals = []
# 	start_time = time.time()
# 	allow_train = False

# 	state, ep_finished = env.reset(), False
# 	ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

# 	for t in range(int(args.max_timesteps+1)):
# 		maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
		
# 		if allow_train:
# 			action = RL_agent.select_action(np.array(state))
# 		else:
# 			action = env.action_space.sample()

# 		next_state, reward, ep_finished, _ = env.step(action) 
		
# 		ep_total_reward += reward
# 		ep_timesteps += 1

# 		if ep_timesteps >= 500: ep_finished = 1
# 		# done = float(ep_finished) if ep_timesteps < 500 else 0
# 		done = ep_finished
		
# 		RL_agent.replay_buffer.add(state, action, next_state, reward, done)

# 		state = next_state

# 		if allow_train and not args.use_checkpoints:
# 			RL_agent.train()

# 		if ep_finished: 
# 			print(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")

# 			if allow_train and args.use_checkpoints:
# 				RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

# 			if t >= args.timesteps_before_training:
# 				allow_train = True

# 			state, done = env.reset(), False
# 			ep_total_reward, ep_timesteps = 0, 0
# 			ep_num += 1 

# 	# Save final model
# 	RL_agent.save_model(args.result_path)


def train_offline(RL_agent, env, eval_env, paths, args):
    # RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))
    RL_agent.replay_buffer.load_paths(paths)

    evals = []
    start_time = time.time()

    for t in range(int(args.max_timesteps+1)):
        if args.eval_during_training:
            maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=False)
        RL_agent.train()
        if t % 1000 == 0:
            print(f"training_steps: {t}")

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
			while not done and cntr < 500:
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
          
def rollout(RL_agent, eval_env, episode_length, num_episodes):

    total_reward = np.zeros(num_episodes)
    for ep in range(num_episodes):
        state, done = eval_env.reset(), False
        cntr = 0
        while not done and cntr < episode_length:
            action = RL_agent.select_action(np.array(state), use_exploration=False)
            state, reward, done, _ = eval_env.step(action)
            total_reward[ep] += reward
            cntr += 1
        print(f"Episode {ep} reward: {total_reward[ep]}")    
    
    return total_reward


if __name__ == "__main__":
    load_dir = "runs/door/real_gh360/motor_vel/offline/v1_first_try"

    kwargs_fpath = os.path.join(load_dir, "variant.json")
    try:
        with open(kwargs_fpath) as f:
            variant = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
            "Please check filepath and try again.".format(kwargs_fpath))
            
    env_config = variant["environment_kwargs"]
    env_name = variant["environment_kwargs"].pop("env_name")
    # seed = variant["seed"]
    seed = 5
    offline = variant["offline"]
    # use_checkpoints = True

    env = gym.make('gh360_gym/'+env_name, **env_config)
    env = NormalizedBoxEnv(env)
    eval_env = NormalizedBoxEnv(env)    
        

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
    # File
    parser.add_argument('--file_name', default=None)
    parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)
    parser.add_argument('--result_path', default="./results", type=str)
    parser.add_argument('--rollout', default=False, type=bool)
    args = parser.parse_args()


    if args.file_name is None:
        args.file_name = f"TD7_{env_name}_{seed}"

    result_path = os.path.join(load_dir, "run_"+str(0))

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
    print(f"max action: {env.action_space.high}")
    max_action = float(max(env.action_space.high))

    hp = TD7.Hyperparameters(**variant["hyperparameters"])
    hp.dir_path = result_path
    hp.continue_learning = True
    
    RL_agent = TD7.Agent(state_dim, action_dim, max_action, offline=offline, hp=hp)

    # rollout(RL_agent=RL_agent, eval_env=eval_env, episode_length=100, num_episodes=10)


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



def train_online(RL_agent, env, eval_env, args):
    t = 0
    if RL_agent.continue_learning:
        buffer_paths = np.load(args.result_path+"/buffer_paths.npy", allow_pickle=True)
        RL_agent.replay_buffer.load_paths(buffer_paths)
    elif args.init_buffer_paths:
        buffer_paths = np.load(args.result_path+"/init_buffer_paths.npy", allow_pickle=True)
        RL_agent.replay_buffer.load_paths(buffer_paths)
        print(f"Loaded initial buffer paths of length: {len(buffer_paths[0]['observations'])}")
        print(f"Buffer size: {RL_agent.replay_buffer.size}")
        t = RL_agent.replay_buffer.size-1
        

    evals = []
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset(), False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    # for i in range(int(args.max_timesteps+1)):
    while t < int(args.max_timesteps+1):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
        
        if allow_train:
            action = RL_agent.select_action(np.array(state))
        else:
            action = env.action_space.sample()

        next_state, reward, ep_finished, _ = env.step(action) 
        
        
        ep_total_reward += reward
        ep_timesteps += 1

        if ep_timesteps >= args.ep_length: 
            ep_finished = 1
            env.step(np.zeros(RL_agent.action_dim))
        # done = float(ep_finished) if ep_timesteps < 500 else 0
        done = ep_finished
        
        RL_agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        if allow_train and not args.use_checkpoints:
            RL_agent.train()

        if ep_finished: 
            print(f"Reward: {ep_total_reward}")
            print(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")

            if allow_train and args.use_checkpoints:
                RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

            if t >= args.timesteps_before_training:
                allow_train = True
                if t >= args.timesteps_before_training and t <= (args.timesteps_before_training+ep_timesteps):
                    print("Saving initial buffer paths")
                    RL_agent.replay_buffer.save_paths(args.result_path+"/init_buffer_paths.npy")

            state, done = env.reset(), False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1 

        t += 1

    # Save final model
    RL_agent.save_model(args.result_path)


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
    if t % args.eval_freq == 0 and t > 0:
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

        eval_env.reset()


if __name__ == "__main__":
    experimental_runs = 1
    for i in range(experimental_runs):
        load_dir = "runs/door/real_gh360/motor_vel/online/v3_open_door_2"

        kwargs_fpath = os.path.join(load_dir, "variant.json")
        try:
            with open(kwargs_fpath) as f:
                variant = json.load(f)
        except FileNotFoundError:
            print("Error opening default controller filepath at: {}. "
                "Please check filepath and try again.".format(kwargs_fpath))
            
        env_config = variant["environment_kwargs"]
        env_name = variant["environment_kwargs"].pop("env_name")
        seed = variant["seed"]
        # seed = i+1
        offline = variant["offline"]
        # use_checkpoints = True

        raw_env = gym.make('gh360_gym/'+env_name, **env_config)
        env = NormalizedBoxEnv(raw_env)
        eval_env = env
        # eval_env = gym.make(args.env)
        

        parser = argparse.ArgumentParser()
        # RL
        # parser.add_argument("--env", default="HalfCheetah-v4", type=str)
        # parser.add_argument("--seed", default=0, type=int)
        # parser.add_argument("--offline", default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_checkpoints', default=True)
        # Evaluation
        parser.add_argument("--timesteps_before_training", default=25e3, type=int)
        parser.add_argument("--eval_freq", default=5e3, type=int)
        parser.add_argument("--eval_eps", default=5, type=int)
        parser.add_argument("--max_timesteps", default=5e6, type=int)
        parser.add_argument("--ep_length", default=500, type=int)
        # File
        parser.add_argument('--file_name', default=None)
        parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)
        parser.add_argument('--result_path', default="./results", type=str)
        parser.add_argument('--rollout', default=False, type=bool)
        args = parser.parse_args()

        # args.ep_length = 500/5
        # args.timesteps_before_training = 25000/5
        # args.eval_freq = 5e3/5
        # args.max_timesteps = 5e6/5

        args.ep_length = variant["episode_length"]
        args.timesteps_before_training = args.ep_length*50
        args.eval_freq = args.ep_length*10
        args.max_timesteps = args.ep_length*10000
        variant["hyperparameters"]["buffer_size"] = args.ep_length*2000
        # variant["hyperparameters"]["batch_size"] = int(args.ep_length/2)
        args.init_buffer_paths = variant["init_buffer_paths"]
        
        args.eval_eps = 1


        if offline:
            # import d4rl
            # d4rl.set_dataset_path(args.d4rl_path)
            expert_paths = np.load(os.path.join("demonstrations/",variant["demo_file_name"]), allow_pickle=True)
            print("Expert paths shape: ", expert_paths.shape)
            random_paths_file = os.path.join("demonstrations/",variant["demo_file_name"])
            random_paths_file = random_paths_file[0:-4] + "_random_paths" + random_paths_file[-4:]
            print(random_paths_file)
            random_paths = np.load(random_paths_file, allow_pickle=True)
            print("Random paths shape: ", random_paths.shape)
            paths = np.concatenate((expert_paths, random_paths))
            # print("Loaded paths length: ", len(paths))
            args.use_checkpoints = False
            args.eval_during_training = variant["eval_during_training"]

        # if args.rollout:
        #     env = gym.make('gh360_gym/'+env_name, **env_config)
        #     env = NormalizedBoxEnv(env)
        #     eval_env = NormalizedBoxEnv(env)

        if args.file_name is None:
            args.file_name = f"TD7_{env_name}_{seed}"

        result_path = os.path.join(load_dir, "run_"+str(i))

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        args.result_path = result_path
        
        # if not os.path.exists("./results"):
        # 	os.makedirs("./results")

        

        print("---------------------------------------")
        print(f"Algorithm: TD7, Env: {env_name}, Seed: {seed}")
        print("---------------------------------------")

        env.seed(seed)
        env.action_space.seed(seed)
        eval_env.seed(seed+100)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if offline:
            print("Expert paths: ", expert_paths[0]['actions'].shape)
            print("Random paths: ", random_paths[0]['actions'].shape)
            print("Paths: ", paths[0]['observations'].shape)
            print("Paths: ", paths[30]['observations'].shape)
            state_dim = paths[0]['observations'].shape[1]
            action_dim = paths[0]['actions'].shape[1]
            max_action = 0.0
            for path in paths:
                new_max_action = np.max(np.abs(path['actions']))
                if new_max_action > max_action: max_action = new_max_action
            print("Max action: ", max_action)
        else:
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0] 
            max_action = float(env.action_space.high[0])

        print("State dim: ", state_dim)
        print("Action dim: ", action_dim)
        print("Max action: ", max_action)
        hp = TD7.Hyperparameters(**variant["hyperparameters"])
        hp.dir_path = result_path
        
        RL_agent = TD7.Agent(state_dim, action_dim, max_action, offline=offline, hp=hp)

        if offline:
            train_offline(RL_agent, env=None, eval_env=None, paths=paths, args=args)
        else:
            train_online(RL_agent, env, eval_env, args)


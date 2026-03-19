import TD7
import os
import json
import gym
import time

import numpy as np
import gh360_gym
from gh360_interfaces.srv import LogTime
from std_msgs.msg import String
import rclpy
from rclpy.node import Node

class LogTimeClient(Node):
    def __init__(self):
        super().__init__('gh360_log_time_client')
        self.cli = self.create_client(LogTime, 'erf_log_time')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = LogTime.Request()

    def send_request(self, time):
        username = String()
        username.data = "GH360"
        self.req.username = username
        self.req.time = time
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def rollout(eval_env, RL_agent, ep_length=130, log_time_client=None):
    try:
        while True:
            state, info = eval_env.reset()
            
            cntr = 0
            total_reward = 0
            success = False
            start_time = time.time()
            while cntr < ep_length:
                action = RL_agent.select_action(np.array(state), use_exploration=False)
                state, reward, done, _ = eval_env.step(action)
                # eval_env.render()
                total_reward += reward
                cntr += 1

                if reward == 1:
                    print(f"Episode successful")
                    success = True
                    break
            
            print(f"Episode reward: {total_reward:.3f}")
            end_time = time.time()
            print("Rollout Finished in {} seconds".format(end_time - start_time))
            if log_time_client and success:
                log_time_client.send_request(end_time - start_time)
            
        
    except KeyboardInterrupt:
        eval_env.reset()
    


if __name__ == "__main__":
    # rclpy.init(args=None)
    load_dir = "/home/gh360/TD7/runs/door/real_gh360/eef_vel/online/v16_erf"

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

    log_time_client = LogTimeClient()
    rollout(env, RL_agent, ep_length, log_time_client)
    log_time_client.destroy_node()
    rclpy.shutdown()
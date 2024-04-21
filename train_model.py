#!/usr/bin/env python3

# Import this file so that the environment is available on gym.make()
import discrete_fetch_reach

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
# from charts_callbackers import ChartsCallback

# load env
env = gym.make('FetchReachDense-custom', max_episode_steps=50) #, render_mode="human")

observation, info = env.reset(seed=42)

new_logger = configure("./logs/", ["stdout", "csv"])

# train model
model = DQN("MultiInputPolicy", env, verbose=1)
#model = DQN.load("dqn_fetch_reach")
#model.set_env(env)
model.set_logger(new_logger)
model.learn(total_timesteps=1e6, log_interval=20)
model.save("dqn_fetch_reach")

print("training finished")

env.close()

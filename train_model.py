#!/usr/bin/env python3

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from envs.discrete_fetch_reach import DiscreteFetchReach


# load env
env = DiscreteFetchReach(max_episode_steps=50)

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

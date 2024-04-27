#!/usr/bin/env python3

import logging
import time

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from env import FetchReachJointsControl


# set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
env_logger = configure("./logs/", ["stdout", "csv"])

# create env
logger.info("Creating environment...")
env = FetchReachJointsControl(max_episode_steps=50)

# train model
logger.info("Starting model training...")
t1 = time.time()

observation, info = env.reset(seed=42)

model = DQN("MultiInputPolicy", env, verbose=1)
# model = DQN.load("dqn_fetch_reach")
# model.set_env(env)
model.set_logger(env_logger)
model.learn(total_timesteps=1e6, log_interval=20)
model.save("models/dqn_fetch_reach")

env.close()

t2 = time.time()
logger.info("Model training finished!")
logger.info(f"Training took {t2-t1:.2f} seconds")

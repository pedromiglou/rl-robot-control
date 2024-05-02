#!/usr/bin/env python3

import logging
import time

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from envs.fetch_reach_cartesian.discrete import FetchReachCartesianDiscrete


RESULTS_FOLDER = "./results/fetch_reach_cartesian_discrete"

# set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename=f'{RESULTS_FOLDER}/logs.log')
env_logger = configure(RESULTS_FOLDER, ["stdout", "csv"])

# create env
logger.info("Creating environment...")
env = FetchReachCartesianDiscrete(max_episode_steps=50, render_mode=None)

# train model
logger.info("Starting model training...")
t1 = time.time()

observation, info = env.reset(seed=42)

model = SAC("MultiInputPolicy", env, verbose=1)
# model = DQN.load(f'{RESULTS_FOLDER}/model')
# model.set_env(env)
model.set_logger(env_logger)
model.learn(total_timesteps=2e6, log_interval=50)
model.save(f'{RESULTS_FOLDER}/model')

env.close()

t2 = time.time()
logger.info("Model training finished!")
logger.info(f"Training took {t2-t1:.2f} seconds")

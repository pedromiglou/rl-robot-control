#!/usr/bin/env python3

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
import time

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from custom_callback import CustomCallback
from larcc_env.wrapped_env import WrappedEnv


RESULTS_FOLDER = "./results/larcc_joints_continuous"
# RESULTS_FOLDER = "./results/larcc_joints_continuous/position_only"
# RESULTS_FOLDER = "./results/larcc_joints_continuous/orientation_only"

# set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename=f'{RESULTS_FOLDER}/logs.log', filemode='w')
env_logger = configure(RESULTS_FOLDER, ["stdout", "csv"])

logger.info("Creating environment...")
env = WrappedEnv(max_episode_steps=50)#, render_mode="human")
eval_env = WrappedEnv(max_episode_steps=50)

logger.info("Setting up callbacks...")
callback = CustomCallback(eval_env, best_model_save_path=RESULTS_FOLDER)

logger.info("Starting model training...")
t1 = time.time()

observation, info = env.reset(seed=42)

model = SAC("MultiInputPolicy", env, verbose=1)
# model = SAC.load(f'{RESULTS_FOLDER}/model')
# model.set_env(env)
model.set_logger(env_logger)
model.learn(total_timesteps=5e7, log_interval=500, callback=callback)
model.save(f'{RESULTS_FOLDER}/final_model')

env.close()

t2 = time.time()
logger.info("Model training finished!")
logger.info(f"Training took {t2-t1:.2f} seconds")

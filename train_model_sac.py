#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure

from larcc_env.wrapped_env import WrappedEnv


RESULTS_FOLDER = "./results/fetch_reach_joints_continuous/orientation_only"

# set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename=f'{RESULTS_FOLDER}/logs.log', filemode='w')
env_logger = configure(RESULTS_FOLDER, ["stdout", "csv"])

# create env
logger.info("Creating environment...")
env = WrappedEnv(max_episode_steps=50)#, render_mode="human")
eval_env = WrappedEnv(max_episode_steps=50)

# Stop training if there is no improvement after more than 3 evaluations
logger.info("Setting up callbacks...")
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
eval_callback = EvalCallback(eval_env, eval_freq=500*50, n_eval_episodes=100, callback_after_eval=stop_train_callback, verbose=1, best_model_save_path=RESULTS_FOLDER, deterministic=True)

# train model
logger.info("Starting model training...")
t1 = time.time()

observation, info = env.reset(seed=42)

model = SAC("MultiInputPolicy", env, verbose=1)
# model = SAC.load(f'{RESULTS_FOLDER}/model')
# model.set_env(env)
model.set_logger(env_logger)
model.learn(total_timesteps=5e6, log_interval=500, callback=eval_callback)
model.save(f'{RESULTS_FOLDER}/final_model')

env.close()

t2 = time.time()
logger.info("Model training finished!")
logger.info(f"Training took {t2-t1:.2f} seconds")

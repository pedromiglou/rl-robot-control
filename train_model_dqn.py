#!/usr/bin/env python3

import logging
import time

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure

from envs.fetch_reach_cartesian_discrete.env import FetchReachCartesianDiscrete


RESULTS_FOLDER = "./results/fetch_reach_cartesian_discrete"

# set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename=f'{RESULTS_FOLDER}/logs.log', filemode='w')
env_logger = configure(RESULTS_FOLDER, ["stdout", "csv"])

# create env
logger.info("Creating environment...")
env = FetchReachCartesianDiscrete(max_episode_steps=50, render_mode=None)
eval_env = FetchReachCartesianDiscrete(max_episode_steps=50, render_mode=None, deterministic=True)

# Stop training if there is no improvement after more than 10 evaluations
logger.info("Setting up callbacks...")
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
eval_callback = EvalCallback(eval_env, eval_freq=500*50, n_eval_episodes=50, callback_after_eval=stop_train_callback, verbose=1, best_model_save_path=RESULTS_FOLDER)

# train model
logger.info("Starting model training...")
t1 = time.time()

observation, info = env.reset(seed=42)

model = DQN("MultiInputPolicy", env, verbose=1, exploration_fraction=0.01)
# model = DQN.load(f'{RESULTS_FOLDER}/model')
# model.set_env(env)
model.set_logger(env_logger)
model.learn(total_timesteps=1e7, log_interval=500, callback=eval_callback)
model.save(f'{RESULTS_FOLDER}/final_model')

env.close()

t2 = time.time()
logger.info("Model training finished!")
logger.info(f"Training took {t2-t1:.2f} seconds")

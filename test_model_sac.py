#!/usr/bin/env python3

import gymnasium as gym
from stable_baselines3 import SAC

import envs.larcc_joints_continuous.env


RESULTS_FOLDER = "./results/fetch_reach_joints_continuous/position_only"
#RESULTS_FOLDER = "./results/fetch_reach_joints_continuous"

# load env
env = gym.make("Larcc", max_episode_steps=50, render_mode="human")
# env = FetchReachJointsContinuous(max_episode_steps=50, render_mode="rgb_array", record=True)

observation, info = env.reset(seed=42)

# test model
model = SAC.load(f'{RESULTS_FOLDER}/best_model')

observation, info = env.reset()
for _ in range(500):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:# or info["is_success"]:
        observation, info = env.reset()

env.close()

#!/usr/bin/env python3

from stable_baselines3 import DQN

from envs.fetch_reach_cartesian_discrete.env import FetchReachCartesianDiscrete


RESULTS_FOLDER = "./results/fetch_reach_cartesian_discrete"

# load env
env = FetchReachCartesianDiscrete(max_episode_steps=50, render_mode="human")
# env = FetchReachCartesianDiscrete(max_episode_steps=50, render_mode="rgb_array", record=True)

observation, info = env.reset(seed=42)

# test model
model = DQN.load(f'{RESULTS_FOLDER}/best_model')

observation, info = env.reset()
for _ in range(500):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated or info["is_success"]:
        observation, info = env.reset()

env.close()

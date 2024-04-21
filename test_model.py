#!/usr/bin/env python3

### Import this file so that the environment is available on gym.make()
import discrete_fetch_reach

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN


# load env
env = gym.make('FetchReachDense-custom', max_episode_steps=50, render_mode="human")

# record video
# env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
#                   episode_trigger=lambda x: True)

observation, info = env.reset(seed=42)

# test model
model = DQN.load("models/dqn_fetch_reach_corrected")

observation, info = env.reset()
for _ in range(500):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated or info["is_success"]:
        observation, info = env.reset()

env.close()

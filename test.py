import gymnasium as gym
import random

env = gym.make('FetchReach-v2', max_episode_steps=1000, render_mode="human")

actions = [
    [0.05, 0.05, 0.0, 0.0],
    [0.05, -0.05, 0.0, 0.0],
    [-0.05, 0.05, 0.0, 0.0],
    [-0.05, -0.05, 0.0, 0.0],
]

observation, info = env.reset(seed=42)


for _ in range(1000):
    #action = env.action_space.sample()
    action = random.choice(actions)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

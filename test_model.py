import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

# load env
env = gym.make('FetchReachDense-v2', max_episode_steps=50, render_mode="rgb_array")

# convert action space from continuous to discrete
values = [-0.2, 0, 0.2]
actions = [(x,y,z,0) for x in values for y in values for z in values]

env.action_space = gym.spaces.discrete.Discrete(len(actions))

env.continuous_step = env.step
env.step = lambda x: env.continuous_step(actions[x])

env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
                  episode_trigger=lambda x: True)

observation, info = env.reset(seed=42)

# test model
model = DQN.load("dqn_fetch_reach")

observation, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

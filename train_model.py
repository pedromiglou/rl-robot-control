import gymnasium as gym

from stable_baselines3 import DQN

# load env
env = gym.make('FetchReachDense-v2', max_episode_steps=50)#, render_mode="human")

# convert action space from continuous to discrete
values = [-0.2, 0, 0.2]
actions = [(x,y,z,0) for x in values for y in values for z in values]

env.action_space = gym.spaces.discrete.Discrete(len(actions))

env.continuous_step = env.step
env.step = lambda x: env.continuous_step(actions[x])

observation, info = env.reset(seed=42)

# train model
#model = DQN("MultiInputPolicy", env, verbose=1)
model = DQN.load("dqn_fetch_reach")
model.set_env(env)
model.learn(total_timesteps=500000, log_interval=10)
model.save("dqn_fetch_reach")

print("training finished")

env.close()

#!/usr/bin/env python3

import gymnasium as gym
import numpy as np

from gymnasium import spaces


class DiscreteFetchReach(gym.Env):
    def __init__(self, **kwargs):
        self.env = gym.make('FetchReachDense-v2', **kwargs)

        # set discrete actions
        values = [-1, 0, 1]
        self.discrete_actions = [(x,y,z,0) for x in values for y in values for z in values]
        for i in range(len(self.discrete_actions)):
            self.discrete_actions[i] = np.array(self.discrete_actions[i], dtype=np.float32)
            if np.linalg.norm(self.discrete_actions[i]) != 0:
                self.discrete_actions[i] *= 0.2 / np.linalg.norm(self.discrete_actions[i])
            
        self.action_space = spaces.discrete.Discrete(len(self.discrete_actions))
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(self.discrete_actions[action])
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

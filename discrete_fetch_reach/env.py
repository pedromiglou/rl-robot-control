#!/usr/bin/env python3

import gymnasium as gym
import numpy as np

from gymnasium.spaces.discrete import Discrete
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


class DiscreteFetchReach(gym.Env):
    def __init__(self, record=False, **kwargs):
        self.env = gym.make('FetchReachDense-v2', **kwargs)
        self.record = record

        if self.record:
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path="./videos/discrete_fetch_reach"
            )

        # set discrete actions
        values = [-1, 0, 1]
        self.discrete_actions = [(x,y,z,0) for x in values for y in values for z in values]
        for i in range(len(self.discrete_actions)):
            self.discrete_actions[i] = np.array(self.discrete_actions[i], dtype=np.float32)
            if np.linalg.norm(self.discrete_actions[i]) != 0:
                self.discrete_actions[i] *= 0.2 / np.linalg.norm(self.discrete_actions[i])
            
        self.action_space = Discrete(len(self.discrete_actions))
        self.observation_space = self.env.observation_space

    def step(self, action):
        if self.record:
            self.video_recorder.capture_frame()

        return self.env.step(self.discrete_actions[action])
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        if self.record:
            self.video_recorder.capture_frame()
            self.video_recorder.close()

        self.env.close()

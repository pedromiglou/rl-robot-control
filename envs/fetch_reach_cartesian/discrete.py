#!/usr/bin/env python3

import gymnasium as gym
import numpy as np

from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.discrete import Discrete
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from envs.fetch_reach_cartesian.utils import distance_between_points


class FetchReachCartesianDiscrete(gym.Env):
    """
    Gymanasium environment wrapper
    """
    def __init__(self, record=False, **kwargs):
        self.env = gym.make(
            'FetchReachDense-v2',
            reward_type="dense",
            width=720,
            height=720,
            default_camera_config = {
                "distance": 2.0,
                "azimuth": 132.0,
                "elevation": -14.0,
                "lookat": np.array([1.3, 0.75, 0.55])
            },
            **kwargs
        )

        self.record = record
        if self.record:
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path="./results/fetch_reach_cartesian_discrete/demo"
            )

        # set discrete actions with always the same distance delta
        values = [-1, 0, 1]
        self.discrete_actions = [(x,y,z,0) for x in values for y in values for z in values]
        for i in range(len(self.discrete_actions)):
            self.discrete_actions[i] = np.array(self.discrete_actions[i], dtype=np.float32)
            if np.linalg.norm(self.discrete_actions[i]) != 0:
                self.discrete_actions[i] /= np.linalg.norm(self.discrete_actions[i])
        
        # action space: movements of the end effector in 26 directions + stay still
        self.action_space = Discrete(len(self.discrete_actions))

        # observation space: end effector position, velocity, and target position
        self.observation_space = Dict({"observation": Box(-np.inf, np.inf, shape=(6,)), "desired_goal": Box(-np.inf, np.inf, shape=(3,))})

        # create a dict to store relevant info for the reward function
        self.reward_info = {}

    def fix_obs(self, obs):
        obs["observation"] = np.concatenate((obs["observation"][0:3], obs["observation"][5:8]))
        obs.pop("achieved_goal")
        return obs
    
    def compute_reward(self, obs, Kp=1.0):
        # compute the position error
        dist = distance_between_points(obs["desired_goal"], obs["observation"][:3])
        pos_error = (self.reward_info["initial_distance"] - dist) / self.reward_info["initial_distance"] # [-inf, 1]
        # compute the reward
        return Kp * pos_error

    def step(self, action):
        if self.record:
            self.video_recorder.capture_frame()
        
        obs, reward, terminated, truncated, info = self.env.step(self.discrete_actions[action])

        obs = self.fix_obs(obs)
        
        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()

        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs = self.fix_obs(obs)

        self.reward_info["initial_distance"] = distance_between_points(obs["desired_goal"], obs["observation"][:3])

        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()
        
        return obs, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        if self.record:
            self.video_recorder.close()

        self.env.close()

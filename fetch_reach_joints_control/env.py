#!/usr/bin/env python3

import gymnasium as gym
import numpy as np

from gymnasium_robotics.utils import mujoco_utils
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
from gymnasium.envs.registration import register
from gymnasium.spaces.discrete import Discrete
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


class CustomMujocoFetchReachEnv(MujocoFetchReachEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_action(self, action):
        pass
        #print("gg")

register(
    # unique identifier for the env `name-version`
    id='FetchReachDense-custom',
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="env:CustomMujocoFetchReachEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=50,
)


class FetchReachJointsControl(gym.Env):
    def __init__(self, record=False, **kwargs):
        self.env = gym.make('FetchReachDense-custom', **kwargs)

        self.joint_names = [
            "robot0:shoulder_pan_joint",
            "robot0:shoulder_lift_joint",
            "robot0:upperarm_roll_joint",
            "robot0:elbow_flex_joint",
            "robot0:forearm_roll_link",
            "robot0:wrist_flex_link",
            "robot0:wrist_roll_joint"
        ]
        
        self.record = record
        if self.record:
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path="./videos/discrete_fetch_reach"
            )

        # set discrete actions
        self.discrete_actions = [[0,0,0,0,0,0,0]]

        for i in range(7):
            a = [0,0,0,0,0,0,0]
            a[i] = 1
            self.discrete_actions.append(a)

            a = [0,0,0,0,0,0,0]
            a[i] = -1
            self.discrete_actions.append(a)

        for i in range(len(self.discrete_actions)):
            self.discrete_actions[i] = np.array(self.discrete_actions[i], dtype=np.float32)
            if np.linalg.norm(self.discrete_actions[i]) != 0:
                self.discrete_actions[i] *= 0.1 / np.linalg.norm(self.discrete_actions[i])
            
        self.action_space = Discrete(len(self.discrete_actions))
        self.observation_space = self.env.observation_space

    def step(self, action):
        if self.record:
            self.video_recorder.capture_frame()

        x = mujoco_utils.robot_get_obs(self.env.model, self.env.data, self.joint_names)[0]
        action = np.array(self.discrete_actions[action])

        #print(x)
        action += x
        #print(action)

        for i in range(7):            
            mujoco_utils.set_joint_qpos(self.env.model, self.env.data, self.joint_names[i], action[i])
            mujoco_utils.reset_mocap2body_xpos(self.env.model, self.env.data)

        action = np.array([0, 0, 0, 0])

        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        if self.record:
            self.video_recorder.capture_frame()
            self.video_recorder.close()

        self.env.close()

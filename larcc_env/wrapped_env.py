#!/usr/bin/env python3

import gymnasium as gym
import numpy as np

from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

import larcc_env.base_env


class WrappedEnv(gym.Env):
    """
    Larcc environment wrapper to use with the RL algorithms.
    """
    def __init__(self, record_path=None, **kwargs):
        self.env = gym.make(
            'Larcc',
            **kwargs
        )

        # self.joint_names = [
        #     "robot0:shoulder_pan_joint",
        #     "robot0:shoulder_lift_joint",
        #     "robot0:upperarm_roll_joint",
        #     "robot0:elbow_flex_joint",
        #     "robot0:forearm_roll_joint",
        #     "robot0:wrist_flex_joint",
        #     "robot0:wrist_roll_joint"
        # ]

        # self.joint_values_file = open("./results/fetch_reach_cartesian_discrete/joint_values.txt", "w")

        self.record = record_path is not None
        if self.record:
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path=record_path
            )

        # set discrete actions with always the same distance delta
        # values = [-1, 0, 1]
        # self.discrete_actions = [(x,y,z,0) for x in values for y in values for z in values]
        # for i in range(len(self.discrete_actions)):
        #     self.discrete_actions[i] = np.array(self.discrete_actions[i], dtype=np.float32)
        #     if np.linalg.norm(self.discrete_actions[i]) != 0:
        #         self.discrete_actions[i] /= np.linalg.norm(self.discrete_actions[i])
        
        # action space: movements of the end effector in 26 directions + stay still
        # self.action_space = Discrete(len(self.discrete_actions))
        self.action_space = Box(-1, 1, shape=(6,), dtype="float32")

        # observation space: end effector position, velocity, and target position
        self.observation_space = Dict(
            dict(
                desired_goal=Box(
                    -1, 1, shape=(7,), dtype="float64"
                ),
                achieved_goal=Box(
                    -1, 1, shape=(7,), dtype="float64"
                ),
                observation=Box(
                    -1, 1, shape=(6,), dtype="float64"
                ),
            )
        )

    def normalize_obs(self, obs):
        obs["observation"] = obs["observation"]/(2*np.pi)

        obs["achieved_goal"][0] = (obs["achieved_goal"][0] - 0.100) / 1.8
        obs["achieved_goal"][1] = (obs["achieved_goal"][1] - 0.950) / 1.8
        obs["achieved_goal"][2] = (obs["achieved_goal"][2] - 0.780) / 1.8

        obs["desired_goal"][0] = (obs["desired_goal"][0] - 0.100) / 1.8
        obs["desired_goal"][1] = (obs["desired_goal"][1] - 0.950) / 1.8
        obs["desired_goal"][2] = (obs["desired_goal"][2] - 0.780) / 1.8

        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self.normalize_obs(obs)

        # joint_values = list(mujoco_utils.robot_get_obs(self.env.model, self.env.data, self.joint_names)[1])
        # self.joint_values_file.write(f"{','.join([str(v) for v in joint_values])}\n")
        
        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()

            #if info["is_success"]: # if the episode is successful, capture more frames
                #for _ in range(10):
                #    self.video_recorder.capture_frame()

        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs = self.normalize_obs(obs)

        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()
        
        return obs, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        #self.joint_values_file.close()
        if self.record:
            self.video_recorder.close()

        self.env.close()

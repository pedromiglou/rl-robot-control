#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import os

from gymnasium.envs.registration import register
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.discrete import Discrete
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium_robotics.envs.fetch import MujocoFetchEnv
from gymnasium_robotics.utils import mujoco_utils

from utils import point_distance


class CustomMujocoFetchReachEnv(MujocoFetchEnv, EzPickle):
    """
    This class is needed to overwrite functions in MujocoFetchEnv.
    """
    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=os.path.join("fetch", "reach.xml"),
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.06,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


# register the custom environment
register(
    # unique identifier for the env `name-version`
    id='FetchReachDense-custom',
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=CustomMujocoFetchReachEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=50,
)


class FetchReachCartesianDiscrete(gym.Env):
    """
    Gymanasium environment wrapper
    """
    def __init__(self, record=False, **kwargs):
        self.env = gym.make(
            'FetchReachDense-custom',
            reward_type="dense",
            width=1280,
            height=720,
            default_camera_config = {
                "distance": 2.0,
                "azimuth": 132.0,
                "elevation": -14.0,
                "lookat": np.array([1.3, 0.75, 0.55])
            },
            **kwargs
        )

        self.joint_names = [
            "robot0:shoulder_pan_joint",
            "robot0:shoulder_lift_joint",
            "robot0:upperarm_roll_joint",
            "robot0:elbow_flex_joint",
            "robot0:forearm_roll_joint",
            "robot0:wrist_flex_joint",
            "robot0:wrist_roll_joint"
        ]

        # self.joint_values_file = open("./results/fetch_reach_cartesian_discrete/joint_values.txt", "w")

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
        dist = point_distance(obs["desired_goal"], obs["observation"][:3])
        pos_error = (self.reward_info["initial_distance"] - dist) / self.reward_info["initial_distance"] # [-inf, 1]
        # compute the reward
        return Kp * pos_error

    def step(self, action):
        if self.record:
            self.video_recorder.capture_frame()
        
        obs, reward, terminated, truncated, info = self.env.step(self.discrete_actions[action])

        obs = self.fix_obs(obs)

        reward = self.compute_reward(obs)

        # joint_values = list(mujoco_utils.robot_get_obs(self.env.model, self.env.data, self.joint_names)[1])
        # self.joint_values_file.write(f"{','.join([str(v) for v in joint_values])}\n")
        
        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()

            if info["is_success"]: # if the episode is successful, capture more frames
                for _ in range(10):
                    self.video_recorder.capture_frame()

        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs = self.fix_obs(obs)

        self.reward_info["initial_distance"] = point_distance(obs["desired_goal"], obs["observation"][:3])

        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()
        
        return obs, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        # self.joint_values_file.close()
        if self.record:
            self.video_recorder.close()

        self.env.close()

#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import os
import random

from gymnasium.envs.registration import register
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium_robotics.utils import mujoco_utils
from gymnasium_robotics.envs.fetch import MujocoFetchEnv

from utils import euler_to_quaternion, point_distance, random_euler_angles


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
            model_path=os.path.join(os.path.dirname(__file__), "model/env.xml"),
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)
    
    def _is_success(self, achieved_goal, desired_goal):
        return False
        # compute the position error
        #pos_error = point_distance(achieved_goal[:3], desired_goal[:3])
        # compute the orientation error
        #quat_error = point_distance(achieved_goal[3:], desired_goal[3:]) # to replace
        # compute the reward
        #return - pos_error - quat_error > - 0.5 # to replace

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal[:3]# - sites_offset[0]
        self.model.site_quat[site_id] = self.goal[3:]
        self._mujoco.mj_forward(self.model, self.data)
    
    def _sample_goal(self):
        #goal_pos = super()._sample_goal()
        goal_pos = np.array([0, 0, 1])
        # goal_quat = euler_to_quaternion(*random_euler_angles())
        possible_quats = [
            [0,0,1,0],
            [0,0.114,0.987,0.114],
            [0,0.114,0.987,-0.114],
            [0,-0.114,0.987,0.114],
            [0,-0.114,0.987,-0.114],
        ]
        goal_quat = random.choice(possible_quats)
        return np.concatenate((goal_pos, goal_quat))
    
    def _set_action(self, action):
        pass

    def compute_reward(self, achieved_goal, goal, info):
        return -1000


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


class FetchReachJointsContinuous(gym.Env):
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
                "distance": 2.3,
                "azimuth": 132.0,
                "elevation": -10.0,
                "lookat": np.array([0, 0, 1])
            },
            **kwargs
        )

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        
        self.record = record
        if self.record:
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path="./results/fetch_reach_joints_continuous/demo"
            )

        # action space: movements of the seven joints
        self.action_space = Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # observation space: joint positions, velocities, and target position
        self.observation_space = Dict({"observation": Box(-np.inf, np.inf, shape=(12,)), "desired_goal": Box(-np.inf, np.inf, shape=(7,))})

        # store initial distance for reward computation
        self.initial_distance = None
    
    def fix_obs(self, obs):
        obs["observation"] = np.array([])
        for i in range(6):
            obs["observation"] = np.append(obs["observation"], mujoco_utils.get_joint_qpos(self.env.model, self.env.data, self.joint_names[i])[0])
            obs["observation"] = np.append(obs["observation"], mujoco_utils.get_joint_qvel(self.env.model, self.env.data, self.joint_names[i])[0])

        obs.pop("achieved_goal")
        return obs
    
    def compute_reward(self, obs, Kp=1.0, Ko=0.25):
        print(self.env.data.mocap_pos[0])
        # compute the position error
        pos_error = point_distance(obs["desired_goal"][:3], self.env.data.mocap_pos[0])
        pos_reward = (self.initial_distance - pos_error) / self.initial_distance # [-inf, 1]
        
        # compute the orientation error
        quat_error = 1 - np.dot(obs["desired_goal"][3:], self.env.data.mocap_quat[0])
        quat_reward = 1 - quat_error / 2 # [0, 1]

        # compute the reward
        reward = Kp * pos_reward + Ko * quat_reward
        return reward

    def step(self, action):
        # guarantee that the action is within the action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # scale the action to a maximum movement of 0.1 radians in each joint
        action *= 0.1

        # update the joint positions
        current_joint_pos = np.array([])
        for i in range(6):
            current_joint_pos = np.append(current_joint_pos, mujoco_utils.get_joint_qpos(self.env.model, self.env.data, self.joint_names[i])[0])

        new_joint_pos = np.array(current_joint_pos) + action
        
        for i in range(6):
            mujoco_utils.set_joint_qpos(self.env.model, self.env.data, self.joint_names[i], new_joint_pos[i])
        
        mujoco_utils.reset_mocap2body_xpos(self.env.model, self.env.data)

        obs, reward, terminated, truncated, info = self.env.step(np.array([0, 0, 0, 0]))

        obs = self.fix_obs(obs)

        reward = self.compute_reward(obs)

        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()

            if info["is_success"]: # if the task is completed, capture more frames
                for _ in range(10):
                    self.video_recorder.capture_frame()

        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        start = [-0.0023272673236292007, -1.7968598804869593, 1.5711200873004358, -1.4860928815654297, -1.6272171179400843, -0.07619315782655889]
        #start = [0.01725006103515625, -1.9415461025633753, 1.8129728476153772, -1.5927173099913539, -1.5878670851336878, 0.03150486946105957]

        for i in range(6):
            mujoco_utils.set_joint_qpos(self.env.model, self.env.data, self.joint_names[i], start[i])

        self.render()

        obs = self.fix_obs(obs)

        self.initial_distance = point_distance(obs["desired_goal"][:3], self.env.data.mocap_pos[0])

        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()

        return obs, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        if self.record:
            self.video_recorder.close()

        self.env.close()

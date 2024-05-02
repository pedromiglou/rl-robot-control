#!/usr/bin/env python3

import gymnasium as gym
import numpy as np

from gymnasium.envs.registration import register
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium_robotics.utils import mujoco_utils
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv


def random_euler_angles():
    # Generate random angles for roll, pitch, and yaw
    roll = np.random.uniform(0, 2*np.pi)
    pitch = np.random.uniform(0, 2*np.pi)
    yaw = np.random.uniform(0, 2*np.pi)

    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.

    Parameters:
    - roll: Roll angle in radians.
    - pitch: Pitch angle in radians.
    - yaw: Yaw angle in radians.

    Returns:
    - quaternion: Tuple containing the quaternion (w, x, y, z).
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z


class CustomMujocoFetchReachEnv(MujocoFetchReachEnv):
    """
    This class is needed to overwrite functions in MujocoFetchReachEnv.
    """
    def _is_success(self, achieved_goal, desired_goal):
        # compute the position error
        pos_error = np.sum([(p1 - p2) ** 2 for p1, p2 in zip(achieved_goal, desired_goal)]) ** 0.5
        # compute the orientation error
        quat_error = np.sum([(q1 - q2) ** 2 for q1, q2 in zip(achieved_goal, desired_goal)]) ** 0.5
        # compute the reward
        return - pos_error - quat_error > - 0.5 

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)
    
    def _sample_goal(self):
        goal_pos = super()._sample_goal()
        goal_quat = euler_to_quaternion(*random_euler_angles())
        return np.concatenate([goal_pos, goal_quat])
    
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


class FetchReachJointsControl(gym.Env):
    """
    Gymanasium environment wrapper
    """
    def __init__(self, record=False, **kwargs):
        self.env = gym.make(
            'FetchReachDense-custom',
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

        self.joint_names = [
            "robot0:shoulder_pan_joint",
            "robot0:shoulder_lift_joint",
            "robot0:upperarm_roll_joint",
            "robot0:elbow_flex_joint",
            "robot0:forearm_roll_joint",
            "robot0:wrist_flex_joint",
            "robot0:wrist_roll_joint"
        ]
        
        self.record = record
        if self.record:
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path="./videos/discrete_fetch_reach_sac"
            )

        # action space: movements of the seven joints
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # observation space: joint positions, velocities, and target position
        self.observation_space = Dict({"observation": Box(-np.inf, np.inf, shape=(14,)), "desired_goal": Box(-np.inf, np.inf, shape=(7,))})

        # create a dict to store relevant info for the reward function
        self.reward_info = {}
    
    def fix_obs(self, obs):
        obs["observation"] = np.concatenate(mujoco_utils.robot_get_obs(self.env.model, self.env.data, self.joint_names))
        obs.pop("achieved_goal")
        return obs
    
    def compute_reward(self, obs, Kp=1.0, Ko=1.0):
        # compute the position error
        pos_error = np.sum([(p1 - p2) ** 2 for p1, p2 in zip(obs["desired_goal"][:3], self.env.data.mocap_pos)]) ** 0.5
        # compute the orientation error
        quat_error = np.sum([(q1 - q2) ** 2 for q1, q2 in zip(obs["desired_goal"][3:], self.env.data.mocap_quat)]) ** 0.5
        # compute the reward
        reward = -Kp * pos_error - Ko * quat_error
        return reward

    def step(self, action):
        # guarantee that the action is within the action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # scale the action to a maximum movement of 0.1 radians in each joint
        action *= 0.1

        # update the joint positions
        new_joint_pos = mujoco_utils.robot_get_obs(self.env.model, self.env.data, self.joint_names)[0] + action
        
        for i in range(7):
            mujoco_utils.set_joint_qpos(self.env.model, self.env.data, self.joint_names[i], new_joint_pos[i])
        
        mujoco_utils.reset_mocap2body_xpos(self.env.model, self.env.data)

        obs, reward, terminated, truncated, info = self.env.step(np.array([0, 0, 0, 0]))

        obs = self.fix_obs(obs)

        reward = self.compute_reward(obs)

        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()

        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs = self.fix_obs(obs)

        self.reward_info["max_pos_error"] = (sum((p1 - p2) ** 2 for p1, p2 in zip(obs["desired_goal"][:3], self.env.data.mocap_pos)))**0.5
        self.reward_info["max_quat_error"] = (sum((q1 - q2) ** 2 for q1, q2 in zip(obs["desired_goal"][3:], self.env.data.mocap_quat)))**0.5

        if self.record: # before returning, capture a frame if recording
            self.video_recorder.capture_frame()

        return obs, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        if self.record:
            self.video_recorder.close()

        self.env.close()

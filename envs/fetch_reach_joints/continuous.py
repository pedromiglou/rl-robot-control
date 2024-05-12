#!/usr/bin/env python3

import numpy as np
import os
import random

from gymnasium.envs.registration import register
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations

from utils import euler_to_quaternion, point_distance, random_euler_angles


class LarccEnv(MujocoRobotEnv, EzPickle):
    """Class for Larcc environment inspired by the Fetch environments."""

    def __init__(self, distance_threshold=0.05, **kwargs):
        # distance threshold for successful episode
        self.distance_threshold = distance_threshold

        # store initial distance for reward computation
        self.initial_distance = None

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        self.initial_joint_values = [
            -0.0023272673236292007,
            -1.7968598804869593,
            1.5711200873004358,
            -1.4860928815654297,
            -1.6272171179400843,
            -0.07619315782655889
        ]

        MujocoRobotEnv.__init__(
            self,
            model_path=os.path.join(os.path.dirname(__file__), "model/env.xml"),
            initial_qpos={ k: v for k,v in zip(self.joint_names, self.initial_joint_values) },
            n_actions=6,
            n_substeps=20,
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

        EzPickle.__init__(self, reward_type="dense", **kwargs)
    
    def get_eef(self):
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "grip_site"
        )

        # there should be a better way to get the quaternion
        return np.concatenate((self.data.site_xpos[site_id], self.data.mocap_quat[0]))

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info, Kp=1.0, Ko=0.25):
        # compute the position error
        pos_error = point_distance(goal[:3], achieved_goal[:3])
        pos_reward = (self.initial_distance - pos_error) / self.initial_distance # [-inf, 1]

        # compute the orientation error
        quat_error = 1 - np.dot(goal[3:], achieved_goal[3:])
        quat_reward = 1 - quat_error / 2 # [0, 1]

        # compute the reward
        return Kp * pos_reward + Ko * quat_reward

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope

        # scale the action to a maximum movement of 0.1 radians in each joint
        action *= 0.1

        # update the joint positions
        current_joint_pos = np.array([])
        for i in range(6):
            current_joint_pos = np.append(current_joint_pos, self._utils.get_joint_qpos(self.model, self.data, self.joint_names[i])[0])

        new_joint_pos = np.array(current_joint_pos) + action

        # apply action to simulation.
        for i in range(6):
           self._utils.set_joint_qpos(self.model, self.data, self.joint_names[i], new_joint_pos[i])

    def _get_obs(self):
        dt = self.n_substeps * self.model.opt.timestep

        # observation
        observation = np.array([])
        for i in range(6):
            observation = np.append(observation, self._utils.get_joint_qpos(self.model, self.data, self.joint_names[i])[0])
            observation = np.append(observation, self._utils.get_joint_qvel(self.model, self.data, self.joint_names[i])[0])

        # robot_qvel *= dt # to match mujoco velocity #TODO

        # achieved_goal
        achieved_goal = self.get_eef()

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal
        }

    def _sample_goal(self): #TODO rewrite everything
        #goal_pos = super()._sample_goal()
        goal_pos = np.array([0, 0, 1])
        self.initial_distance = point_distance(self.get_eef()[:3], goal_pos)

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

    def _is_success(self, achieved_goal, desired_goal):
        d = point_distance(achieved_goal[:3], desired_goal[:3])
        return (d < self.distance_threshold).astype(np.float32)

    def _step_callback(self):
        pass
        # function to apply additional constraints on the simulation
        # some constraint
        #....#

        # update
        # self._mujoco.mj_forward(self.model, self.data)    

    def _render_callback(self):
        # Visualize target.
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal[:3]# - sites_offset[0]
        self.model.site_quat[site_id] = self.goal[3:]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
           self.data.act[:] = None

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        for _ in range(10):
            self._mujoco_step(None)

    # executing _mujoco_step breaks the robot so the method is skipped
    def _mujoco_step(self, action):
       pass

# register the custom environment
register(
    # unique identifier for the env `name-version`
    id='Larcc',
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=LarccEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=50,
)

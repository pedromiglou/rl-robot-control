#!/usr/bin/env python3

import os
import sys
sys.path.append(f'{os.environ["HOME"]}/catkin_ws/src/rl-robot-control/model_training')

import numpy as np
import rospy

from geometry_msgs.msg import Pose
from stable_baselines3 import SAC
from std_msgs.msg import Float64MultiArray

from larcc_env.wrapped_env import WrappedEnv


class RLMovementPlanner:
    def __init__(self):
        self.env = WrappedEnv(max_episode_steps=50, render_mode="human")
        self.model = SAC.load(f'{os.environ["HOME"]}/catkin_ws/src/rl-robot-control/model_training/results/larcc_joints_continuous/best_model')

        self.planned_movement_publisher = rospy.Publisher("/planned_movement", Float64MultiArray, queue_size=1)
        self.goal_pose_subscriber = rospy.Subscriber("/goal_pose", Pose, self.goal_pose_callback)

    def goal_pose_callback(self, msg):
        goal = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        
        action_list = []
        observation, info = self.env.reset(goal=goal)
        for _ in range(50):
            action, _states = self.model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = self.env.step(action)
            action_list.append(action)
        
        self.env.close()

        self.planned_movement_publisher.publish(Float64MultiArray(
            data=np.sum(np.array(action_list), axis=0)*np.array([0.08, 0.08, 0.12, 0.12, 0.12, 0.12])))


def main():
    default_node_name = 'rl_movement_planner'
    rospy.init_node(default_node_name, anonymous=False)

    rl_movement_planner = RLMovementPlanner()

    rospy.spin()


if __name__ == '__main__':
    main()

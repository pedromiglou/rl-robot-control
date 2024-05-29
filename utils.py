#!/usr/bin/env python3

import numpy as np


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


def point_distance(point1, point2):
    """
    Compute the Euclidean distance between two points in n-dimensional space.
    
    Parameters:
        point1 (ndarray): NumPy array representing the coordinates of the first point.
        point2 (ndarray): NumPy array representing the coordinates of the second point.
        
    Returns:
        float: Euclidean distance between the two points.
    """
    return np.linalg.norm(point1 - point2)


def random_euler_angles():
    # Generate random angles for roll, pitch, and yaw
    roll = np.random.uniform(0, 2*np.pi)
    pitch = np.random.uniform(0, 2*np.pi)
    yaw = np.random.uniform(0, 2*np.pi)

    return roll, pitch, yaw


def quaternion_to_transformation_matrix(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w,0],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,0],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2,0],
        [0,0,0,1]
    ])
    return R

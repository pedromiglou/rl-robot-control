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

    print([roll, pitch, yaw])

    roll, pitch, yaw = (3.4570793849477655, 3.351624046629514, 3.682197036971132)

    #roll = -0.136
    #pitch = -1.089
    #yaw = 1.711

    return roll, pitch, yaw

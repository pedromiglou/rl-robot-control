#!/usr/bin/env python3

import math


def distance_between_points(point1, point2):
    """
    Calculate the Euclidean distance between two points in n-dimensional space.

    Parameters:
    - point1: Tuple containing coordinates of the first point.
    - point2: Tuple containing coordinates of the second point.

    Returns:
    - distance: Euclidean distance between the two points.
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensionality!")

    squared_distance = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    distance = math.sqrt(squared_distance)
    return distance

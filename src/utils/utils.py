"""
utility functions
"""


from typing import Tuple, Union, List

import numpy as np


def convert_coords(string: str):
    """
    take a string of numbers from a xml file and converts it to a numpy array
    :param string: string of coordinates
    :return:
    """
    points = [[int(num) for num in x.split(',')] for x in string.split()]

    return np.array(points)


def get_bbox(points: np.ndarray, corners: Union[None, List[int]] = None, tablebbox: Tuple[int, int, int, int] = None) -> \
Tuple[int, int, int, int]:
    """
    creates a bounding box around all given points
    :param points: np array of shape (N x 2) containing a list of points
    :param corners: corners can be defined if this is the case only the corner points are used for bb
    :param tablebbox: if given, bbox is calculated relative to table
    :return: x_min, y_min, x_max, y_max
    """
    if corners:
        points = points[corners]

    x_max, x_min = points[:, 0].max(), points[:, 0].min()
    y_max, y_min = points[:, 1].max(), points[:, 1].min()
    # swap 0 and 1 for tablebox if np.flip in extract_glosat_annotation
    if tablebbox:
        x_min -= tablebbox[0]
        y_min -= tablebbox[1]
        x_max -= tablebbox[0]
        y_max -= tablebbox[1]

    return x_min, y_min, x_max, y_max

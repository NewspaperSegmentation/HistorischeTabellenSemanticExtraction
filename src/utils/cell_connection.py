from typing import List

import numpy as np

from src.utils.utils import get_bbox


class Cell:
    def __init__(self, points: np.ndarray, corners: List[int]):
        corners = points[corners]
        y_sort = sorted(corners.tolist(), key=lambda point: point[1])
        x_sort = sorted(corners.tolist(), key=lambda point: point[0])
        self.left = y_sort[:2]
        self.right = y_sort[-2:]
        self.bottom = x_sort[:2]
        self.top = x_sort[-2:]

        self.corner_bb = get_bbox(points, corners)
        self.max_bb = get_bbox(points)


def row(cell_a, cell_b):
    if cell_a.right == cell_b.left:
        return True
    return False


def col(cell_a, cell_b):
    if cell_a.bottom == cell_b.top:
        return True
    return False

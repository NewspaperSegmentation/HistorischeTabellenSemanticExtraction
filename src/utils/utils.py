"""Utility functions."""

from typing import Tuple, Union, List

import numpy as np
import torch

from torchvision.utils import draw_bounding_boxes


def show_prediction(image: torch.Tensor,
                    pred: torch.Tensor,
                    target: Union[torch.Tensor, None] = None) -> torch.Tensor:
    """
    Visualize the predicted and ground truth on top of the image.

    Args:
        image: image that was predicted
        pred: predicted bounding boxes
        target: ground truth bounding boxes

    Returns:
        torch.Tensor: visualized image

    """
    _, y_size, x_size = image.shape
    boxes = []
    if target is not None:
        boxes.extend([box for box in target['boxes']])
    boxes.extend([box for box in pred])

    colors = ["green"] * len(target['boxes'])
    colors.extend(["red"] * len(pred))

    result = draw_bounding_boxes((image * 256).to(torch.uint8), torch.stack(boxes),
                                 colors=colors,
                                 width=5)

    return result


def convert_coords(string: str) -> np.ndarray:
    """
    Takes a string of coordinates from a xml file and converts it to a numpy array.

    Args:
        string: string of coordinates

    Returns:
        np.ndarray with coordinates

    """
    points = [[int(num) for num in x.split(',')] for x in string.split()]

    return np.array(points)


def get_bbox(points: np.ndarray,
             corners: Union[None, List[int]] = None,
             tablebbox: Tuple[int, int, int, int] = None) -> Tuple[int, int, int, int]:
    """
    Creates a bounding box around all given points.

    Args:
        points: np.ndarray of shape (N x 2) containing a list of points
        corners: corners can be defined if this is the case only the corner points are used for bb
        tablebbox: if given, bbox is calculated relative to table

    Returns:
        coordinates of bounding box in the format (x_min, y_min, x_max, y_max)

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

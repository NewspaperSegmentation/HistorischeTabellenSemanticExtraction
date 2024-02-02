"""Utility functions."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes


def get_image(image: torch.Tensor, boxes: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Draws bounding boxes on the given image and returns it as torch Tensor.

    Args:
        image: image to draw bounding boxes on
        boxes: Dict of bounding boxes to draw on the image.
               Keys are used as labels

    Returns:
            torch.Tensor of image with bounding boxes
    """
    colorplate = ["green", "red", "blue", "yellow"]

    colors = []
    labels = []
    coords = torch.zeros((0, 4))
    for idx, (label, item) in enumerate(boxes.items()):
        labels.extend([label] * len(item))
        colors.extend([colorplate[idx]] * len(item))
        coords = torch.vstack((coords, item))

    result = draw_bounding_boxes(
        (image * 256).to(torch.uint8), coords, colors=colors, width=5
    )

    return result


def plot_image(
    image: torch.Tensor,
    boxes: Dict[str, torch.Tensor],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot an image with given bounding boxes using pyplot.

    Args:
        image: image to plot
        boxes: dictionary with bounding boxes. Keys are labels.
        title: title of the plot
        save_path: path to save the plot as image
    """
    # create image with annotation
    image = get_image(image, boxes)

    # plot image
    plt.imshow(image)

    # add title if existing
    if title is not None:
        plt.title(title)

    # save images if save_as folder is given
    if save_path is not None:
        path = f"{Path(__file__).parent.absolute()}/../data/assets/images/{save_path}/"
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}{save_path}_{title}.png")

    # show plot
    plt.show()


def plot_annotations(folder: str, save_as: Optional[str] = None) -> None:
    """
    Shows target for model from given folder.

    Args:
        folder: path to folder of preprocessed
        save_as: name of the folder to save the images in (optional)
                 Images are saved in data/assets/images/
    """
    # List of bboxes for every property
    textlist = []
    has_textregion = False
    tablelist = []
    celllist = []
    rowlist = []
    collist = []
    imglist = []

    # iterate over files in folder and extract bboxes
    files = os.listdir(folder)
    for filename in sorted(files):
        if "textregions" in filename:
            textlist = torch.load(folder + filename)
            has_textregion = True

        if "table" in filename and filename.endswith(".jpg"):
            imglist.append(plt.imread(folder + filename))

        if "table" in filename and filename.endswith(".pt"):
            tablelist = torch.load(folder + filename)

        if "cell" in filename and filename.endswith(".pt"):
            celllist.append(torch.load(folder + filename))

        if "row" in filename and filename.endswith(".pt"):
            rowlist.append(torch.load(folder + filename))

        if "col" in filename and filename.endswith(".pt"):
            collist.append(torch.load(folder + filename))

    # plot tables
    plot_image(
        plt.imread(folder + sorted(files)[0]),
        tablelist,
        title="tables",
        save_path=save_as,
    )

    # plot tables and textregions if textregions exists
    if has_textregion:
        plot_image(
            plt.imread(folder + sorted(files)[0]),
            boxes={"tables": tablelist, "textregions": textlist},
            title="tables and textregions",
            save_path=save_as,
        )

    # plot cells, rows and columns
    for idx, img in enumerate(imglist):
        plot_image(
            img, boxes={"cells": celllist[idx]}, title="cells", save_path=save_as
        )
        plot_image(img, boxes={"rows": rowlist[idx]}, title="rows", save_path=save_as)
        plot_image(
            img, boxes={"columns": collist[idx]}, title="columns", save_path=save_as
        )


def convert_coords(string: str) -> np.ndarray:
    """
    Takes a string of coordinates from a xml file and converts it to a numpy array.

    Args:
        string: string of coordinates

    Returns:
        np.ndarray with coordinates

    """
    points = [[int(num) for num in x.split(",")] for x in string.split()]

    return np.array(points)


def get_bbox(
    points: np.ndarray,
    corners: Union[None, List[int]] = None,
    tablebbox: Tuple[int, int, int, int] = None,
) -> Tuple[int, int, int, int]:
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

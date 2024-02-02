"""Implementation of used metrics."""

from typing import Union, List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import torch
from torchvision.ops import box_iou


def calc_stats(
        pred: torch.Tensor,
        target: torch.Tensor,
        iou_thresholds: Optional[Union[float, torch.Tensor]] = None,
):
    """
    Calculates true positives, false positives, false negatives, precision, recall and f1.

    Args:
        pred: set of predicted boxes
        target: set of ground truth boxes
        iou_thresholds: IoU thresholds for calculating true- and false positives

    Returns:
        true positives, false positives. false negatives, precision, recall, f1
    """
    # ensure iou_thresholds is a tensor
    if iou_thresholds is None:
        iou_thresholds = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    elif isinstance(iou_thresholds, float):
        iou_thresholds = torch.tensor([iou_thresholds])

    matrix = box_iou(pred, target)
    n_pred, n_target = matrix.shape

    pred_iuo = matrix.amax(dim=1)
    target_iuo = matrix.amax(dim=0)

    mean_pred_iou = torch.mean(pred_iuo)
    mean_target_iuo = torch.mean(target_iuo)

    tp = torch.sum(
        pred_iuo.expand(len(iou_thresholds), n_pred) >= iou_thresholds[:, None], dim=1)
    fp = len(matrix) - tp
    fn = torch.sum(
        target_iuo.expand(len(iou_thresholds), n_target) < iou_thresholds[:, None], dim=1)

    return tp, fp, fn, mean_pred_iou, mean_target_iuo


def calc_metrics(tp: torch.Tensor,
                 fp: torch.Tensor,
                 fn: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcs precision, recall and f1 metric.

    Args:
        tp: Array of true positives on different IoU thresholds
        fp: Array of false positives on different IoU thresholds
        fn: Array of false negatives on different IoU thresholds

    Returns:
        precision, recall and f1
    """
    precision = tp / (tp + fp)
    precision = torch.nan_to_num(precision)

    recall = tp / (tp + fn)
    recall = torch.nan_to_num(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = torch.nan_to_num(f1)

    return precision, recall, f1


def weighted_f1(
        f1: torch.Tensor,
        iou_thresholds: Optional[Union[float, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Calculates f1 score weighted by IoU similar to the metric from GloSAT paper.

    Args:
        f1: Array of f1-scores on different IoU thresholds
        iou_thresholds: Array of the different IoU thresholds

    Returns:
        weighted F1 score
    """
    # ensure iou_thresholds is a tensor
    if iou_thresholds is None:
        iou_thresholds = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    elif isinstance(iou_thresholds, float):
        iou_thresholds = torch.tensor([iou_thresholds])

    if isinstance(iou_thresholds, float):
        iou_thresholds = torch.tensor([iou_thresholds])

    weighted_average = (f1 @ iou_thresholds) / iou_thresholds.sum()
    return weighted_average


def probabilities_ious(pred: Dict[str, torch.Tensor],
                       target: Dict[str, torch.Tensor],
                       ) -> Tuple[List[float], List[float]]:
    """
    Calculates the probability and the achieved IoU scores of the predicted bounding boxes.

    Args:
        pred: Array of predicted bounding boxes
        target: Array of ground truth bounding boxes

    Returns:
        List of probabilities and List of the achieved IoU
    """
    probabilities = list(pred["scores"])
    ious = list(box_iou(pred["boxes"], target["boxes"]).amax(dim=1))

    return probabilities, ious


def threshold_graph(
        probabilities: torch.Tensor,
        ious: torch.Tensor,
        name: str,
        path: str,
        iou_thresholds: Optional[Union[float, torch.Tensor]] = None,
) -> None:
    """
    Plots the true positives and false positives corresponding to the probability thresholds.

    Args:
        probabilities: Array of predicted bounding box probabilities
        ious: Array of the achieved IoU of the predicted bounding box
        name: Name of the model
        path: path to save the graphs ticks
        iou_thresholds: IoU thresholds used for calculating true and false positives. Defaults are
                        [0.5, 0.6, 0.7, 0.8, 0.9]
    """
    # ensure iou_thresholds is a tensor
    if iou_thresholds is None:
        iou_thresholds = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    elif isinstance(iou_thresholds, float):
        iou_thresholds = torch.tensor([iou_thresholds])

    sorted_probs, _ = torch.sort(probabilities)
    prob_thresholds = torch.unique(sorted_probs)

    true_positives = torch.zeros(len(prob_thresholds), len(iou_thresholds))
    false_positives = torch.zeros(len(prob_thresholds), len(iou_thresholds))

    for i, t in enumerate(prob_thresholds):
        mask = probabilities >= t
        data = ious[mask]
        data = data.expand(len(iou_thresholds), len(data))

        true_positives[i] = torch.sum(data >= iou_thresholds[:, None], dim=1)
        false_positives[i] = torch.sum(data < iou_thresholds[:, None], dim=1)

    for idx, iou_t in enumerate(iou_thresholds):
        plt.xlim(0, 1.05)
        plt.title(f"{name}: detection at {round(iou_t.item(), 1)} IoU Threshold")
        plt.plot(prob_thresholds, true_positives[:, idx], label="true positives")
        plt.plot(prob_thresholds, false_positives[:, idx], label="false positives")
        plt.xlabel("probability threshold")
        plt.ylabel("count")
        plt.legend()
        plt.savefig(f"{path}/threshold_graph_{round(iou_t.item() * 10)}.png")
        plt.clf()


if __name__ == "__main__":
    pred = torch.tensor([[10, 20, 30, 40],
                         [40, 50, 60, 70],
                         [45, 20, 65, 40],
                         [85, 25, 105, 55]])

    target = torch.tensor([[15, 25, 30, 40],
                           [45, 55, 65, 75],
                           [15, 55, 35, 70],
                           [85, 25, 104, 54]])

    wf1 = weighted_f1(pred, target)
    print(f"{wf1=}")

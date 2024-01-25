"""
implementation of used metrics
"""
from typing import Union

import torch
import matplotlib.pyplot as plt
from torchvision.ops import box_iou


def calc_stats(pred: torch.Tensor, target: torch.Tensor, threshold: Union[float, torch.Tensor] = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])):
    """
    calculates true positives, false positives, false negatives, precision, recall and f1 for the given
    thresholds on IoU
    :param matrix: matrix of IoUs between ground truth and prediction
    :param threshold: threshold for IoU
    :return: true positives, false positives. false negatives, precision, recall, f1
    """
    matrix = box_iou(pred, target)

    if isinstance(threshold, float):
        threshold = torch.tensor([threshold])

    pred_iuo = matrix.amax(dim=1).expand(len(threshold), len(matrix))
    target_iuo = matrix.amax(dim=0).expand(len(threshold), len(matrix))

    mean_pred_iou = torch.mean(pred_iuo)
    mean_target_iuo = torch.mean(target_iuo)

    tp = torch.sum(pred_iuo >= threshold[:, None],dim=1)
    fp = len(matrix) - tp
    fn = torch.sum(target_iuo < threshold[:, None], dim=1)
    # torch.sum(matrix.amax(dim=0) < threshold)

    return tp, fp, fn, mean_pred_iou, mean_target_iuo


def calc_metrics(tp, fp, fn):
    precision = tp / (tp + fp)
    precision = torch.nan_to_num(precision)

    recall = tp / (tp + fn)
    recall = torch.nan_to_num(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = torch.nan_to_num(f1)

    return precision, recall, f1


def weightedF1(f1, threshold: Union[float, torch.Tensor] = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])):
    """
    F1 score weighted by IoU similar to the metric from GloSAT paper
    :param pred: predicted bounding boxes
    :param target: ground truth bounding boxes
    :return: weighted F1 score
    """
    if isinstance(threshold, float):
        threshold = torch.tensor([threshold])

    weighted_average = (f1 @ threshold) / threshold.sum()
    return weighted_average



def main(pred, target):
    # add bboxes
    for box in pred:
        # with bounding box x being height and y being width (if np.flip in extract_glosat_annotation)
        ymin = box[1]
        xmin = box[0]
        ymax = box[3]
        xmax = box[2]
        xlist = [xmin, xmin, xmax, xmax, xmin]
        ylist = [ymin, ymax, ymax, ymin, ymin]
        plt.plot(xlist, ylist, color='red')

    # add bboxes
    for box in target:
        # with bounding box x being height and y being width (if np.flip in extract_glosat_annotation)
        ymin = box[1]
        xmin = box[0]
        ymax = box[3]
        xmax = box[2]
        xlist = [xmin, xmin, xmax, xmax, xmin]
        ylist = [ymin, ymax, ymax, ymin, ymin]
        plt.plot(xlist, ylist, color='green')

    plt.show()


if __name__ == '__main__':
    pred = torch.tensor([[10, 20, 30, 40],
                         [40, 50, 60, 70],
                         [45, 20, 65, 40],
                         [85, 25, 105, 55]])

    target = torch.tensor([[15, 25, 30, 40],
                           [45, 55, 65, 75],
                           [15, 55, 35, 70],
                           [85, 25, 104, 54]])

    main(pred, target)
    weightedF1(pred, target)


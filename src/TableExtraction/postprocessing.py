from typing import Dict

import torch
from torchvision.ops.boxes import box_area, _box_inter_union


def postprocess(prediction: Dict[str, torch.Tensor],
                threshold: float = 0.5,
                method: str = 'iom') -> Dict[str, torch.Tensor]:
    """
    Postprocesses the predicted boxes using a non maxima suppression for overlapping boxes
    :param prediction: output of the model
    :param threshold: threshold for overlap
    :param method:
    :return: boxes
    """
    if method not in ('iou', 'iom'):
        raise NameError(f"Method must be one of 'iou' or 'iom', got {method}!")

    boxes = prediction['boxes']
    scores = prediction['scores']

    # calc matrix of intersection depending on method
    inter, union = _box_inter_union(boxes, boxes)
    area = box_area(prediction['boxes'])
    min_matrix = torch.min(area.unsqueeze(1), area.unsqueeze(0))
    matrix: torch.Tensor = inter / min_matrix if method == 'iom' else inter/ union
    matrix.fill_diagonal_(0)

    # indices of intersections over threshold
    indices = (matrix > threshold).nonzero()

    # calc box indices to keep
    values = scores[indices]
    drop_indices = indices[torch.arange(len(indices)), torch.argmin(values, dim=1)]
    keep_indices = torch.tensor(list(set(range(len(scores))) - set(drop_indices.tolist())))

    # remove non maxima
    boxes = prediction['boxes'][keep_indices]
    scores = prediction['scores'][keep_indices]

    return {'boxes': boxes, 'scores': scores}


if __name__ == '__main__':
    import torch

    # Example vector and indices (as PyTorch tensors)
    vector = torch.tensor([5, 3, 8, 2, 7])
    indices = torch.tensor([[0, 1], [2, 4], [3, 0]])

    # Extract values from vector using indices
    values = vector[indices]

    # Find the index of the minimum value along the second axis (axis=1)
    min_index = torch.argmin(values, dim=1)

    # Get the original indices
    original_indices = indices[torch.arange(len(indices)), min_index]

    # Get positions not in original_indices
    not_in_original_indices = torch.tensor(list(set(range(len(vector))) - set(original_indices.tolist())))

    print("Positions of the vector not in original_indices:", not_in_original_indices)
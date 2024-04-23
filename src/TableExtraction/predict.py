"""Prediction for a single image."""

import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import torch
from skimage import io
from torchvision.models.detection import FasterRCNN, MaskRCNN

from src.TableExtraction.postprocessing import postprocess
from src.TableExtraction.trainer import get_model
from src.TableExtraction.utils.utils import draw_prediction


def predict(model: Union[FasterRCNN, MaskRCNN], image: torch.Tensor):
    """
    Predicts image with given model.

    Args:
        model: Model to use for prediction
        image: image to predict

    Returns:
        Dictionary with prediction ('bounding boxes', 'scores' and ('masks'))
        and a visualisation of the prediction
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    image = image.to(device)
    pred = model([image])
    pred = {k: v.detach().cpu() for k, v in pred[0].items()}
    pred = postprocess(pred, method='iom', threshold=.6)

    result = draw_prediction(image.detach().cpu(), pred)

    return pred, result


def get_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser(description="evaluation")

    parser.add_argument(
        "--objective",
        "-o",
        type=str,
        default="Table",
        help="Objective of the model",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Name of the model-file to evaluate",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    model = get_model(args.objective, args.model)

    path = (f"{Path(__file__).parent.absolute()}/../../data/Newspaper/valid/"
            f"Koelnische Zeitung 1866.06-1866.09 - 0179/region_10/image.jpg")
    image = torch.tensor(io.imread(path)).permute(2, 0, 1).float()
    print(f"{image.shape=}")

    pred, visualisation = predict(model, image)

    plt.imshow(visualisation.permute(1, 2, 0))
    plt.savefig('../../data/MaskRCNNPredictionExample4.png', dpi=1000)

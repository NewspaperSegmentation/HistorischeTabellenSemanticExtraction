import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import FasterRCNN

from src.TableExtraction.postprocessing import postprocess
from src.TableExtraction.trainer import get_model
from src.TableExtraction.utils.utils import get_image


def predict(model: FasterRCNN, image: torch.Tensor):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    image = image.to(device)
    output = model([image])
    output = {k: v.detach().cpu() for k, v in output[0].items()}
    output = postprocess(output)

    result = get_image(image.detach().cpu(), output)

    return output, result


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
    from src.TableExtraction.utils.post_utils import row_col_estimate

    args = get_args()
    print(args)

    model = get_model(args.objective, args.model)

    image = torch.load((f"{Path(__file__).parent.absolute()}/../../data/BonnData/valid/"
                        f"I_HA_Rep_89_Nr_16160_0212/I_HA_Rep_89_Nr_16160_0212_table_0.pt")) / 256

    print(f"{image.shape=}")

    output, result = predict(model, image)

    cols, rows = row_col_estimate(output["boxes"])

    plt.imshow(result.permute(1, 2, 0))
    plt.vlines(x=rows, ymin=0, ymax=result.shape[1], colors='red', linestyles='-', linewidth=.5)
    plt.hlines(y=cols, xmin=0, xmax=result.shape[2], colors='red', linestyles='-', linewidth=.5)
    plt.show()
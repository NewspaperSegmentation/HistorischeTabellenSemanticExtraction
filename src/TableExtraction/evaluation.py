"""Evaluates the model performance on a test dataset."""
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.TableExtraction.customdataset import CustomDataset
from src.TableExtraction.trainer import get_model
from src.TableExtraction.utils.metrics import (
    calc_metrics,
    calc_stats,
    probabilities_ious,
    threshold_graph,
    weighted_f1,
)
from src.TableExtraction.utils.utils import get_image


def evaluation(
        model: torch.nn.Module, dataset: CustomDataset, name: str, cuda: int = 0
) -> None:
    """
    Evaluates the given model on the given dataset.

    Args:
        model: model to test
        dataset: (test) dataset to evaluate on
        name: name of the folder to save the results
        cuda: number of cuda device to use
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda}")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    df = pd.DataFrame(
        columns=["image_number", "mean_pred_iou", "mean_target_iuo", "wf1"], index=[0]
    )

    os.makedirs(
        f"{Path(__file__).parent.absolute()}/../../logs/evaluation/" f"{name}/",
        exist_ok=True,
    )

    all_tp = torch.zeros(5)
    all_fp = torch.zeros(5)
    all_fn = torch.zeros(5)

    probabilities = []
    ious_list = []

    idx = 0
    for img, target in tqdm(dataloader, desc="evaluation"):
        img = img.to(device)

        output = model([img[0]])
        output = {k: v.detach().cpu() for k, v in output[0].items()}
        target = {k: v[0] for k, v in target.items()}

        boxes = {"prediction": output["boxes"], "ground truth": target["boxes"]}
        result = get_image(img.detach().cpu(), boxes)

        result_image = Image.fromarray(result.permute(1, 2, 0).numpy())
        result_image.save(
            f"{Path(__file__).parent.absolute()}/../../logs/evaluation/"
            f"{name}/{idx}_{target['img_number']}.png"
        )

        prob, ious = probabilities_ious(output, target)
        probabilities.extend(prob)
        ious_list.extend(ious)

        tp, fp, fn, mean_pred_iou, mean_target_iuo = calc_stats(
            output["boxes"], target["boxes"]
        )
        precision, recall, f1 = calc_metrics(tp, fp, fn)
        wf1 = weighted_f1(f1)

        all_tp += tp
        all_fp += fp
        all_fn += fn

        metrics = {
            "image_number": target["img_number"],
            "mean_pred_iou": mean_pred_iou.item(),
            "mean_target_iuo": mean_target_iuo.item(),
            "wf1": wf1.item(),
            "prediction_count": len(output),
        }

        iuos = [9, 8, 7, 6, 5]
        metrics.update({f"tp_{iuos[i]}": list(tp)[i].item() for i in range(5)})
        metrics.update({f"fp_{iuos[i]}": list(fp)[i].item() for i in range(5)})
        metrics.update({f"fn_{iuos[i]}": list(fn)[i].item() for i in range(5)})
        metrics.update(
            {f"precision_{iuos[i]}": list(precision)[i].item() for i in range(5)}
        )
        metrics.update({f"recall_{iuos[i]}": list(recall)[i].item() for i in range(5)})
        metrics.update({f"f1_{iuos[i]}": list(f1)[i].item() for i in range(5)})

        df = pd.concat([df, pd.DataFrame(metrics, index=[0])])

        idx += 1

    all_precision, all_recall, all_f1 = calc_metrics(all_tp, all_fp, all_fn)
    all_wf1 = weighted_f1(all_f1)

    threshold_graph(
        torch.tensor(probabilities),
        torch.tensor(ious_list),
        name,
        f"{Path(__file__).parent.absolute()}/../../logs/evaluation/{name}/",
    )

    with open(
            f"{Path(__file__).parent.absolute()}/../../logs/evaluation/"
            f"{name}/{name}_overview.txt",
            "w",
    ) as f:
        f.write(f"true positives: {[x.item() for x in list(all_tp)]}\n")
        f.write(f"false positives: {[x.item() for x in list(all_fp)]}\n")
        f.write(f"false negatives: {[x.item() for x in list(all_fn)]}\n")
        f.write(f"precision: {[x.item() for x in list(all_precision)]}\n")
        f.write(f"recall: {[x.item() for x in list(all_recall)]}\n")
        f.write(f"F1 score: {[x.item() for x in list(all_f1)]}\n")
        f.write(f"weighted F1 score: {all_wf1=}\n")

    df.to_csv(
        f"{Path(__file__).parent.absolute()}/../../logs/evaluation/" f"{name}/{name}.csv"
    )


def get_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser(description="evaluation")

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="BonnData",
        help="Name of the dataset to evaluate on",
    )

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

    testdataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/" f"../../data/{args.dataset}/test",
        args.objective
    )

    print(f"{len(testdataset)=}")

    evaluation(model, testdataset, name=args.model)

"""Module to train models on table datasets."""

import os
from pathlib import Path
import argparse
from typing import Optional, Union

import numpy as np
import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from tqdm import tqdm

from src.TableExtraction.customdataset import CustomDataset
from src.TableExtraction.utils.utils import get_image

LR = 0.00001


class Trainer:
    """Class to train models."""

    def __init__(
            self,
            model: FasterRCNN,
            traindataset: CustomDataset,
            testdataset: CustomDataset,
            optimizer: Optimizer,
            name: str,
            cuda: int = 0,
    ) -> None:
        """
        Trainer class to train models.

        Args:
            model: model to train
            traindataset: dataset to train on
            testdataset: dataset to validate model while trainings process
            optimizer: optimizer to use
            name: name of the model in save-files and tensorboard
            cuda: number of used cuda device
        """
        self.device = (
            torch.device(f"cuda:{cuda}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"using {self.device}")

        self.model = model.to(self.device)
        self.optimizer = optimizer

        self.trainloader = DataLoader(
            traindataset, batch_size=1, shuffle=False, num_workers=0
        )
        self.testloader = DataLoader(
            testdataset, batch_size=1, shuffle=False, num_workers=0
        )

        self.bestavrgloss: Union[float, None] = None
        self.epoch = 0
        self.name = name

        # setup tensor board
        train_log_dir = f"{Path(__file__).parent.absolute()}/../logs/runs/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

        self.example_image, self.example_target = testdataset[0]
        self.train_example_image, self.train_example_target = traindataset[0]

    def save(self, name: str = "") -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs(f"{Path(__file__).parent.absolute()}/../models/", exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"{Path(__file__).parent.absolute()}/../models/{name}",
        )

    def load(self, name: str = "") -> None:
        """
        Load the given model.

        Args:
            name: name of the model
        """
        self.model.load_state_dict(
            torch.load(f"{Path(__file__).parent.absolute()}/../models/{name}.pt")
        )

    def train(self, epoch: int) -> None:
        """
        Train model for given number of epochs.

        Args:
            epoch: number of epochs
        """
        for self.epoch in range(1, epoch + 1):
            print(f"start epoch {self.epoch}:")
            self.train_epoch()
            avgloss = self.valid()

            # early stopping
            if self.bestavrgloss is None or self.bestavrgloss > avgloss:
                self.bestavrgloss = avgloss
                self.save(f"{self.name}_es.pt")

        # save model after training
        self.save(f"{self.name}_end.pt")

    def train_epoch(self) -> None:
        """Trains one epoch."""
        loss_lst = []
        loss_classifier_lst = []
        loss_box_reg_lst = []
        loss_objectness_lst = []
        loss_rpn_box_reg_lst = []

        for img, target in tqdm(self.trainloader, desc="training"):
            img = img.to(self.device)
            target["boxes"] = target["boxes"][0].to(self.device)
            target["labels"] = target["labels"][0].to(self.device)

            self.optimizer.zero_grad()
            output = model([img[0]], [target])
            loss = sum(v for v in output.values())
            loss.backward()
            self.optimizer.step()

            loss_lst.append(loss.detach().cpu().item())
            loss_classifier_lst.append(output["loss_classifier"].detach().cpu().item())
            loss_box_reg_lst.append(output["loss_box_reg"].detach().cpu().item())
            loss_objectness_lst.append(output["loss_objectness"].detach().cpu().item())
            loss_rpn_box_reg_lst.append(
                output["loss_rpn_box_reg"].detach().cpu().item()
            )

            del img, target, output, loss

        # logging
        self.writer.add_scalar(
            "Training/loss",
            np.mean(loss_lst),
            global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Training/loss_classifier",
            np.mean(loss_classifier_lst),
            global_step=self.epoch,
        )  # type: ignore
        self.writer.add_scalar(
            "Training/loss_box_reg",
            np.mean(loss_box_reg_lst),
            global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Training/loss_objectness",
            np.mean(loss_objectness_lst),
            global_step=self.epoch,
        )  # type: ignore
        self.writer.add_scalar(
            "Training/loss_rpn_box_reg",
            np.mean(loss_rpn_box_reg_lst),
            global_step=self.epoch,
        )  # type: ignore
        self.writer.flush()  # type: ignore

        del (
            loss_lst,
            loss_classifier_lst,
            loss_box_reg_lst,
            loss_objectness_lst,
            loss_rpn_box_reg_lst,
        )

    def valid(self) -> float:
        """
        Validates current model on validation set.

        Returns:
            current loss
        """
        loss = []
        loss_classifier = []
        loss_box_reg = []
        loss_objectness = []
        loss_rpn_box_reg = []

        for img, target in tqdm(self.testloader, desc="validation"):
            img = img.to(self.device)
            target["boxes"] = target["boxes"][0].to(self.device)
            target["labels"] = target["labels"][0].to(self.device)

            output = self.model([img[0]], [target])

            loss.append(sum(v for v in output.values()).cpu().detach())
            loss_classifier.append(output["loss_classifier"].detach().cpu().item())
            loss_box_reg.append(output["loss_box_reg"].detach().cpu().item())
            loss_objectness.append(output["loss_objectness"].detach().cpu().item())
            loss_rpn_box_reg.append(output["loss_rpn_box_reg"].detach().cpu().item())

            del img, target, output

        meanloss = np.mean(loss)

        # logging
        self.writer.add_scalar(
            "Valid/loss",
            meanloss,
            global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Valid/loss_classifier",
            np.mean(loss_classifier),
            global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Valid/loss_box_reg",
            np.mean(loss_box_reg),
            global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Valid/loss_objectness",
            np.mean(loss_objectness),
            global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Valid/loss_rpn_box_reg",
            np.mean(loss_rpn_box_reg),
            global_step=self.epoch
        )  # type: ignore
        self.writer.flush()  # type: ignore

        self.model.eval()

        # predict example form training set
        pred = self.model([self.train_example_image.to(self.device)])
        boxes = {
            "ground truth": self.train_example_target["boxes"],
            "prediction": pred[0]["boxes"].detach().cpu(),
        }
        result = get_image(self.train_example_image, boxes)
        self.writer.add_image(
            "Training/example", result[:, ::2, ::2], global_step=self.epoch
        )  # type: ignore

        # predict example form validation set
        pred = self.model([self.example_image.to(self.device)])
        boxes = {
            "ground truth": self.example_target["boxes"],
            "prediction": pred[0]["boxes"].detach().cpu(),
        }
        result = get_image(self.example_image, boxes)
        self.writer.add_image(
            "Valid/example", result[:, ::2, ::2], global_step=self.epoch
        )  # type: ignore

        self.model.train()

        return meanloss


def get_model(objective: str, load_weights: Optional[str] = None) -> FasterRCNN:
    """
    Creates a FasterRCNN model for training, using the specified objective parameter.

    Args:
        objective: objective of the model (should be 'tables', 'cell', 'row' or 'col')
        load_weights: name of the model to load

    Returns:
        FasterRCNN model
    """
    params = {
        "tables": {"box_detections_per_img": 10},
        "cell": {"box_detections_per_img": 200},
        "row": {"box_detections_per_img": 100},
        "col": {"box_detections_per_img": 100},
    }

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **params[objective]
    )

    if load_weights:
        model.load_state_dict(
            torch.load(
                f"{Path(__file__).parent.absolute()}/../models/" f"{load_weights}.pt"
            )
        )

    return model


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="preprocess")

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="model",
        help="Name of the model, for saving and logging",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=250,
        help="Number of epochs",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="BonnData",
        help="which dataset should be used for training",
    )

    parser.add_argument(
        "--objective",
        "-o",
        type=str,
        default="table",
        help="objective of the model ('table', 'cell', 'row' or 'col')",
    )

    parser.add_argument('--augmentations', action=argparse.BooleanOptionalAction)
    parser.set_defaults(augmentations=False)

    return parser.parse_args()


if __name__ == "__main__":
    from torchvision import transforms

    args = get_args()

    # check args
    if args.name == 'model':
        raise ValueError("Please enter a valid model name!")

    if args.objective not in ['table', 'col', 'row', 'cell']:
        raise ValueError("Please enter a valid objective must be 'table', 'col', 'row' or 'cell'!")

    if args.dataset not in ['BonnData', 'GloSAT']:
        raise ValueError("Please enter a valid dataset must be 'BonnData' or 'GloSAT'!")

    if args.epochs <= 0:
        raise ValueError("Please enter a valid number of epochs must be >= 0!")

    name = (f"{args.name}_{Dataset}_{args.objective}"
            f"{'_aug' if args.augmentations else ''}_e{args.epochs}")
    model = get_model(args.objective)

    transform = None
    if args.augmentatios:
        transform = torch.nn.Sequential(
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [transforms.ColorJitter(brightness=(0.5, 1.5), saturation=(0, 2))]
                ),
                p=0.1,
            ),
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [transforms.GaussianBlur(kernel_size=9, sigma=(2, 10))]
                ),
                p=0.1,
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            transforms.RandomGrayscale(p=0.1))


    traindataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../data/{args.dataset}/train/",
        objective,
        transforms=transform,
    )

    validdataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../data/{args.dataset}/valid/", objective
    )

    print(f"{len(traindataset)=}")
    print(f"{len(validdataset)=}")

    optimizer = AdamW(model.parameters(), lr=LR)

    trainer = Trainer(model, traindataset, validdataset, optimizer, name)
    trainer.train(args.epochs)

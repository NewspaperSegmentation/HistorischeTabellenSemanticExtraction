"""Module to train baseline models."""

import os
from pathlib import Path
import argparse
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm

from monai.losses import DiceLoss
from monai.networks.nets import BasicUNet

from src.TableExtraction.dataset_baseline import CustomDataset as BaselineDataset

LR = 0.00001


class CELoss(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super(CELoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        pred = self.softmax(pred)
        return self.ce_loss(pred, target.squeeze(dim=1))


class Trainer:
    """Class to train models."""

    def __init__(
            self,
            model: nn.Module,
            traindataset: BaselineDataset,
            testdataset: BaselineDataset,
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
            mask_prediction: Set True if you want to get masks predicted
            cuda: number of used cuda device
        """
        print(f"{torch.cuda.is_available()=}")
        self.device = (
            torch.device(f"cuda:{cuda}")
            if torch.cuda.is_available() and cuda >= 0
            else torch.device("cpu")
        )
        print(f"using {self.device}")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = CELoss(weight=torch.tensor([0.001, 1.0]).to(self.device))
        # DiceLoss(include_background=False,
        #          to_onehot_y=True,
        #          softmax=True)

        self.trainloader = DataLoader(
            traindataset, batch_size=32, shuffle=True, num_workers=8
        )
        self.testloader = DataLoader(
            testdataset, batch_size=1, shuffle=False, num_workers=8
        )

        self.bestavrgloss: Union[float, None] = None
        self.epoch = 0
        self.name = name

        # setup tensor board
        train_log_dir = f"{Path(__file__).parent.absolute()}/../../logs/runs/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

        self.example_image, self.example_target = testdataset[2]
        self.train_example_image, self.train_example_target = traindataset[0]

    def save(self, name: str = "") -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs(f"{Path(__file__).parent.absolute()}/../../models/", exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"{Path(__file__).parent.absolute()}/../../models/{name}",
        )

    def load(self, name: str = "") -> None:
        """
        Load the given model.

        Args:
            name: name of the model
        """
        self.model.load_state_dict(
            torch.load(f"{Path(__file__).parent.absolute()}/../../models/{name}.pt")
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

        for images, targets in tqdm(self.trainloader, desc="training"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            output = model(images)
            loss = self.loss_fn(output, targets)
            loss.backward()
            self.optimizer.step()

            loss_lst.append(loss.detach().cpu().item())

            del images, targets, output, loss

        self.log_loss('Training', loss=np.mean(loss_lst))

        del loss_lst

    def valid(self) -> float:
        """
        Validates current model on validation set.

        Returns:
            current loss
        """
        loss_lst = []

        for images, targets in tqdm(self.testloader, desc="validation"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            output = model(images)
            loss = self.loss_fn(output, targets)

            loss_lst.append(loss.cpu().detach())

            del images, targets, output, loss

        self.log_loss('Valid', loss=np.mean(loss_lst))

        self.log_examples('Training')
        self.log_examples('Valid')

        return np.mean(loss_lst)

    def log_examples(self, dataset: str):
        """Predicts and logs a example image form the training- and from the validation set."""
        self.model.eval()

        example = self.train_example_image if dataset == 'Training' else self.example_image

        # predict example form training set
        pred = self.model(example[None].to(self.device))

        result = draw_segmentation_masks(image=example,
                                         masks=pred[0, 1] > 0.5,
                                         alpha=0.5,
                                         colors='red')

        # log in tensorboard
        self.writer.add_image(
            f"{dataset}/example",
            result[:, ::2, ::2],
            global_step=self.epoch
        )  # type: ignore

        self.model.train()

    def log_loss(self, dataset: str, loss: float):
        """
        Logs the loss values to tensorboard.

        Args:
            dataset: Name of the dataset the loss comes from ('Training' or 'Valid')
            loss: average over all loss

        """
        # logging
        self.writer.add_scalar(
            f"{dataset}/loss",
            loss,
            global_step=self.epoch
        )  # type: ignore

        self.writer.flush()  # type: ignore


def get_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser(description="training")

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="test",
        help="Name of the model, for saving and logging",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=2,
        help="Number of epochs",
    )

    parser.add_argument(
        "--load",
        "-l",
        type=str,
        default=None,
        help="name of a model to load",
    )

    parser.add_argument(
        "--cuda",
        "-c",
        type=int,
        default=-1,
        help="number of the cuda device (use -1 for cpu)",
    )

    parser.add_argument('--augmentations', "-a", action=argparse.BooleanOptionalAction)
    parser.set_defaults(augmentations=False)

    return parser.parse_args()


if __name__ == "__main__":
    from torchvision import transforms

    args = get_args()

    # check args
    if args.name == 'model':
        raise ValueError("Please enter a valid model name!")

    if args.epochs <= 0:
        raise ValueError("Please enter a valid number of epochs must be >= 0!")

    print(f'start training:\n'
          f'\tname: {args.name}\n'
          f'\tepochs: {args.epochs}\n'
          f'\tload: {args.load}\n'
          f'\tcuda: {args.cuda}\n')

    name = (f"{args.name}_baseline"
            f"{'_aug' if args.augmentations else ''}_e{args.epochs}")
    model = BasicUNet(spatial_dims=2, in_channels=3, out_channels=2)

    transform = None
    if args.augmentations:
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

    traindataset: Dataset = BaselineDataset(
        f"{Path(__file__).parent.absolute()}/../../data/Newspaper/train/",
        augmentations=transform,
    )

    validdataset: Dataset = BaselineDataset(
        f"{Path(__file__).parent.absolute()}/../../data/Newspaper/valid/",
        cropping=False
    )

    print(f"{len(traindataset)=}")
    print(f"{len(validdataset)=}")

    optimizer = AdamW(model.parameters(), lr=LR)

    trainer = Trainer(model,
                      traindataset,
                      validdataset,
                      optimizer,
                      name,
                      cuda=args.cuda)
    trainer.train(args.epochs)

"""Newspaper Class for newspaper mask R-CNN."""

import glob
from typing import Tuple

import numpy as np
import torch
from torch.nn import Sequential, ModuleList
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage import io
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):  # type: ignore
    """Newspaper Class for training."""

    def __init__(self, path: str,
                 augmentations: Module = None,
                 cropping: bool = True) -> None:
        """
        Newspaper Class for training.

        Args:
            path: path to folder with images
            transformation: torchvision transforms for on-the-fly augmentations
        """
        super().__init__()
        self.data = [x for x in glob.glob(f"{path}/*/*")]
        self.cropping = cropping
        self.crop = RandomCropAndResize(size=(256, 256))
        self.augmentations = augmentations

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns image and target (boxes, labels, img_number) from dataset.

        Args:
            index: index of datapoint

        Returns:
            image, target

        """

        image = torch.tensor(io.imread(f"{self.data[index]}/image.jpg")).permute(2, 0, 1) / 256
        target = torch.tensor(np.load(f"{self.data[index]}/baselines.npy")).float()[None]

        width_pad = max(256 - image.shape[1], 0)
        height_pad = max(256 - image.shape[2], 0)
        image = F.pad(image, (0, height_pad, 0, width_pad))
        target = F.pad(target, (0, height_pad, 0, width_pad))

        if self.cropping:
            image, target = self.crop(image, target)

        if self.augmentations:
            image = self.augmentations(image)

        return image, target.long()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            length of the dataset

        """
        return len(self.data)


class RandomCropAndResize(Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, image: torch.Tensor, target: torch.Tensor):

        # _, width, height = image.shape
        # kernel_size = np.random.choice([4])
        # if kernel_size > 1:
        #     resize = transforms.Resize((width // kernel_size, height // kernel_size))
        #     image = resize(image)
        #     target = F.max_pool2d(target, kernel_size)

        # Randomly crop the image and mask
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)
        image = transforms.functional.crop(image, i, j, h, w)
        target = transforms.functional.crop(target, i, j, h, w)
        return image, target


if __name__ == "__main__":
    augmentations = Sequential(
        transforms.RandomApply(
            ModuleList(
                [transforms.ColorJitter(brightness=(0.5, 1.5), saturation=(0, 2))]
            ),
            p=0,
        ),
        transforms.RandomApply(
            ModuleList(
                [transforms.GaussianBlur(kernel_size=9, sigma=(2, 10))]
            ),
            p=0,
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0),
        transforms.RandomGrayscale(p=1),
    )

    dataset = CustomDataset('../../data/Newspaper/train', cropping=False)
    print(len(dataset))

    image, target = dataset[0]
    print(f"{image.shape=}, {target.shape=}")

    plt.imshow(image.permute(1, 2, 0))
    plt.imshow(target[0], alpha=0.5, cmap='gray')
    plt.show()

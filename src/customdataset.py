from PIL import Image
from torch.utils.data import Dataset
import glob
import torch
import os

from pathlib import Path


class CustomDataset(Dataset):
    def __init__(self, path, objective, transforms=None) -> None:
        super().__init__()
        if objective == "tables":
            self.data = list(glob.glob(f"{path}/*"))
        else:
            self.data = list(glob.glob(f"{path}/*/*_table_*.pt"))
        self.objective = objective
        self.transforms = transforms

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        if self.objective == "tables":
            imgnum = self.data[index].split(os.sep)[-1]
            img = torch.load(f"{self.data[index]}/{imgnum}.pt") / 256
            target = torch.load(f"{self.data[index]}/{imgnum}_tables.pt")
        else:
            imgnum = self.data[index].split(os.sep)[-2]
            tablenum = self.data[index].split(os.sep)[-1].split("_")[-1][-4]
            img = torch.load(self.data[index]) / 256
            target = torch.load(f"{'/'.join(self.data[index].split(os.sep)[:-1])}/{imgnum}_{self.objective}_{tablenum}.pt")
        if self.transforms:
            img = self.transforms(img)
        return img, {'boxes': target, 'labels': torch.ones(len(target), dtype=torch.int64), 'img_number': imgnum}

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    import numpy as np
    from torchvision import transforms

    transform = torch.nn.Sequential(
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=(0.5,1.5), saturation=(0,2))]), p=0),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=9, sigma=(2,10))]), p=0),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0),
        transforms.RandomGrayscale(p=1)
    )

    dataset = CustomDataset(f'{Path(__file__).parent.absolute()}/../data/GloSAT/train', 'tables', transforms=transform)
    img, target = dataset[3]

    result = Image.fromarray((img.permute(1, 2, 0).numpy()*255).astype(np.uint8))
    result.save(f'{Path(__file__).parent.absolute()}/../data/assets/Originals_SampleImage.png')

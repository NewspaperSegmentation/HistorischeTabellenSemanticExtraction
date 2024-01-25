from torch.utils.data import Dataset
import glob
import torch
import os

from pathlib import Path


class CustomDataset(Dataset):
    def __init__(self, path, objective) -> None:
        super().__init__()
        if objective == "tables":
            self.data = list(glob.glob(f"{path}/*"))
        else:
            self.data = list(glob.glob(f"{path}/*/*_table_*.pt"))
        self.objective = objective

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
        return img, {'boxes': target, 'labels': torch.ones(len(target), dtype=torch.int64), 'img_number': imgnum}

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = CustomDataset(f'{Path(__file__).parent.absolute()}/../data/GloSAT/test', 'tables')
    img, target = dataset[0]

    print(target)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

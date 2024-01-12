from torch.utils.data import Dataset
import glob
import torch
import os

class dataset(Dataset):
    def __init__(self, path, objective) -> None:
        super().__init__()
        if objective=="tables":
            self.data = list(glob.glob(f"{path}/*"))
        else:
            self.data = list(glob.glob(f"{path}/*/*_table_*"))
        self.objective = objective

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        #return super().__getitem__(index)
        return torch.randn(3, 1600, 1000), torch.rand((11, 4))
        if self.objective=="tables":
            imgnum = self.data[index].split(os.sep)[-1]
            img = torch.load(f"{self.data[index]}/{imgnum}.jpg")
            target = torch.load(f"{self.data[index]}/{imgnum}_tables.txt")
        else:
            imgnum = self.data[index].split(os.sep)[-2]
            tablenum = self.data[index].split(os.sep)[-1].split("_")[-1][-5]
            img = torch.load(self.data[index])
            target = torch.load(f"{self.data[index].split(os.sep)[:-1]}/{imgnum}_{self.objective}_{tablenum}.txt")
        return img, target

    def __len__(self) -> int:
        return len(self.data)

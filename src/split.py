"""Splits data into training, validation and test split."""

import os
import random
import glob
import shutil
from tqdm import tqdm
from typing import Tuple, List


def copy(folder: str, split: List[str], name: str) -> None:
    """
    Copies the files form a given split to a new folder.

    Args:
        folder: folder to copy the files form
        split: list of files to copy
        name: name of the new folder (should be 'train', 'valid', 'test')
    """
    os.makedirs(f'{folder}/../{name}')

    for f in tqdm(split, desc='copying'):
        numb = f.split(os.sep)[-1]
        shutil.copytree(f, f'{folder}/../{name}/{numb}')


def split_dataset(folder: str, split: Tuple[float, float, float]) -> None:
    """
    Splits the dataset into training, validation and test split.

    Args:
        folder: folder with preprocessed dataset
        split: Tuple with partitions of splits (should sum up to 1)
    """
    # seed the process for reproducibility
    random.seed(42)

    # get folder with datapoints form preprocessed dataset
    files = glob.glob(f"{folder}/*")

    # shuffle
    shuffled_list = random.sample(files, len(files))

    # calc split indices
    n = len(files)
    s1, s2 = int(n * split[0]), int(n * (split[0] + split[1]))

    # create splits
    train = shuffled_list[:s1]
    valid = shuffled_list[s1:s2]
    test = shuffled_list[s2:]

    print(f"{len(train)=}")
    print(f"{len(valid)=}")
    print(f"{len(test)=}")

    # copy data in 3 new folder
    copy(folder, train, "train")
    copy(folder, valid, "valid")
    copy(folder, test, "test")


if __name__ == '__main__':
    split_dataset("../data/Tables/preprocessed/", (.8, .1, .1))

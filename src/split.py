import os
import random
import glob
import shutil
from tqdm import tqdm
from typing import Tuple


def copy(folder, split, name):
    os.makedirs(f'{folder}/../{name}')

    for f in tqdm(split, desc='copying'):
        numb = f.split(os.sep)[-1]
        shutil.copytree(f, f'{folder}/../{name}/{numb}')


def split(folder: str, split: Tuple[float, float, float]):
    seed_value = 42
    random.seed(seed_value)

    files = glob.glob(f"{folder}/*")
    shuffled_list = random.sample(files, len(files))

    n = len(files)
    s1, s2 = int(n * split[0]), int(n * (split[0] + split[1]))

    train = shuffled_list[:s1]
    valid = shuffled_list[s1:s2]
    test = shuffled_list[s2:]

    print(f"{len(train)=}")
    print(f"{len(valid)=}")
    print(f"{len(test)=}")

    copy(folder, train, "train")
    copy(folder, valid, "valid")
    copy(folder, test, "test")


if __name__ == '__main__':
    split("../data/Tables/preprocessed/", (.8, .1, .1))
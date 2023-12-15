import os
import zipfile
from pathlib import Path
from urllib import request


def download_glosat() -> None:
    """
    download script for GloSAT dataset
    :return: None
    """
    print("process can take a quite some time")
    print("downloading...")

    # define paths and create folder
    url = "https://zenodo.org/records/5363457/files/datasets.zip"
    target = f"{Path(__file__).parent.absolute()}/../data/GloSAT/raw/"
    file = f"{target}/datasets.zip"
    os.makedirs(target, exist_ok=True)

    # download dataset
    request.urlretrieve(url, file)

    # extract zip file
    with zipfile.ZipFile(file, 'r') as zipper:
        folder = Path(target)
        folder.mkdir(exist_ok=True)
        zipper.extractall(path=folder)

    # remove zip file
    os.remove(file)


def download_ours() -> None:
    # TODO: write script to download our dataset in the Datafolder
    pass


def main(dataset: str = 'GloSAT'):
    if dataset.lower() == 'glosat':
        download_glosat()
    elif dataset.lower() == 'ours':
        download_ours()

    else:
        raise Exception('No download script for this dataset!')


if __name__ == '__main__':
    main('GloSAT')

"""Script to download datasets."""
import gzip
import os
import tarfile
import zipfile
from pathlib import Path
from urllib import request


def download_glosat() -> None:
    """Download script for GloSAT dataset."""
    print("process can take a quite some time")
    print("downloading...")

    # define paths and create folder
    url = "https://zenodo.org/records/5363457/files/datasets.zip"
    target = f"{Path(__file__).parent.absolute()}/../../data/GloSAT/"
    file = f"{target}/datasets.zip"
    os.makedirs(target, exist_ok=True)

    # download dataset
    request.urlretrieve(url, file)

    # extract zip file
    print("extracting...")
    with zipfile.ZipFile(file, "r") as zipper:
        folder = Path(target)
        folder.mkdir(exist_ok=True)
        zipper.extractall(path=folder)

    # remove zip file
    os.remove(file)


def download_pubtables() -> None:
    """Download script for pubtables-1m dataset."""
    print("process can take a quite some time")

    files = [
        # "PubTables-1M-Detection_Annotations_Test.tar.gz",
        # "PubTables-1M-Detection_Annotations_Train.tar.gz",
        "PubTables-1M-Detection_Annotations_Val.tar.gz",
        # "PubTables-1M-Detection_Images_Test.tar.gz",
        # "PubTables-1M-Detection_Images_Train_Part1.tar.gz",
        # "PubTables-1M-Detection_Images_Train_Part2.tar.gz",
        "PubTables-1M-Detection_Images_Val.tar.gz",
        # "PubTables-1M-Structure_Annotations_Test.tar.gz",
        # "PubTables-1M-Structure_Annotations_Train.tar.gz",
        "PubTables-1M-Structure_Annotations_Val.tar.gz",
        # "PubTables-1M-Structure_Images_Test.tar.gz",
        # "PubTables-1M-Structure_Images_Train.tar.gz",
        "PubTables-1M-Structure_Images_Val.tar.gz"
    ]

    target = f"{Path(__file__).parent.absolute()}/../../data/PubTables/"

    for file in files:
        print(f"downloading {file} ...")
        url = (f"https://huggingface.co/datasets/bsmock/pubtables-1m/"
               f"resolve/main/{file}?download=true")
        request.urlretrieve(url, target + file)

        # extract zip file
        os.makedirs(target + file[:-7], exist_ok=True)
        with gzip.open(target + file, 'rb') as f_in:
            with tarfile.open(fileobj=f_in, mode='r') as tar:   # type: ignore
                tar.extractall(path=target + file[:-7])

        # remove zip file
        os.remove(target + file)


def download_ours() -> None:
    """Download script for our data."""
    # TODO: implement download script for our dataset
    pass


def main(dataset: str = "GloSAT") -> None:
    """
    Download script for datasets. 'ours' currently not implemented.

    Args:
        dataset: name of the dataset ('GloSAT' or 'ours')

    Raises:
        Exception: if there is no download script for the given dataset
    """
    if dataset.lower() == "glosat":
        download_glosat()
    elif dataset.lower() == "ours":
        download_ours()

    else:
        raise Exception("No download script for this dataset!")


if __name__ == "__main__":
    download_pubtables()

"""
script to demostarte targets for model
"""
from typing import Optional

from matplotlib import pyplot as plt
from pathlib import Path
import os


def pltbox(image, boundingboxes: list, title: Optional[str] = None):
    """
    plots bounding boxes in image
    """
    plt.imshow(image)
    for box in boundingboxes:
        # with bounding box x being height and y being width (if np.flip in extract_glosat_annotation)
        ymin = box[1]
        xmin = box[0]
        ymax = box[3]
        xmax = box[2]
        xlist = [xmin, xmin, xmax, xmax, xmin]
        ylist = [ymin, ymax, ymax, ymin, ymin]
        plt.plot(xlist, ylist)

    if title is not None:
        plt.title(title)
    plt.show()


def plot(folder: str):
    """
    shows target for model from given folder
    :param folder: path to folder of preprocessed
    :return:
    """
    tablelist = []
    celllist = []
    rowlist = []
    collist = []
    imglist = []
    files = os.listdir(folder)
    # for root, dirs, files in os.walk(folder):
    for filename in sorted(files):
        if "table" in filename and filename.endswith('.jpg'):
            img = plt.imread(folder + filename)
            imglist.append(img)

        if "table" in filename and filename.endswith('.txt'):
            with open(folder + filename, 'r') as f:
                tablelist = [tuple([int(l) for l in line.split()]) for line in f]

        if "cell" in filename:
            with open(folder + filename, 'r') as f:
                celldata = [tuple([int(l) for l in line.split()]) for line in f]
            celllist.append(celldata)

        if "row" in filename:
            with open(folder + filename, 'r') as f:
                rowdata = [tuple([int(l) for l in line.split()]) for line in f]
            rowlist.append(rowdata)

        if "col" in filename:
            with open(folder + filename, 'r') as f:
                coldata = [tuple([int(l) for l in line.split()]) for line in f]
            collist.append(coldata)

    pltbox(plt.imread(folder + sorted(files)[0]), tablelist, title='tables')

    for idx, img in enumerate(imglist):
        pltbox(img, celllist[idx], title='cells')
        pltbox(img, rowlist[idx], title='rows')
        pltbox(img, collist[idx], title='columns')


def main():
    pass


if __name__ == '__main__':
    plot(f'{Path(__file__).parent.absolute()}/../data/preprocessed/113/')

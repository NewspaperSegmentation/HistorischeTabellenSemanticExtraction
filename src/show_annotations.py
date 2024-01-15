"""
script to demostarte targets for model
"""
from typing import Optional, List, Tuple

import torch
from matplotlib import pyplot as plt
from pathlib import Path
import os


def pltbox(image, boundingboxes: List[Tuple[int, int, int, int]], title: Optional[str] = None,
           textregions: Optional[list] = None, save_as: Optional[str] = None):
    """
    plots bounding boxes in image
    :param image:  image of page
    :param boundingboxes: List of Boundingboxes to plot
    :param title: title of the plot (optional)
    :param textregions: Boundingboxes of textregions (optional)
    :param save_as: name of the folder to save the image in. Images are saved in data/assets/images/  (optional)
    :return:
    """
    # plot image
    plt.imshow(image)
    y, x, _ = image.shape

    # add bboxes
    for box in boundingboxes:
        # with bounding box x being height and y being width (if np.flip in extract_glosat_annotation)
        ymin = box[1] * y
        xmin = box[0] * x
        ymax = box[3] * y
        xmax = box[2] * x
        xlist = [xmin, xmin, xmax, xmax, xmin]
        ylist = [ymin, ymax, ymax, ymin, ymin]
        plt.plot(xlist, ylist)

    # add textregions if existing
    if textregions:
        for idx, textbox in enumerate(textregions):
            ymin = textbox[1] * y
            xmin = textbox[0] * x
            ymax = textbox[3] * y
            xmax = textbox[2] * x
            xlist = [xmin, xmin, xmax, xmax, xmin]
            ylist = [ymin, ymax, ymax, ymin, ymin]
            plt.plot(xlist, ylist, 'g')

    # add title if existing
    if title is not None:
        plt.title(title)

    # save images if save_as folder is given
    if save_as is not None:
        path = f"{Path(__file__).parent.absolute()}/../data/assets/images/{save_as}/"
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}{save_as}_{title}.png")

    # show plot
    plt.show()


def plot(folder: str, save_as: Optional[str] = None):
    """
    shows target for model from given folder
    :param folder: path to folder of preprocessed
    :param save_as: name of the folder to save the images in. Images are saved in data/assets/images/  (optional)
    :return:
    """

    # List of bboxes for every property
    textlist = []
    has_textregion = False
    tablelist = []
    celllist = []
    rowlist = []
    collist = []
    imglist = []

    # iterate over files in folder and extract bboxes
    files = os.listdir(folder)
    for filename in sorted(files):
        if "textregions" in filename:
            textlist = torch.load(folder + filename)
            has_textregion = True

        if "table" in filename and filename.endswith('.jpg'):
            imglist.append(plt.imread(folder + filename))

        if "table" in filename and filename.endswith('.pt'):
            tablelist = torch.load(folder + filename)

        if "cell" in filename and filename.endswith('.pt'):
            celllist.append(torch.load(folder + filename))

        if "row" in filename and filename.endswith('.pt'):
            rowlist.append(torch.load(folder + filename))

        if "col" in filename and filename.endswith('.pt'):
            collist.append(torch.load(folder + filename))

    # plot tables
    pltbox(plt.imread(folder + sorted(files)[0]), tablelist, title='tables', save_as=save_as)

    # plot tables and textregions if textregions exists
    if has_textregion:
        pltbox(plt.imread(folder + sorted(files)[0]), tablelist, title='tables and textregions',
               textregions=textlist, save_as=save_as)

    # plot cells, rows and columns
    for idx, img in enumerate(imglist):
        pltbox(img, celllist[idx], title='cells', save_as=save_as)
        pltbox(img, rowlist[idx], title='rows', save_as=save_as)
        pltbox(img, collist[idx], title='columns', save_as=save_as)


if __name__ == '__main__':
    # plot(f'{Path(__file__).parent.absolute()}/../data/Tables/preprocessed/IMG_20190821_132903/', save_as='OurExampleIMG_20190821_132903')
    plot(f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed/4/', save_as='GloSATExample4')

"""
script to demostarte targets for model
"""
from typing import Optional, List, Tuple

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

    # add bboxes
    for box in boundingboxes:
        # with bounding box x being height and y being width (if np.flip in extract_glosat_annotation)
        ymin = box[1]
        xmin = box[0]
        ymax = box[3]
        xmax = box[2]
        xlist = [xmin, xmin, xmax, xmax, xmin]
        ylist = [ymin, ymax, ymax, ymin, ymin]
        plt.plot(xlist, ylist)

    # add textregions if existing
    if textregions:
        for idx, textbox in enumerate(textregions):
            ymin = textbox[1]
            xmin = textbox[0]
            ymax = textbox[3]
            xmax = textbox[2]
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
            with open(folder + filename, 'r') as f:
                textlist = [tuple([int(l) for l in line.split()]) for line in f]
            has_textregion = True

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
    plot(f'{Path(__file__).parent.absolute()}/../data/GLoSAT/preprocessed/13/', save_as='GloSATExample13')

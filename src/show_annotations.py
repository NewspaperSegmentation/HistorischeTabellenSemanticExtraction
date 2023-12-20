"""
script to demostarte targets for model
"""
from matplotlib import pyplot as plt
from pathlib import Path
import os

def pltbox(image, boundingboxes:list):
    """
    plots bounding boxes in image
    """
    plt.imshow(image)
    for box in boundingboxes:
        #with bounding box x being height and y being width (if np.flip in extract_glosat_annotation)
        #ymin = box[0]
        #xmin = box[1]  
        #ymax = box[2]
        #xmax =box[3]

        ymin = box[1]
        xmin = box[0]
        ymax = box[3]
        xmax =box[2]
        xlist = [xmin,xmin,xmax,xmax,xmin]
        ylist = [ymin,ymax,ymax,ymin,ymin]
        plt.plot(xlist, ylist, 'b')
    plt.show()

def plot(folder: str):
    """
    shows target for model from given folder
    :param folder: path to folder of preprocessed
    :return:
    """
    celllist= []
    rowlist = []
    imglist = []
    files = os.listdir(folder)
    #for root, dirs, files in os.walk(folder):
    for filename in sorted(files):
        if "row" in filename:
            with open(folder+filename, 'r') as f:
                rowdata = [tuple([int(l) for l in line.split()]) for line in f ]
            rowlist.append(rowdata)
        if "cell" in filename:
            with open(folder+filename, 'r') as f:
                celldata = [tuple([int(l) for l in line.split()]) for line in f ]
            celllist.append(celldata)
        if "table" in filename and filename.endswith('.jpg'):
            img = plt.imread(folder+filename)
            imglist.append(img)
    for idx, img in enumerate(imglist):
        pltbox(img, celllist[idx])
        pltbox(img, rowlist[idx])
def main():
    pass

if __name__ == '__main__':
    plot(f'{Path(__file__).parent.absolute()}/../data/preprocessed/4/')
"""
script to demostarte targets for model
"""
from matplotlib import pyplot as plt

def pltbox(image, boundingboxes:list):
    plt.imshow(image)
    for box in boundingboxes:
        ymin = box[1]
        xmin = box[0]
        ymax = box[3]
        xmax =box[2]
        xlist = [xmin,xmin,xmax,xmax,xmin]
        ylist = [ymin,ymax,ymax,ymin,ymin]
        plt.plot(xlist, ylist)
    plt.show()

def plot(folder: str):
    """
    shows target for model from given folder
    :param folder: path to folder of preprocessed
    :return:
    """
    pass

def main():
    pass

if __name__ == '__main__':
    pltbox()
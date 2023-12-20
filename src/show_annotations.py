"""
script to demostarte targets for model
"""
from matplotlib import pyplot as plt

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
    pass

def main():
    pass

if __name__ == '__main__':
    pltbox()
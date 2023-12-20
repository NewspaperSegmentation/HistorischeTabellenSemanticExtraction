import os
from pathlib import Path
from pprint import pprint
from typing import List
from PIL import Image
import torch

import numpy as np
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt

from utils.utils import convert_coords, get_bbox
from show_annotations import pltbox, plot


def extract_annotation(file: str) -> List[dict]:
    """
    extracts annotation data from file
    :param file: path to file
    :return:
    """
    tables = [{'coords': None, 'cells': None, 'columns': None, 'rows': None}]

    # extract info
    # calc cell, col and row coords relative to table
    # coords like torch indexing

    return tables


def extract_glosat_annotation(file: str, mode: str = 'maximum', table_relative :bool =True) -> List[dict]:
    """
    extracts annotation data from transkribus xml file
    :param file: path to file
    :param mode: mode for bounding box extraction options are 'maximum' and 'corners'
    'maximum': creates a bounding box including all points annotated
    'corners': creates a bounding box by connecting the corners of the annotation
    :param table_relative: if True, bounding boxes for cell and row are calculated relative to table position
    :return:
    """
    tables = [{'coords': None, 'cells': None, 'columns': None, 'rows': None}]

    with open(file, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Parse the XML content
    soup = BeautifulSoup(xml_content, 'xml')

    page = soup.find('Page')
    size = (int(page['imageHeight']), int(page['imageWidth']))

    # Find all TextLine elements and extract the Baseline points
    for table in soup.find_all('TableRegion'):
        t = {'coords': table.find('Coords')['points'], 'cells': [], 'columns': [], 'rows': []}
        i=0
        currentrow = []
        if table_relative:
            coord = get_bbox(convert_coords(t['coords']))
        else:
            coord =None
        for cell in table.find_all('TableCell'):
            # get points and corners of cell
            points = convert_coords(cell.find('Coords')['points'])
            corners = [int(x) for x in cell.find('CornerPts').text.split()] if mode == 'corners' else None
            #points = np.flip(points, 1) #why np.flip? turned off for now since this way coord order is inconsistent between table coords and cell/row coords

            # get bounding box
            bbox = get_bbox(points, corners,coord)

            # add to dictionary
            t['cells'].append(bbox)

            if cell.get('rowSpan')!=None: #credit cell rausnehmen
                if int(cell['row']) <= i:
                    currentrow.extend(points.tolist())
                    i = max(int(cell['row']) + int(cell['rowSpan']) -1,i)
                else:
                    bbox = get_bbox(np.array(currentrow),tablebbox=coord)
                    t['rows'].append(bbox)
                    currentrow.clear()
                    i+=1
                    currentrow.extend(points.tolist())
        if currentrow:
            bbox = get_bbox(np.array(currentrow),tablebbox=coord)
            t['rows'].append(bbox)

        tables.append(t)
    del tables[0]

    # extract info
    # calc cell, col and row coords relative to table
    # coords like torch indexing

    return tables

def preprocess(image: str, tables: List[dict]) -> None:
    """
    does preprocessing to the image and cuts outs tables. Then save image and all cut out rois as different files
    :param image: path to image
    :param tables: extracted annotations
    :return:
    """
    # TODO: define preprocessing process

    # create new folder for image files

    splitname = image.split('JPEGImages/')[1]
    target = f"{Path(__file__).parent.absolute()}/../data/preprocessed/"+splitname[:-4]+"/"
    os.makedirs(target, exist_ok=True)
    img = Image.open(image)

    # preprocessing image

    # cut out rois

    # save image (naming: image_file_name . pt)

    impath = f"{target}/"+splitname[:-4]+".pt"
    #img.save(impath)
    #torch.save(image, impath)
    img.save(f"{target}/"+splitname[:-4]+".jpg")

    # save text bounding boxs (naming: image_file_name _ texts . pt)
    #not added yet since not available for glosat

    # save one file for every roi
    tablelist = []
    for idx, tab in enumerate(tables):
        pass

        # cut out table form image save as (naming: image_file_name _ table _ idx . pt)
    
        coord = get_bbox(convert_coords(tab['coords']))
        tablelist.append(coord)
        tableimg = img.crop((coord))
        tablepath = f"{target}/"+splitname[:-4]+"_table_"+str(idx)+".pt"
        #torch.save(tableimg,tablepath)
        tableimg.save(f"{target}/"+splitname[:-4]+"_table_"+str(idx)+".jpg")

        # cell bounding boxs (naming: image_file_name _ cell _ idx . pt)
        cellpath = f"{target}/"+splitname[:-4]+"_cell_"+str(idx)+".txt"
        #torch.save()
        cellfile = open(cellpath, "w")
        cellfile.write('\n'.join('{} {} {} {}'.format(cell[0],cell[1], cell[2], cell[3]) for cell in tab['cells']))
        cellfile.close()
        # column bounding boxs (naming: image_file_name _ col _ idx . pt)

        # row bounding boxs (naming: image_file_name _ row _ idx . pt)
        rowpath = f"{target}/"+splitname[:-4]+"_row_"+str(idx)+".txt"
        rowfile = open(rowpath, "w")
        rowfile.write('\n'.join('{} {} {} {}'.format(cell[0],cell[1], cell[2], cell[3]) for cell in tab['rows']))
        rowfile.close()
    # save tabel bounding boxs (naming: image_file_name _ tables . pt)
    tablepath = f"{target}/"+splitname[:-4]+"_tables.txt"
    tablefile = open(tablepath, "w")
    tablefile.write('\n'.join('{} {} {} {}'.format(cell[0],cell[1], cell[2], cell[3]) for cell in tablelist))
    tablefile.close()


def main(folder: str, dataset_type: str):
    """
    takes the folder of a dataset and preprocesses it. Save preprocessed images and files with bounding boxes
    table.pt: file with bounding boxes of tables format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    text.pt: file with bounding boxes of text format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    cell.pt: file with bounding boxes of cell format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    column.pt: file with bounding boxes of column format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    row.pt: file with bounding boxes of row format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    :param folder:
    :param dataset_type:
    :return:
    """
    # TODO: does preprocessing over data in folder
    pass

    for file in folder:
        # find out if Glosat and our dataformat is the same
        # write individual or one function to extract information
        tables = extract_annotation(file) # maybe two one for every dataset

        # cut out tables

        # preprocess images
        preprocess(file, tables)



        # row bounding boxs



if __name__ == '__main__':
    tables = extract_glosat_annotation(f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/Fine/Transkribus/4.xml', 'corners', table_relative=False)
    #pprint(tables)
    img = plt.imread(f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/JPEGImages/4.jpg')
    #print(tables)
    #pltbox(img, tables[0]['rows'])
    pltbox(img,[x for t in tables for x in t['rows']] )
    tables = extract_glosat_annotation(f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/Fine/Transkribus/4.xml', 'corners')
    #print(img.shape)
    preprocess(f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/JPEGImages/4.jpg', tables)
    plot(f'{Path(__file__).parent.absolute()}/../data/preprocessed/4/')
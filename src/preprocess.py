from pathlib import Path
from pprint import pprint
from typing import List

import numpy as np
from bs4 import BeautifulSoup

from src.utils.utils import convert_coords, get_bbox


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


def extract_glosat_annotation(file: str, mode: str = 'maximum') -> List[dict]:
    """
    extracts annotation data from file
    :param file: path to file
    :param mode: mode for bounding box extraction options are 'maximum' and 'corners'
    'maximum': creates a bounding box including all points annotated
    'corners': creates a bounding box by connecting the corners of the annotation
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
        for cell in table.find_all('TableCell'):
            # get points and corners of cell
            points = convert_coords(cell.find('Coords')['points'])
            corners = [int(x) for x in cell.find('CornerPts').text.split()] if mode == 'corners' else None
            points = np.flip(points, 1)

            # get bounding box
            bbox = get_bbox(points, corners)

            # add to dictionary
            t['cells'].append(bbox)

        tables.append(t)


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

    # preprocessing image

    # cut out rois

    # save image (naming: image_file_name . pt)

    # save tabel bounding boxs (naming: image_file_name _ tables . pt)

    # save text bounding boxs (naming: image_file_name _ texts . pt)

    # save one file for every roi
    for idx, tab in enumerate(tables):
        pass

        # cut out table form image save as (naming: image_file_name _ table _ idx . pt)

        # cell bounding boxs (naming: image_file_name _ cell _ idx . pt)

        # column bounding boxs (naming: image_file_name _ col _ idx . pt)

        # row bounding boxs (naming: image_file_name _ row _ idx . pt)


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
    tables = extract_glosat_annotation(f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/Coarse/Transkribus/1.xml', 'corners')
    pprint(tables)
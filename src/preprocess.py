"""
script for extracting annotations and preprocessing images
"""

import glob
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from bs4 import BeautifulSoup
from torchvision import transforms
from tqdm import tqdm

from utils.utils import convert_coords, get_bbox
from show_annotations import plot


def extract_annotation(file: str, mode: str = 'maximum', table_relative: bool = True) -> (List[dict], List[dict]):
    """
    extracts annotation data from transkribus xml file
    :param file: path to file
    :param mode: mode for bounding box extraction options are 'maximum' and 'corners'
    'maximum': creates a bounding box including all points annotated
    'corners': creates a bounding box by connecting the corners of the annotation
    :param table_relative: if True, bounding boxes for cell and row are calculated relative to table position
    :return:
    """
    tables = []
    textregions = []

    with open(file, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Parse the XML content
    soup = BeautifulSoup(xml_content, 'xml')

    page = soup.find('Page')

    # Find all TextLine elements and extract the Baseline points
    for textline in soup.find_all('TextRegion'):
        text = {'coords': get_bbox(convert_coords(textline.find('Coords')['points']))}
        textregions.append(text)

    for table in soup.find_all('TableRegion'):
        t = {'coords': get_bbox(convert_coords(table.find('Coords')['points'])), 'cells': [], 'columns': [], 'rows': []}
        i = 0  # row counter
        currentrow = []  # list of points in current row
        maxcol = 0

        columns = {}  # dictionary of columns and their points
        rows = {}  # dictionary of rows and their points
        col_joins = []  # list of join operations for columns
        row_joins = []  # list of join operations for rows

        for cell in table.find_all('TableCell'):
            maxcol = max(maxcol, int(cell['col']))
        firstcell = table.find('TableCell')

        if cell.get('rowSpan') != None and int(firstcell['colSpan']) == maxcol + 1 and int(firstcell['row']) == 0:
            coord1 = t['coords']
            # print(t['coords'])
            coord2 = get_bbox(convert_coords(firstcell.find('Coords')['points']))
            t['coords'] = (coord1[0], coord2[3], coord1[2], coord1[3])
            # print(t['coords'], 'h')
        if table_relative:
            coord = t['coords']
        else:
            coord = None

        # iterate over cells in table
        for cell in table.find_all('TableCell'):
            # get points and corners of cell
            points = convert_coords(cell.find('Coords')['points'])

            # uses corner points of cell if mode is 'corners'
            corners = [int(x) for x in cell.find('CornerPts').text.split()] if mode == 'corners' else None

            # get bounding box
            bbox = get_bbox(points, corners, coord)

            # add to dictionary
            x_flat = bbox[0] >= bbox[2]                 # bbox flatt in x dim
            y_flat = bbox[1] >= bbox[3]                 # bbox flatt in y dim
            noHeaderCell = cell.get('rowSpan') is not None and not (int(cell['colSpan']) == maxcol + 1 and int(cell['row']) == 0) # check if Cell is a header for table
            if not x_flat and not y_flat and noHeaderCell:
                t['cells'].append(bbox)

                # calc rows
                # add row number to dict
                if int(cell['row']) in rows.keys():
                    rows[int(cell['row'])].extend(points.tolist())
                else:
                    rows[int(cell['row'])] = points.tolist()

                # when cell over multiple rows create a join operation
                if int(cell['rowSpan']) > 1:
                    row_joins.extend([(int(cell['row']), int(cell['row']) + s) for s in range(1, int(cell['rowSpan']))])

                # add col number to dict
                if int(cell['col']) in columns.keys():
                    columns[int(cell['col'])].extend(points.tolist())
                else:
                    columns[int(cell['col'])] = points.tolist()

                # when cell over multiple columns create a join operation
                if int(cell['colSpan']) > 1:
                    col_joins.extend([(int(cell['col']), int(cell['col']) + s) for s in range(1, int(cell['colSpan']))])

        # join overlapping rows
        row_set = DisjointSet(rows.keys())
        for join in row_joins:
            if join[0] in rows.keys() and join[1] in rows.keys():
                row_set.merge(*join)

        rows = [[point for key in lst for point in rows[key]] for lst in row_set.subsets()]
        t['rows'] = [get_bbox(np.array(col), tablebbox=coord) for col in rows]

        # join overlapping columns
        col_set = DisjointSet(columns.keys())
        for join in col_joins:
            if join[0] in columns.keys() and join[1] in columns.keys():
                col_set.merge(*join)

        columns = [[point for key in lst for point in columns[key]] for lst in col_set.subsets()]
        t['columns'] = [get_bbox(np.array(col), tablebbox=coord) for col in columns]

        if t['columns'] and t['rows']:
            tables.append(t)

    return tables, textregions


def preprocess(image: str, tables: List[dict], target: str, file_name: str, text: List[dict] = None) -> None:
    """
    does preprocessing to the image and cuts outs tables. Then save image and all cut out rois as different files
    :param image: path to image
    :param tables: extracted annotations
    :param target: folder to save the results in
    :param file_name: name of image
    :return:
    """
    # create function to convert PIL Image to torch Tensor
    to_tensor = transforms.PILToTensor()

    # create new folder for image files
    # this way of splitting won't work for different folder structure, probably need to change it for our dataset
    target = f"{target}/{file_name}/"
    os.makedirs(target, exist_ok=True)

    # preprocessing image
    img = Image.open(image)
    img = ImageOps.exif_transpose(img)  # turns image in right orientation

    # save image as jpg and pt
    img.save(f"{target}/" + file_name + ".jpg")
    torch.save(to_tensor(img), f"{target}/" + file_name + ".pt")

    # save one file for every roi
    tablelist = []
    for idx, tab in enumerate(tables):
        # get table coords
        coord = tab['coords']
        tablelist.append(coord)

        # crop table from image
        tableimg = img.crop((coord))

        # save image of table
        torch.save(to_tensor(tableimg), f"{target}/" + file_name + "_table_" + str(idx) + ".pt")
        tableimg.save(f"{target}/" + file_name + "_table_" + str(idx) + ".jpg")

        # cell bounding boxs (naming: image_file_name _ cell _ idx . pt)
        cells = torch.tensor(tab['cells'])
        torch.save(cells, f"{target}/" + file_name + "_cell_" + str(idx) + ".pt")

        # column bounding boxs (naming: image_file_name _ col _ idx . pt)
        columns = torch.tensor(tab['columns'])
        torch.save(columns, f"{target}/" + file_name + "_col_" + str(idx) + ".pt")

        # row bounding boxs (naming: image_file_name _ row _ idx . pt)
        rows = torch.tensor(tab['rows'])
        torch.save(rows, f"{target}/" + file_name + "_row_" + str(idx) + ".pt")

    # save table bounding boxs (naming: image_file_name _ tables . pt)
    table = torch.tensor(tablelist)
    torch.save(table, f"{target}/" + file_name + "_tables.pt")

    # save text bounding boxes (naming: image_file_name _ texts . pt)
    textlist = []
    if text:
        for idx, region in enumerate(text):
            textlist.append(region['coords'])

        texts = torch.tensor(textlist)
        torch.save(texts, f"{target}/" + file_name + "_textregions" + ".pt")


def main(datafolder: str, imgfolder: str, targetfolder: str):
    """
    takes the folder of a dataset and preprocesses it. Save preprocessed images and files with bounding boxes
    table.pt: file with bounding boxes of tables format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    text.pt: file with bounding boxes of text format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    cell.pt: file with bounding boxes of cell format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    column.pt: file with bounding boxes of column format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    row.pt: file with bounding boxes of row format (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    :param targetfolder: path for saving the images
    :return:
    """
    print("Processing folder, this may take a little while!")

    files = [x for x in glob.glob(f"{datafolder}/*.xml")]
    file_names = [os.path.splitext(os.path.basename(path))[0] for path in files]
    images = [f"{imgfolder}/{x}.jpg" for x in file_names]

    for file_name, file, img in tqdm(zip(file_names, files, images), desc='preprocessing', total=len(files)):
        # check for strange files
        if plt.imread(img).ndim == 3:
            table, text = extract_annotation(file)
            preprocess(img, table, targetfolder, file_name, text)


if __name__ == '__main__':
    ours = False
    glosat = True

    if ours:
        main(datafolder=f'{Path(__file__).parent.absolute()}/../data/Tables/TableDataset/',
             imgfolder=f'{Path(__file__).parent.absolute()}/../data/Tables/TableDataset/',
             targetfolder=f'{Path(__file__).parent.absolute()}/../data/Tables/preprocessed/')

        plot(f'{Path(__file__).parent.absolute()}/../data/Tables/preprocessed/IMG_20190821_141527/')

    if glosat:
        main(datafolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/Fine/Transkribus/',
             imgfolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/JPEGImages/',
             targetfolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed/')

        main(datafolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Test/Fine/Transkribus/',
             imgfolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Test/JPEGImages/',
             targetfolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed/')

        plot(f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed/4/')

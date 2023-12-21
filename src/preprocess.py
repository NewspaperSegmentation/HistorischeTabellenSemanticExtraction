"""
script for extracting annotations and preprocessing images
"""

import glob
import os
from pathlib import Path
from typing import List
from PIL import Image, ImageOps

import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from bs4 import BeautifulSoup
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
    size = (int(page['imageHeight']), int(page['imageWidth']))

    # Find all TextLine elements and extract the Baseline points
    for textline in soup.find_all('TextRegion'):
        text = {'coords': get_bbox(convert_coords(textline.find('Coords')['points'])) }
        textregions.append(text)

    for table in soup.find_all('TableRegion'):
        t = {'coords': table.find('Coords')['points'], 'cells': [], 'columns': [], 'rows': []}
        i = 0  # row counter
        currentrow = []  # list of points in current row

        columns = {}    # dictionary of columns and their points
        rows = {}       # dictionary of rows and their points
        col_joins = []  # list of join operations for columns
        row_joins = []  # list of join operations for rows

        if table_relative:
            coord = get_bbox(convert_coords(t['coords']))
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
            t['cells'].append(bbox)

            # calc rows
            if cell.get('rowSpan') != None:  # credit cell rausnehmen

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

        tables.append(t)

    return tables, textregions


def preprocess(image: str, tables: List[dict], target: str, file_name: str, text: List[dict]=None) -> None:
    """
    does preprocessing to the image and cuts outs tables. Then save image and all cut out rois as different files
    :param image: path to image
    :param tables: extracted annotations
    :param target: folder to save the results in
    :param file_name: name of image
    :return:
    """

    # create new folder for image files
    # this way of splitting won't work for different folder structure, probably need to change it for our dataset
    target = f"{target}/{file_name}/"
    os.makedirs(target, exist_ok=True)

    # preprocessing image
    img = Image.open(image)
    img = ImageOps.exif_transpose(img)  # turns image in right orientation

    # save image (naming: image_file_name . pt)
    img.save(f"{target}/" + file_name + ".jpg")

    # save text bounding boxs (naming: image_file_name _ texts . pt)
    textlist= []
    if text:
        for idx, region in enumerate(text):
            coord = region['coords']
            textlist.append(coord)
        textpath = f"{target}/" + file_name + "_textregions" + ".txt"
        textfile = open(textpath, "w")
        textfile.write('\n'.join('{} {} {} {}'.format(cell[0], cell[1], cell[2], cell[3]) for cell in textlist))
        textfile.close()
    # not added yet since not available for glosat

    # save one file for every roi
    tablelist = []
    for idx, tab in enumerate(tables):
        # cut out table form image save as (naming: image_file_name _ table _ idx . pt)
        coord = get_bbox(convert_coords(tab['coords']))
        tablelist.append(coord)
        tableimg = img.crop((coord))
        tableimg.save(f"{target}/" + file_name + "_table_" + str(idx) + ".jpg")

        # cell bounding boxs (naming: image_file_name _ cell _ idx . pt)
        cellpath = f"{target}/" + file_name + "_cell_" + str(idx) + ".txt"
        cellfile = open(cellpath, "w")
        cellfile.write('\n'.join('{} {} {} {}'.format(cell[0], cell[1], cell[2], cell[3]) for cell in tab['cells']))
        cellfile.close()

        # column bounding boxs (naming: image_file_name _ col _ idx . pt)
        colpath = f"{target}/" + file_name + "_col_" + str(idx) + ".txt"
        colfile = open(colpath, "w")
        colfile.write('\n'.join('{} {} {} {}'.format(col[0], col[1], col[2], col[3]) for col in tab['columns']))
        colfile.close()

        # row bounding boxs (naming: image_file_name _ row _ idx . pt)
        rowpath = f"{target}/" + file_name + "_row_" + str(idx) + ".txt"
        rowfile = open(rowpath, "w")
        rowfile.write('\n'.join('{} {} {} {}'.format(row[0], row[1], row[2], row[3]) for row in tab['rows']))
        rowfile.close()

    # save tabel bounding boxs (naming: image_file_name _ tables . pt)
    tablepath = f"{target}/" + file_name + "_tables.txt"
    tablefile = open(tablepath, "w")
    tablefile.write('\n'.join('{} {} {} {}'.format(cell[0], cell[1], cell[2], cell[3]) for cell in tablelist))
    tablefile.close()


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
        table, text = extract_annotation(file)
        preprocess(img, table, targetfolder, file_name,text)


if __name__ == '__main__':
    ours = True
    glosat = True

    if ours:
        main(datafolder=f'{Path(__file__).parent.absolute()}/../data/Tables/TableDataset/',
             imgfolder=f'{Path(__file__).parent.absolute()}/../data/Tables/TableDataset/',
             targetfolder=f'{Path(__file__).parent.absolute()}/../data/Tables/preprocessed/')

        plot(f'{Path(__file__).parent.absolute()}/../data/Tables/preprocessed/IMG_20190821_141527/')

    if glosat:
        #main(datafolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/Fine/Transkribus/',
        #     imgfolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/JPEGImages/',
        #     targetfolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed/')

        plot(f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed/8/')

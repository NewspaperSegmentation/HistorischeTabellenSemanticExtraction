import os
from pathlib import Path
from typing import List
from PIL import Image

import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils.utils import convert_coords, get_bbox
from show_annotations import plot


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


def extract_glosat_annotation(file: str, mode: str = 'maximum', table_relative: bool = True) -> List[dict]:
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

    with open(file, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Parse the XML content
    soup = BeautifulSoup(xml_content, 'xml')

    page = soup.find('Page')
    size = (int(page['imageHeight']), int(page['imageWidth']))

    # Find all TextLine elements and extract the Baseline points
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
    # this way of splitting won't work for different folder structure, probably need to change it for our dataset
    splitname = image.split('JPEGImages/')[1]
    target = f"{Path(__file__).parent.absolute()}/../data/preprocessed/" + splitname[:-4] + "/"
    os.makedirs(target, exist_ok=True)
    img = Image.open(image)

    # preprocessing image

    # cut out rois

    # save image (naming: image_file_name . pt)
    img.save(f"{target}/" + splitname[:-4] + ".jpg")

    # save text bounding boxs (naming: image_file_name _ texts . pt)
    # not added yet since not available for glosat

    # save one file for every roi
    tablelist = []
    for idx, tab in enumerate(tables):
        # cut out table form image save as (naming: image_file_name _ table _ idx . pt)
        coord = get_bbox(convert_coords(tab['coords']))
        tablelist.append(coord)
        tableimg = img.crop((coord))
        tableimg.save(f"{target}/" + splitname[:-4] + "_table_" + str(idx) + ".jpg")

        # cell bounding boxs (naming: image_file_name _ cell _ idx . pt)
        cellpath = f"{target}/" + splitname[:-4] + "_cell_" + str(idx) + ".txt"
        cellfile = open(cellpath, "w")
        cellfile.write('\n'.join('{} {} {} {}'.format(cell[0], cell[1], cell[2], cell[3]) for cell in tab['cells']))
        cellfile.close()

        # column bounding boxs (naming: image_file_name _ col _ idx . pt)
        colpath = f"{target}/" + splitname[:-4] + "_col_" + str(idx) + ".txt"
        colfile = open(colpath, "w")
        colfile.write('\n'.join('{} {} {} {}'.format(col[0], col[1], col[2], col[3]) for col in tab['columns']))
        colfile.close()

        # row bounding boxs (naming: image_file_name _ row _ idx . pt)
        rowpath = f"{target}/" + splitname[:-4] + "_row_" + str(idx) + ".txt"
        rowfile = open(rowpath, "w")
        rowfile.write('\n'.join('{} {} {} {}'.format(row[0], row[1], row[2], row[3]) for row in tab['rows']))
        rowfile.close()

    # save tabel bounding boxs (naming: image_file_name _ tables . pt)
    tablepath = f"{target}/" + splitname[:-4] + "_tables.txt"
    tablefile = open(tablepath, "w")
    tablefile.write('\n'.join('{} {} {} {}'.format(cell[0], cell[1], cell[2], cell[3]) for cell in tablelist))
    tablefile.close()


def main(datafolder: str, imgfolder: str, dataset_type: str):
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
    print("Processing Folder, this may take a little while!")

    files = os.listdir(datafolder)
    maxnum = max([int(x[:-4]) for x in files])
    tables = [None] * (maxnum + 1)
    # find out if Glosat and our dataformat is the same
    # write individual or one function to extract information
    # tables = extract_annotation(file) # maybe two one for every dataset
    if dataset_type.lower() == 'glosat':
        for file in tqdm(files, desc='preprocessing', total=len(files)):
            table = extract_glosat_annotation(datafolder + file)
            splitname = file[:-4]
            tables[int(splitname)] = table
    elif dataset_type.lower() == 'ours':
        for file in tqdm(files, desc='preprocessing', total=len(files)):
            table = extract_annotation(datafolder + file)
            splitname = file[:-4]
            tables[int(splitname)] = table
    else:
        raise Exception('No annotation script for this dataset!')

    images = os.listdir(imgfolder)
    for img in tqdm(images, desc='saving', total=len(images)):
        splitname = img[:-4]
        # preprocess images
        preprocess(imgfolder + img, tables[int(splitname)])


if __name__ == '__main__':
    main(datafolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/Fine/Transkribus/',
         imgfolder=f'{Path(__file__).parent.absolute()}/../data/GloSAT/datasets/Train/JPEGImages/',
         dataset_type='glosat')

    plot(f'{Path(__file__).parent.absolute()}/../data/preprocessed/8/')

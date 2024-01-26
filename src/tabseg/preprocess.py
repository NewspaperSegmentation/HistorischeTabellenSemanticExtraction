from typing import List


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


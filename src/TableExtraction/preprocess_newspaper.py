"""Preprocess for the Newspaper dataset to predict Textlines."""

import glob
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple

import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from bs4 import BeautifulSoup, PageElement
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from skimage import io
from tqdm import tqdm
import re

from src.TableExtraction.utils.utils import get_bbox


def is_valid(box: torch.Tensor):
    """
    Checks if given bounding box has a valid size.

    Args:
        box: bounding box (xmin, ymin, xmax, ymax)

    Returns:
        True if bounding box is valid
    """
    if box[2] - box[0] <= 0:
        return False
    if box[3] - box[1] <= 0:
        return False
    return True


def draw_baseline_target(shape: Tuple[int, int],
                         baselines: List[torch.Tensor],
                         width: int = 3) -> np.ndarray:
    """
    Draw baseline target for given shape.

    Args:
        shape: shape of image
        baselines: list of baselines
        width: width of the drawn baselines
    """
    # Create a blank image filled with ones (white)
    image = np.zeros(shape, dtype=np.uint8)
    image = Image.fromarray(image)

    # Draw the baselines
    draw = ImageDraw.Draw(image)
    for line in baselines:
        draw.line([(x[1], x[0]) for x in line], fill=1, width=width)

    return np.array(image)


def get_tag(textregion: PageElement):
    """
    Returns the tag of the given textregion

    Args:
        textregion: PageElement of Textregion

    Returns:
        Given tag of that Textregion
    """
    desc = textregion['custom']
    match = re.search(r"\{type:.*;\}", desc)
    if match is None:
        return 'UnknownRegion'
    return match.group()[6:-2]

def extract(xml_path: str) -> List[Dict[str, List[torch.Tensor]]]:
    """
    Extracts the annotation from the xml file.

    Args:
        xml_path: path to the xml file.

    Returns:
        A list of dictionary representing all Textregions in the given document
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    paragraphs = []

    text_regions = page.find_all('TextRegion')
    for region in text_regions:
        if get_tag(region) not in ['heading', 'article_', 'caption', 'header', 'paragraph']:
            continue

        coords = region.find('Coords')
        part = torch.tensor([tuple(map(int, point.split(','))) for
                             point in coords['points'].split()])[:, torch.tensor([1, 0])]
        part = torch.tensor(get_bbox(part))

        region_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {'part': part,
                                                                           'bboxes': [],
                                                                           'masks': [],
                                                                           'baselines': []}

        text_region = region.find_all('TextLine')
        for text_line in text_region:
            bbox = text_line.find('Coords')
            baseline = text_line.find('Baseline')
            if baseline:
                # get and shift baseline
                line = torch.tensor([tuple(map(int, point.split(','))) for
                                     point in baseline['points'].split()])[:, torch.tensor([1, 0])]
                line -= part[:2].unsqueeze(0)
                region_dict['baselines'].append(line)   # type: ignore

                # get mask
                polygone = torch.tensor([tuple(map(int, point.split(','))) for
                                         point in bbox['points'].split()])[:, torch.tensor([1, 0])]

                # move mask to be in subimage
                polygone -= part[:2].unsqueeze(0)

                # calc bbox for line
                box = torch.tensor(get_bbox(polygone))[torch.tensor([1, 0, 3, 2])]
                box = box.clip(min=0)

                # add bbox to data
                if is_valid(box):
                    region_dict['bboxes'].append(box)   # type: ignore

                    # add mask to data
                    region_dict['masks'].append(polygone)   # type: ignore

        if region_dict['bboxes']:
            region_dict['bboxes'] = torch.stack(region_dict['bboxes'])   # type: ignore
            paragraphs.append(region_dict)

    return paragraphs


def plot(image: torch.Tensor, target: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> None:
    """
    Plots the image and target to check the preprocessing.

    Args:
        image: image
        target: preprocessed target dict
    """
    paragraph = image[target['part'][0]: target['part'][2], target['part'][1]: target['part'][3]]

    fig, ax = plt.subplots(dpi=500)
    ax.imshow(paragraph)
    colors = ['yellow', 'orange']
    for i, mask in enumerate(target['masks']):
        ax.add_patch(Polygon(mask, fill=True, color=colors[i % 2], linewidth=.1, alpha=0.5))

    colors = ['red', 'blue']
    for i, box in enumerate(target['bboxes']):
        ax.add_patch(Rectangle(box[:2], box[3] - box[1], box[2] - box[0],
                         fill=False,
                         color=colors[i % 2],
                         linewidth=.1))

    for line in target['baselines']:
        ax.add_patch(Polygon(line, fill=False, color='black', linewidth=.1, closed=False))

    plt.savefig('../../data/baselineExampleTargetExample.png', dpi=1000)
    plt.show()


def rename_files(folder_path: str) -> None:
    """
    Renames all files and folders in given folder by replacing 'ö' with 'oe'.

    Args:
        folder_path: path to folder
    """
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for filename in files:
        # Check if the file name contains 'ö'
        if 'ö' in filename:
            # Replace 'ö' with 'oe' in the filename
            new_filename = filename.replace('ö', 'oe')

            # Construct the full old and new paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")


def main(image_path: str, target_path: str, output_path: str, create_baseline_target:bool) -> None:
    """
    Preprocesses the complete dataset so it can be used for training.

    Args:
        image_path: path to images
        target_path: path to xml files
        output_path: path to save folder
        create_baseline_target: if True creates targets for baseline UNet
    """
    to_tensor = transforms.PILToTensor()

    image_paths = [x for x in glob.glob(f"{image_path}/*.jpg")]
    target_paths = [f"{target_path}/{x.split(os.sep)[-1][:-4]}.xml" for x in image_paths]

    print(f"{len(image_paths)=}")
    print(f"{len(target_paths)=}")

    for img_path, tar_path in tqdm(zip(image_paths, target_paths),
                                   total=len(image_paths),
                                   desc='preprocessing'):

        document_name = img_path.split(os.sep)[-1][:-4]
        try:
            os.makedirs(f"{output_path}/{document_name}/", exist_ok=False)
        except OSError:
            continue
        regions = extract(tar_path)

        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        image = to_tensor(image).permute(1, 2, 0).to(torch.uint8)

        for i, region in enumerate(regions):
            # create dict for subimage
            os.makedirs(f"{output_path}/{document_name}/region_{i}", exist_ok=True)

            # save subimage
            subimage = image[region['part'][0]: region['part'][2],
                             region['part'][1]: region['part'][3]]

            io.imsave(f"{output_path}/{document_name}/region_{i}/image.jpg",
                      subimage.to(torch.uint8))

            # save target information
            torch.save(region['masks'],
                       f"{output_path}/{document_name}/region_{i}/masks.pt")
            torch.save(region['baselines'],
                       f"{output_path}/{document_name}/region_{i}/baselines.pt")
            torch.save(region['bboxes'],
                       f"{output_path}/{document_name}/region_{i}/bboxes.pt")

            if create_baseline_target:
                target = draw_baseline_target(subimage.shape[:2], region['baselines'])
                np.save(f"{output_path}/{document_name}/region_{i}/baselines.npy", target)


if __name__ == "__main__":
    main(f'{Path(__file__).parent.absolute()}/../../data/newspaper-dataset-main-images/images',
         f'{Path(__file__).parent.absolute()}/../../data/pero_lines_bonn_regions',
         f'{Path(__file__).parent.absolute()}/../../data/Newspaper/preprocessed',
         create_baseline_target=True)

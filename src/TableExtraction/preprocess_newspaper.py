"""Preprocess for the Newspaper dataset to predict Textlines."""

import glob
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple

import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage import draw
from bs4 import BeautifulSoup, PageElement
from matplotlib import pyplot as plt
from matplotlib import patches
from skimage import io
from shapely.ops import split
from shapely.geometry import LineString, Polygon
from tqdm import tqdm
import re

from src.TableExtraction.utils.utils import get_bbox


def plot_line_and_polygon(line, polygon):
    # Plot the line
    x, y = line.xy
    plt.plot(x, y, color='blue', linewidth=3)

    # Plot the polygon
    poly_patch = patches.Polygon(polygon.exterior.coords, facecolor='orange', edgecolor='red',
                                 alpha=0.5)
    plt.gca().add_patch(poly_patch)

    # Set plot limits
    plt.xlim(min(min(line.xy[0]), *polygon.exterior.xy[0]) - 1,
             max(max(line.xy[0]), *polygon.exterior.xy[0]) + 1)
    plt.ylim(min(min(line.xy[1]), *polygon.exterior.xy[1]) - 1,
             max(max(line.xy[1]), *polygon.exterior.xy[1]) + 1)

    # Show plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def split_textbox(box: Polygon, baseline: LineString):
    # extend line to avoid not completely intersecting polygone
    points = list(baseline.coords)
    new_coords = [(-100, points[0][1])] + points + [(points[-1][0] + 100, points[-1][1])]
    baseline = LineString(new_coords)

    # Use the split method to split the polygon
    parts = split(box, baseline)

    if len(parts.geoms) >= 2:
        # Determine which part is above and below the split line
        ascender = parts.geoms[0] if parts.geoms[0].centroid.y > baseline.centroid.y else \
            parts.geoms[1]
        descender = parts.geoms[1] if parts.geoms[0].centroid.y > baseline.centroid.y else \
            parts.geoms[0]
        return np.array(list(ascender.exterior.coords)), np.array(list(descender.exterior.coords))

    else:
        raise ValueError('Baseline and polygone not intersecting!')


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
                         textlines: List[torch.Tensor],
                         mask_regions: List[torch.Tensor],
                         name: str,
                         width: int = 3) -> np.ndarray:
    """
    Draw baseline target for given shape.

    Args:
        shape: shape of target
        baselines: list of baselines
        textlines: polygone around the textline
        width: width of the drawn baselines
    """
    # Create a blank target filled with ones (white)
    target = np.zeros((*shape, 5), dtype=np.uint8)
    target[:, :, 4] = 1  # mask with default value true

    # create PILDraw instances to draw baselines and limiters
    baseline_img = Image.fromarray(target[:, :, 2])
    baseline_draw = ImageDraw.Draw(baseline_img)

    limiter_img = Image.fromarray(target[:, :, 3])
    limiter_draw = ImageDraw.Draw(limiter_img)

    # Draw targets
    for baseline, textline in zip(baselines, textlines):
        # calc ascender and descender
        line = LineString(torch.flip(baseline, dims=[1]))
        polygon = Polygon(torch.flip(textline, dims=[1]))

        # draw baseline
        baseline_draw.line(line.coords, fill=1, width=width)

        try:
            ascender, descender = split_textbox(polygon, line)
            # draw ascender
            if len(ascender) >= 3:
                rr, cc = draw.polygon(ascender[:, 1], ascender[:, 0], shape=shape)
                target[rr, cc, 0] = 1

            # draw descender
            if len(descender) >= 3:
                rr, cc = draw.polygon(descender[:, 1], descender[:, 0], shape=shape)
                target[rr, cc, 1] = 1

        except ValueError:
            print(f"Image {name} has a problem with ascender and descender")

        min_x, min_y, max_x, max_y = polygon.bounds

        limiter_draw.line([(min_x, min_y), (min_x, max_y)], fill=1, width=width)
        limiter_draw.line([(max_x, min_y), (max_x, max_y)], fill=1, width=width)

    target[:, :, 2] = np.array(baseline_img)
    target[:, :, 3] = np.array(limiter_img)

    for mask_region in mask_regions:
        # draw mask to remove not text regions
        if len(mask_region) >= 3:
            rr, cc = draw.polygon(mask_region[:, 1], mask_region[:, 0], shape=shape)
            target[rr, cc, 4] = 0

    return np.array(target)


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


def extract(xml_path: str, create_subimages: bool) -> Tuple[List[Dict[str, List[torch.Tensor]]], List[torch.Tensor]]:
    """
    Extracts the annotation from the xml file.

    Args:
        xml_path: path to the xml file.
        create_subimages: create subimages based on the paragraph segmentation

    Returns:
        A list of dictionary representing all Textregions in the given document
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    paragraphs = []
    mask_regions = []

    text_regions = page.find_all('TextRegion')
    for region in text_regions:
        tag = get_tag(region)

        if tag in ['table', 'header']:
            coords = region.find('Coords')
            part = torch.tensor([tuple(map(int, point.split(','))) for
                                 point in coords['points'].split()])
            mask_regions.append(part)

        if tag in ['heading', 'article_', 'caption', 'paragraph']:
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
                polygon = text_line.find('Coords')
                baseline = text_line.find('Baseline')
                if baseline:
                    # get and shift baseline
                    line = torch.tensor([tuple(map(int, point.split(','))) for
                                         point in baseline['points'].split()])[:, torch.tensor([1, 0])]

                    if create_subimages:
                        line -= part[:2].unsqueeze(0)

                    region_dict['baselines'].append(line)  # type: ignore

                    # get mask
                    polygon_pt = torch.tensor([tuple(map(int, point.split(','))) for
                                               point in polygon['points'].split()])[:,
                                 torch.tensor([1, 0])]

                    # move mask to be in subimage
                    if create_subimages:
                        polygon_pt -= part[:2].unsqueeze(0)

                    # calc bbox for line
                    box = torch.tensor(get_bbox(polygon_pt))[torch.tensor([1, 0, 3, 2])]
                    box = box.clip(min=0)

                    # add bbox to data
                    if is_valid(box):
                        region_dict['bboxes'].append(box)  # type: ignore

                        # add mask to data
                        region_dict['masks'].append(polygon_pt)  # type: ignore

            if region_dict['bboxes']:
                region_dict['bboxes'] = torch.stack(region_dict['bboxes'])  # type: ignore
                paragraphs.append(region_dict)

    return paragraphs, mask_regions


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
        ax.add_patch(patches.Polygon(mask, fill=True, color=colors[i % 2], linewidth=.1, alpha=0.5))

    colors = ['red', 'blue']
    for i, box in enumerate(target['bboxes']):
        ax.add_patch(patches.Rectangle(box[:2], box[3] - box[1], box[2] - box[0],
                                       fill=False,
                                       color=colors[i % 2],
                                       linewidth=.1))

    for line in target['baselines']:
        ax.add_patch(patches.Polygon(line, fill=False, color='black', linewidth=.1, closed=False))

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


def plot_target(image, target, figsize=(50, 10), dpi=500):
    # Plot the first image
    attributes = ['ascenders', 'descenders', 'baselines', 'marker']
    image = image * target[:, :, 4, None]

    fig, axes = plt.subplots(1, len(attributes), figsize=figsize)  # Adjust figsize as needed

    for i, attribute in enumerate(attributes):
        axes[i].imshow(image.astype(np.uint8))
        axes[i].imshow(target[:, :, i].astype(np.uint8), cmap='gray', alpha=0.5)
        axes[i].set_title(attribute, fontsize=26)
        axes[i].axis('off')

    plt.tight_layout()  # Adjust layout
    plt.subplots_adjust(wspace=0.05)  # Adjust space between subplots

    # Display the plot with higher DPI
    plt.savefig(f'{Path(__file__).parent.absolute()}/../../data/Example/TargetExample2.png',
                dpi=dpi)
    plt.show(dpi=dpi)


def main(image_path: str, target_path: str, output_path: str, create_baseline_target: bool,
         create_subimages: bool) -> None:
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
        regions, mask_regions = extract(tar_path, create_subimages=create_subimages)

        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        image = to_tensor(image).permute(1, 2, 0).to(torch.uint8)

        if create_subimages:
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
                    target = draw_baseline_target(subimage.shape[:2], region['baselines'],
                                                  region['masks'], document_name)
                    np.savez_compressed(f"{output_path}/{document_name}/region_{i}/baselines",
                                        array=target)
        else:
            masks = []
            baselines = []
            bboxes = []
            for region in regions:
                # save target information
                masks.extend(region['masks'])
                baselines.extend(region['baselines'])
                bboxes.extend(region['bboxes'])

            torch.save(masks, f"{output_path}/{document_name}/masks.pt")
            torch.save(baselines, f"{output_path}/{document_name}/baselines.pt")
            torch.save(torch.stack(bboxes), f"{output_path}/{document_name}/bboxes.pt")

            if create_baseline_target:
                target = draw_baseline_target(image.shape[:2], baselines, masks, mask_regions, document_name)
                np.savez_compressed(f"{output_path}/{document_name}/baselines",
                                    array=target)


if __name__ == "__main__":
    # main(f'{Path(__file__).parent.absolute()}/../../data/Newspaper/newspaper-dataset-main-images/images',
    #      f'{Path(__file__).parent.absolute()}/../../data/Newspaper/pero_lines_bonn_regions',
    #      f'{Path(__file__).parent.absolute()}/../../data/Newspaper/preprocessed',
    #      create_baseline_target=True,
    #      create_subimages=False)

    test_array = np.load(
        f'{Path(__file__).parent.absolute()}/../../data/Newspaper/preprocessed/Koelnische_Zeitung_1924 - 0008/baselines.npz')[
        'array']
    image = io.imread(
        f'{Path(__file__).parent.absolute()}/../../data/Newspaper/newspaper-dataset-main-images/images/Koelnische_Zeitung_1924 - 0008.jpg')
    plot_target(image, test_array)

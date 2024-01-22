import glob
import shutil
from pathlib import Path


def main():
    images = f"{Path(__file__).parent.absolute()}/../../immediat-tables/images"
    annotations = f"{Path(__file__).parent.absolute()}/../../immediat-tables/annotations"

    target_image = f"{Path(__file__).parent.absolute()}/../data/Tabels/images"
    target_anno = f"{Path(__file__).parent.absolute()}/../data/Tabels/annotations"

    shutil.copytree(images, target_image)
    shutil.copytree(annotations, target_anno)


if __name__ =='__main__':
    main()
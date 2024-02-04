"""This is an example test."""
from pathlib import Path

import torch

from src.TableExtraction.customdataset import CustomDataset
from src.TableExtraction.trainer import get_model


def test() -> bool:
    """Tests the general functionality of predicting tables."""
    # Implement a test here.

    model = get_model('cell')

    dataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/TestData/",
        'cell',
        transforms=None,
    )

    img, target = dataset[0]

    assert isinstance(img, torch.Tensor), f"dataset doesn't return torch tensor got {type(img)} instead!"
    assert img.shape == torch.Size([3, 438, 1144]), f"dataset doesn't return torch tensor got {type(img)} instead!"
    assert list(target.keys()) == ['boxes', 'labels', 'img_number'], (f"dataset doesn't return the right dictionary. "
                                                                      f"Dictionary need to have the keys: "
                                                                      f"['boxes', 'labels', 'img_number'], but got: "
                                                                      f"{list(target.keys())}.")

    assert target['img_number'] == 'I_HA_Rep_89_Nr_16160_0089', (
        f"Target dictionary doesn't have right 'img_number'. Needs to be"
        f"'TextExample', but got {target['img_number']} instead.")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    output = model([img])[0]

    boxes = torch.tensor([[350., 43., 359., 53.],
                          [581., 42., 589., 48.],
                          [908., 18., 1109., 414.],
                          [307., 265., 315., 273.]])

    scores = torch.tensor([0.1301, 0.0752, 0.0747, 0.0516])

    label = torch.tensor([16, 16, 85, 16])

    assert torch.all(torch.round(output['boxes']).eq(boxes)), f"Predicted boxes differ from expected!"
    assert torch.all(torch.round(output['scores'], decimals=4).eq(scores)), f"Predicted scores differ from expected!"
    assert torch.all(output['labels'].eq(label)), f"Predicted labels differ from expected!"

    return True


if __name__ == '__main__':
    test()

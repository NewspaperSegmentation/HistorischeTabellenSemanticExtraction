import os
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

from src.customdataset import CustomDataset
from src.utils.metrics import weightedF1, calc_stats, calc_metrics
from src.utils.utils import show_prediction


def evaluation(model, dataset: CustomDataset, name: str, cuda: int = 0):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(f"cuda:{cuda}") if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    df = pd.DataFrame(
        columns=['image_number', 'mean_pred_iou', 'mean_target_iuo', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1',
                 'wf1'])
    os.makedirs(f"{Path(__file__).parent.absolute()}/../logs/evaluation/{name}/", exist_ok=True)

    all_tp = torch.zeros(5)
    all_fp = torch.zeros(5)
    all_fn = torch.zeros(5)

    idx = 0
    for img, target in tqdm(dataloader, desc='evaluation'):
        img = img.to(device)

        output = model([img[0]])
        output = {k: v.detach().cpu() for k, v in output[0].items()}
        target = {k: v[0] for k, v in target.items()}

        result = show_prediction(img[0].detach().cpu(), output['boxes'], target)
        plt.imshow(result.permute(1, 2, 0))
        # plt.axis('off')
        # plt.savefig(f"{Path(__file__).parent.absolute()}/../logs/evaluation/{name}/{target['img_number']}.png")

        tp, fp, fn, mean_pred_iou, mean_target_iuo = calc_stats(output['boxes'], target['boxes'])
        precision, recall, f1 = calc_metrics(tp, fp, fn)
        wf1 = weightedF1(f1)

        all_tp += tp
        all_fp += fp
        all_fn += fn

        metrics = pd.DataFrame({'image_number': target['img_number'],
                                'mean_pred_iou': mean_pred_iou.item(),
                                'mean_target_iuo': mean_target_iuo.item(),
                                'tp': [x.item() for x in list(tp)],
                                'fp': [x.item() for x in list(fp)],
                                'fn': [x.item() for x in list(fn)],
                                'precision': [x.item() for x in list(precision)],
                                'recall': [x.item() for x in list(recall)],
                                'f1': [x.item() for x in list(f1)],
                                'wf1': wf1.item()})

        df = pd.concat([df, metrics])

        idx += 1

    all_precision, all_recall, all_f1 = calc_metrics(all_tp, all_fp, all_fn)
    all_wf1 = weightedF1(all_f1)

    with open(f"{Path(__file__).parent.absolute()}/../logs/evaluation/{name}/{name}_overview.txt", "w") as f:
        f.write(f"true positives: {[x.item() for x in list(all_tp)]}\n")
        f.write(f"false positives: {[x.item() for x in list(all_fp)]}\n")
        f.write(f"false negatives: {[x.item() for x in list(all_fn)]}\n")
        f.write(f"precision: {[x.item() for x in list(all_precision)]}\n")
        f.write(f"recall: {[x.item() for x in list(all_recall)]}\n")
        f.write(f"F1 score: {[x.item() for x in list(all_f1)]}\n")
        f.write(f"weighted F1 score: {all_wf1=}\n")

    df.to_csv(f"{Path(__file__).parent.absolute()}/../logs/evaluation/{name}/{name}.csv")


if __name__ == '__main__':
    name = "run_tables1_es"
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.load_state_dict(torch.load(f'{Path(__file__).parent.absolute()}/../models/{name}.pt'))

    validdataset = CustomDataset(f'{Path(__file__).parent.absolute()}/../data/GloSAT/valid', 'tables')
    print(f"{len(validdataset)=}")

    evaluation(model, validdataset, name=name)

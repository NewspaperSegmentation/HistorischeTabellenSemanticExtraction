import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from src.customdataset import CustomDataset

LR = 0.002


class Trainer:
    def __init__(self, model, traindataset: CustomDataset, testdataset: CustomDataset, optimizer, name: str, cuda: int = 0) -> None:
        """
        Trainer class to train model
        :param model: model to train
        :param traindataset: dataset to train on
        :param testdataset: dataset to valid model while trainingsprocess
        :param optimizer: optimizer to use
        :param name: name of the model in savefiles and tensorboard
        :param cuda: number of used cuda device
        """
        self.device = torch.device(f"cuda:{cuda}") if torch.cuda.is_available() else torch.device('cpu')
        print(f"using {self.device}")

        self.model = model.to(self.device)
        self.trainloader = DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=0)
        self.testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)
        self.optimizer = optimizer(self.model.parameters(), lr=LR)

        self.bestavrgloss = None
        self.step = 0
        self.name = name

        # setup tensor board
        train_log_dir = f'{Path(__file__).parent.absolute()}/../logs/runs/{self.name}'
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)

    def save(self, name: str = ''):
        """
        save the model in models folder
        :param name: name of the model
        :return:
        """
        os.makedirs(f'{Path(__file__).parent.absolute()}/../models/', exist_ok=True)
        torch.save(self.model.state_dict(), f'{Path(__file__).parent.absolute()}/../models/{name}')

    def train(self, epoch: int):
        """
        train model for number of epochs
        :param epoch: number of epochs
        :return:
        """
        for e in range(1, epoch + 1):
            print(f"start epoch {e}:")
            self.train_epoch()
            avgloss = self.valid()
            if self.bestavrgloss is None or self.bestavrgloss > avgloss:
                self.bestavrgloss = avgloss
                self.save(f"{self.name}_es.pt")

    def train_epoch(self):
        """
        train one epoch
        :return:
        """
        for img, target in tqdm(self.trainloader):
            img.to(self.device)
            # target.to(self.device)

            d = {}
            boxes = target[0]
            boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
            d['boxes'] = boxes
            d['labels'] = torch.zeros(11, dtype=torch.int64)

            self.optimizer.zero_grad()
            output = model([img[0]], [d])
            loss = sum(v for v in output.values())
            loss.backward()
            self.optimizer.step()

            # logging
            self.writer.add_scalar(f'Training/loss', loss.detach().cpu().item(), global_step=self.step)
            self.writer.add_scalar(f'Training/loss_classifier', output['loss_classifier'].detach().cpu().item(),
                                   global_step=self.step)
            self.writer.add_scalar(f'Training/loss_box_reg', output['loss_box_reg'].detach().cpu().item(),
                                   global_step=self.step)
            self.writer.add_scalar(f'Training/loss_objectness', output['loss_objectness'].detach().cpu().item(),
                                   global_step=self.step)
            self.writer.add_scalar(f'Training/loss_rpn_box_reg', output['loss_rpn_box_reg'].detach().cpu().item(),
                                   global_step=self.step)
            self.writer.flush()
            self.step += 1

            del img, target, output, loss

    def valid(self):
        """
        valid current model on validation set
        :return: current loss
        """
        loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = [], [], [], [], []

        for img, target in self.testloader:
            img.to(self.device)
            target.to(self.device)
            output = self.model(img, target)

            loss.append(sum(v for v in output.values()).cpu().detach())
            loss_classifier.append(output['loss_classifier'].detach().cpu().item())
            loss_box_reg.append(output['loss_box_reg'].detach().cpu().item())
            loss_objectness.append(output['loss_objectness'].detach().cpu().item())
            loss_rpn_box_reg.append(output['loss_rpn_box_reg'].detach().cpu().item())

            del img, target, output

        meanloss = np.mean(loss)

        # logging
        self.writer.add_scalar(f'Valid/loss', meanloss, global_step=self.step)
        self.writer.add_scalar(f'Valid/loss_classifier', np.mean(loss_classifier),
                               global_step=self.step)
        self.writer.add_scalar(f'Valid/loss_box_reg', np.mean(loss_box_reg),
                               global_step=self.step)
        self.writer.add_scalar(f'Valid/loss_objectness', np.mean(loss_objectness),
                               global_step=self.step)
        self.writer.add_scalar(f'Valid/loss_rpn_box_reg', np.mean(loss_rpn_box_reg),
                               global_step=self.step)
        self.writer.flush()

        return meanloss


if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    traindataset = CustomDataset(f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed', 'tables')
    validdataset = CustomDataset(f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed', 'tables')
    optimizer = AdamW
    name = 'test2'

    trainer = Trainer(model, traindataset, validdataset, optimizer, name)
    trainer.train(1)

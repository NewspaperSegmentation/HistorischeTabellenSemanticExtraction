import os
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from src.customdataset import CustomDataset
from src.utils.utils import show_prediction

LR = 0.00001


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
        self.trainloader = DataLoader(traindataset, batch_size=1, shuffle=False, num_workers=0)
        self.testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)
        self.optimizer = optimizer(self.model.parameters(), lr=LR)

        self.bestavrgloss = None
        self.epoch = 0
        self.name = name

        # setup tensor board
        train_log_dir = f'{Path(__file__).parent.absolute()}/../logs/runs/{self.name}'
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)

        self.example_image, self.example_target = testdataset[0]
        self.train_example_image, self.train_example_target = traindataset[0]

    def save(self, name: str = ''):
        """
        save the model in models folder
        :param name: name of the model
        :return:
        """
        os.makedirs(f'{Path(__file__).parent.absolute()}/../models/', exist_ok=True)
        torch.save(self.model.state_dict(), f'{Path(__file__).parent.absolute()}/../models/{name}')

    def load(self, name: str = ''):
        """
        load the given model
        :param name: name of the model
        :return:
        """
        self.model.load_state_dict(torch.load(f'{Path(__file__).parent.absolute()}/../models/{name}.pt'))

    def train(self, epoch: int):
        """
        train model for number of epochs
        :param epoch: number of epochs
        :return:
        """
        for self.epoch in range(1, epoch + 1):
            print(f"start epoch {self.epoch}:")
            self.train_epoch()
            avgloss = self.valid()

            # early stopping
            if self.bestavrgloss is None or self.bestavrgloss > avgloss:
                self.bestavrgloss = avgloss
                self.save(f"{self.name}_es.pt")

        # save model after training
        self.save(f"{self.name}_end.pt")

    def train_epoch(self):
        """
        train one epoch
        :return:
        """
        loss_lst = []
        loss_classifier_lst = []
        loss_box_reg_lst = []
        loss_objectness_lst = []
        loss_rpn_box_reg_lst = []

        for img, target in tqdm(self.trainloader, desc='training'):
            img = img.to(self.device)
            target['boxes'] = target['boxes'][0].to(self.device)
            target['labels'] = target['labels'][0].to(self.device)

            self.optimizer.zero_grad()
            output = model([img[0]], [target])
            loss = sum(v for v in output.values())
            loss.backward()
            self.optimizer.step()

            loss_lst.append(loss.detach().cpu().item())
            loss_classifier_lst.append(output['loss_classifier'].detach().cpu().item())
            loss_box_reg_lst.append(output['loss_box_reg'].detach().cpu().item())
            loss_objectness_lst.append(output['loss_objectness'].detach().cpu().item())
            loss_rpn_box_reg_lst.append(output['loss_rpn_box_reg'].detach().cpu().item())

            del img, target, output, loss

        # logging
        self.writer.add_scalar(f'Training/loss', np.mean(loss_lst), global_step=self.epoch)
        self.writer.add_scalar(f'Training/loss_classifier', np.mean(loss_classifier_lst), global_step=self.epoch)
        self.writer.add_scalar(f'Training/loss_box_reg', np.mean(loss_box_reg_lst), global_step=self.epoch)
        self.writer.add_scalar(f'Training/loss_objectness', np.mean(loss_objectness_lst), global_step=self.epoch)
        self.writer.add_scalar(f'Training/loss_rpn_box_reg', np.mean(loss_rpn_box_reg_lst), global_step=self.epoch)
        self.writer.flush()

        del loss_lst, loss_classifier_lst, loss_box_reg_lst, loss_objectness_lst, loss_rpn_box_reg_lst

    def valid(self):
        """
        valid current model on validation set
        :return: current loss
        """
        loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = [], [], [], [], []

        for img, target in tqdm(self.testloader, desc='validation'):
            img = img.to(self.device)
            target['boxes'] = target['boxes'][0].to(self.device)
            target['labels'] = target['labels'][0].to(self.device)

            output = self.model([img[0]], [target])

            loss.append(sum(v for v in output.values()).cpu().detach())
            loss_classifier.append(output['loss_classifier'].detach().cpu().item())
            loss_box_reg.append(output['loss_box_reg'].detach().cpu().item())
            loss_objectness.append(output['loss_objectness'].detach().cpu().item())
            loss_rpn_box_reg.append(output['loss_rpn_box_reg'].detach().cpu().item())

            del img, target, output

        meanloss = np.mean(loss)

        # logging
        self.writer.add_scalar(f'Valid/loss', meanloss, global_step=self.epoch)
        self.writer.add_scalar(f'Valid/loss_classifier', np.mean(loss_classifier),
                               global_step=self.epoch)
        self.writer.add_scalar(f'Valid/loss_box_reg', np.mean(loss_box_reg),
                               global_step=self.epoch)
        self.writer.add_scalar(f'Valid/loss_objectness', np.mean(loss_objectness),
                               global_step=self.epoch)
        self.writer.add_scalar(f'Valid/loss_rpn_box_reg', np.mean(loss_rpn_box_reg),
                               global_step=self.epoch)
        self.writer.flush()

        self.model.eval()
        pred = self.model([self.example_image.to(self.device)])
        result = show_prediction(self.example_image, pred[0]['boxes'].detach().cpu(), self.example_target)
        self.writer.add_image("Valid/example", result[:, ::2, ::2], global_step=self.epoch)

        pred = self.model([self.train_example_image.to(self.device)])
        result = show_prediction(self.train_example_image, pred[0]['boxes'].detach().cpu(), self.train_example_target)
        self.writer.add_image("Training/example", result[:, ::2, ::2], global_step=self.epoch)

        self.model.train()
        return meanloss


if __name__ == '__main__':
    from torchvision import transforms
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    transform = torch.nn.Sequential(
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=(0.5, 1.5), saturation=(0, 2))]),
                               p=0.1),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=9, sigma=(2, 10))]), p=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
        transforms.RandomGrayscale(p=0.1)
    )
    traindataset = CustomDataset(f'{Path(__file__).parent.absolute()}/../data/GloSAT/train', 'tables', transforms=transform)
    validdataset = CustomDataset(f'{Path(__file__).parent.absolute()}/../data/GloSAT/valid', 'tables')
    print(f"{len(traindataset)=}")
    print(f"{len(validdataset)=}")
    optimizer = AdamW
    name = 'transforms_test'

    trainer = Trainer(model, traindataset, validdataset, optimizer, name)
    trainer.train(10)

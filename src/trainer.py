from torch.utils.data import DataLoader
import torch
import os
from pathlib import Path
import torchvision


LR = 0.002

class trainer():
    def __init__(self, model, traindataset, testdataset, optimizer, loss_fn, name, cuda=0) -> None:
        self.device = torch.device(f"cuda:{cuda}") if torch.cuda.is_available() else torch.device('cpu')
        print(f"using {self.device}")
        self.model = model.to(self.device)
        self.trainloader = DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=0)
        self.testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)
        self.optimizer = optimizer(self.model.parameters(), lr=LR)
        self.loss_fn = loss_fn
        self.bestavrgloss = None
        self.name = name

    def save(self, name: str = ''):
        """
        save the model in models folder
        :param name: name of the model
        :return:
        """
        os.makedirs(f'{Path(__file__).parent.absolute()}/../models/', exist_ok=True)
        torch.save(self.model.state_dict(), f'{Path(__file__).parent.absolute()}/../models/{name}')

    def train(self, epoch):
        for e in range(epoch):
            self.trainepoch()
            avrgloss=self.valid()
            if self.bestavrgloss==None or self.bestavrgloss>avrgloss:
                self.bestavrgloss = avrgloss
                self.save(f"{self.name}_es.pt")

    def trainepoch(self):
        avrgloss = []
        for img,target in self.trainloader:
            img.to(self.device)
            target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(img)
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()
            avrgloss.append(loss.cpu().detach())
            del img, target, pred, loss
        avgloss = torch.mean(avrgloss)

    def valid(self):
        avrgloss = []
        for img,target in self.testloader:
            img.to(self.device)
            target.to(self.device)
            pred = self.model(img)
            loss = self.loss_fn(pred, target)
            avrgloss.append(loss.cpu().detach())
            del img, target, pred, loss
        avgloss = torch.mean(avrgloss)
        return avgloss

if __name__ == '__main__':
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
  traindataset = dataset(f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed', 'tables')
  validdataset = dataset(f'{Path(__file__).parent.absolute()}/../data/GloSAT/preprocessed', 'tables')
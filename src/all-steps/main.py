import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision import datasets, transforms
import lightning as L


class Net(L.LightningModule):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.args = args

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=self.args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class DataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.train_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': True,
        }

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        datasets.MNIST('../data', download=True)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset1 = datasets.MNIST(
            '../data', train=True, transform=self.transform)
        self.train_set = dataset1

    def train_dataloader(self):
        train_loader = data.DataLoader(self.train_set, **self.train_kwargs)
        return train_loader


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Lightning MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    args = parser.parse_args()

    model = Net(args)
    datamodule = DataModule(args)

    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()

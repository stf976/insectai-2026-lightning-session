import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision import datasets, transforms
import torchvision
import torchmetrics
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


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

        self.val_acc = torchmetrics.Accuracy(
            task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(
            task='multiclass', num_classes=10)

        self.save_hyperparameters()

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

        self.log("train/loss", loss, on_step=True,
            on_epoch=False, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.log_images(data[0:6], 'train/images', self.global_step)

        return {'loss': loss}

    def log_images(self, images, label, step):
        grid = torchvision.utils.make_grid(images)
        self.logger: TensorBoardLogger
        self.logger.experiment.add_image(label, grid, step)

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)

        predictions = self.predict(output)
        self.val_acc(predictions, target)
        self.log_dict({'val/loss': loss, 'val/acc': self.val_acc},
            on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'val/loss': loss}

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=self.args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)

        predictions = self.predict(output)
        self.test_acc(predictions, target)
        self.log_dict({'test/loss': loss, 'test/acc': self.test_acc},
            on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log_images(data[0:6], 'test/images', batch_idx)

    def predict(self, output):
        predictions = torch.argmax(output, dim=1)
        return predictions


class DataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.train_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 7,
        }
        self.test_kwargs = {
            'batch_size': args.test_batch_size,
            'num_workers': 7,
        }

        if not args.no_accel and torch.accelerator.is_available():
            accel_kwargs = {
                'persistent_workers': True,
                'pin_memory': True,
            }
            self.train_kwargs.update(accel_kwargs)
            self.test_kwargs.update(accel_kwargs)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.save_hyperparameters()

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        datasets.MNIST('../data', download=True)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset1 = datasets.MNIST(
            '../data', train=True, transform=self.transform)
        self.train_set, self.val_set = data.random_split(dataset1, [0.8, 0.2])
        dataset2 = datasets.MNIST(
            '../data', train=False, transform=self.transform)
        self.test_set = dataset2

    def train_dataloader(self):
        train_loader = data.DataLoader(self.train_set, **self.train_kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = data.DataLoader(self.val_set, **self.test_kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = data.DataLoader(self.test_set, **self.test_kwargs)
        return test_loader


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Lightning MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = Net(args)
    datamodule = DataModule(args)

    tb_logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='lightning_logs'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
        monitor='val/loss',
        mode='min',
        filename='epoch={epoch}-step={step}-val_loss={val/loss:.4f}',
        auto_insert_metric_name=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor='val/acc',
        mode='max',
        min_delta=0.001,
        patience=10,
        verbose=True,
    )
    trainer = L.Trainer(
        accelerator='cpu' if args.no_accel else 'auto',
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=args.epochs,
        log_every_n_steps=args.log_interval,
        enable_model_summary=False,
    )
    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(
        ckpt_path='best',
        datamodule=datamodule,
        # only for trusted sources!
        weights_only=False
    )

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

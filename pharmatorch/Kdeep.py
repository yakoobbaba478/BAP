import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../../pharmatorch')
from datasets.dockingdata import DockingDataset
from models.docking.squeezenet import SqueezeNet
import torch.nn.functional as F
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
import csv
import pandas as pd
from sklearn.metrics import r2_score

class Squeezenetmodel(pl.LightningModule):

    def __init__(self,hparams):
        super(Squeezenetmodel,self).__init__()
        self.root = hparams.root
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr
        #self.split = hparams.split
        self.dataset = DockingDataset(self.root)

        self.trainset = self.dataset[:3126]
        self.testset = self.dataset[3126:]
        #self.valset = self.dataset[3538:]
        self.net = SqueezeNet()
    def forward(self, x):
        return self.net(x)
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt] ,[sch]
    def train_dataloader(self):
        return DataLoader(self.trainset,batch_size=self.batch_size, num_workers=8)
    def test_dataloader(self):
        return DataLoader(self.testset,batch_size=32, num_workers=8)
    #def val_datalaoder(self):
    #    return DataLoader(self.valset,batch_size=self.batch_size, num_workers=8)

    def training_step(self, batch, batch_nb):
        x,y = batch[0],batch[1]
        y_hat = self.forward(x)
        #print('shapes:', y.shape, y_hat.shape)
        loss = F.mse_loss(y,y_hat.squeeze(1))
        tensorboard_logs = {'train_loss': loss}
        x = {'loss': loss, 'progress_bar': {'training_loss': loss}, 'log': tensorboard_logs}
        #print(x)
        return {'loss': loss, 'progress_bar': {'training_loss': loss}, 'log': tensorboard_logs}
    #def validation_step(self, batch, batch_nb):
    #    x, y = batch[0], batch[1]
    #    y_hat = self.forward(x)
    #    val_loss = F.mse_loss(y, y_hat)
    #    return {'val_loss': val_loss}
    def test_step(self, batch, batch_nb):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        #print('y:',y,'y_hat:',y_hat)
        loss = F.mse_loss(y_hat.squeeze(1), y)
        r2 = r2_score(y, y_hat.squeeze(1))
        print('r2_score', r2)
        return {'test_loss': F.mse_loss(y, y_hat.squeeze(1))}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default='datasets/refined-set',
                        help="path where dataset is stored")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    hparams = parser.parse_args()
    model = Squeezenetmodel(hparams)
    trainer = pl.Trainer(max_nb_epochs=15)
    trainer.fit(model)
    trainer.save_checkpoint("kdeepmodel.ckpt")
    #model = Sqeezenetmodel.load_from_checkpoint(checkpoint_path="kdeepmodel.ckpt")
    #model = Squeezenetmodel(hparams)
    checkpoint = torch.load('kdeepmodel.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    trainer = pl.Trainer()
    trainer.test(model)


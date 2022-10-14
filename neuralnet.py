import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl


class MoleculeDataset(Dataset):

    def __init__(self):
        x, y = get_features_target()
        self.x = torch.tensor(x.values).float()
        self.y = torch.tensor(y.values).float()
        print(self.x.shape)
        print(self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NeuralNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # if isinstance(hparams, dict):
        #     hparams = Namespace(**hparams)
        # self.save_hyperparameters(hparams)

        self.net = nn.Sequential(
            nn.LazyLinear(16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 1),
        )
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch):
        x, y = batch
        pred = self.net(x)
        return self.loss_function(y, pred)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters()) #, lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(MoleculeDataset(),
                          shuffle=True,
                          batch_size=64
        )



def get_features_target():
    df = pd.read_csv("feature_count_df.csv")
    print(df.columns)
    X = df.iloc[:,14:]
    y = df["mLogD7.4"]
    return X, y


if __name__ == '__main__':
    print(get_features_target())

    model = NeuralNet()
    trainer = pl.Trainer()
    trainer.fit(model)

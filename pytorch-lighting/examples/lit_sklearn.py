from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score


class NDArrayDataset(Dataset):
    def __init__(self, X, Y):
        super(NDArrayDataset, self).__init__()
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': torch.from_numpy(self.X[idx,:]),
            'Y': torch.from_numpy(self.Y[idx,:])
        }

class DataFrameDataset(Dataset):
    def __init__(self, X, Y):
        super(DataFrameDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': torch.from_numpy(self.X.iloc[idx,:].values),
            'Y': torch.from_numpy(self.Y.iloc[idx,:].values)
        }

class SklearnClassifier(pl.LightningModule):
    def __init__(self, model):
        torch.set_grad_enabled(False)
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        
    def forward(self, X):
        # print('============= in forward')
        return torch.from_numpy(self.model.predict(X))
    
    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        pass

    def _loss(self, batch, batch_idx):
        with torch.no_grad():
            X, Y = batch['X'].detach().numpy(), batch['Y'].detach().numpy()
            Y_pred = self.model.predict(X)
            loss = accuracy_score(Y, Y_pred)
        return torch.from_numpy(np.array(loss)), torch.from_numpy(Y), torch.from_numpy(Y_pred)

    def _epoch_end_accuracy(self, outputs):
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        acc = accuracy_score(y_true, y_pred)
        return acc
    
    def training_step(self, batch, batch_idx):
        loss, Y, Y_pred = self._loss(batch, batch_idx)
        self.log('train_loss', loss)
        return {
            'loss': loss,
            'y_true': Y,
            'y_pred': Y_pred
        }

    def training_epoch_end(self, outputs):
        acc = self._epoch_end_accuracy(outputs)
        self.log('train_acc', acc)
    
    def validation_step(self, batch, batch_idx):
        loss, Y, Y_pred = self._loss(batch, batch_idx)
        self.log('val_loss', loss)
        return {
            'loss': loss,
            'y_true': Y,
            'y_pred': Y_pred
        }

    def validation_epoch_end(self, outputs):
        acc = self._epoch_end_accuracy(outputs)
        self.log('val_acc', acc)
    
    def test_step(self, batch, batch_idx):
        loss, Y, Y_pred = self._loss(batch, batch_idx)
        self.log('test_loss', loss) 
        return {
            'loss': loss,
            'y_true': Y,
            'y_pred': Y_pred
        }

    def test_epoch_end(self, outputs):
        acc = self._epoch_end_accuracy(outputs)
        self.log('test_acc', acc)
    
    def configure_optimizers(self):
        pass


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int) 
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # init neptune experiment
    import os
    from utils import init_neptune
    api_key = os.environ['NEPTUNE_API_TOKEN']
    project_name = 'zzragida/examples-lit-sklearn'
    experiment_name = 'pytorch-lightning-sklearn'
    experiment_tags = ['lightning', 'sklearn']

    neptune_logger = init_neptune(args, api_key, project_name, experiment_name, experiment_tags)

    # ------------
    # data
    # ------------
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # train/test split
    Y_col = ['target']
    X_col = [col for col in df.columns if not col in Y_col]
    X_train, X_test, Y_train, Y_test = train_test_split(df[X_col], df[Y_col], test_size=0.2, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train[X_col], Y_train[Y_col], test_size=0.2, random_state=0)

    train_ds = DataFrameDataset(X_train, Y_train)
    val_ds = DataFrameDataset(X_val, Y_val)
    test_ds = DataFrameDataset(X_test, Y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, Y_train)
    print(accuracy_score(Y_test, clf.predict(X_test).ravel()))
    model = SklearnClassifier(clf)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, max_epochs=1, logger=neptune_logger)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()

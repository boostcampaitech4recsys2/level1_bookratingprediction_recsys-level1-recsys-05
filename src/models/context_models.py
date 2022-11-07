import tqdm
import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from ._models import _FactorizationMachineModel, _FieldAwareFactorizationMachineModel
from ._models import rmse, RMSELoss

######################## IMPORT WANDB MODULE
import wandb
########################


class FactorizationMachineModel:
    
    def __init__(self, args, train_dataset, valid_dataset, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.field_dims = data['field_dims']

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100
        self.batch_size = args.BATCH_SIZE
        self.shuffle = args.DATA_SHUFFLE

        self.device = args.DEVICE

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
        
    def train(self):
        # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        
        kfold = KFold(n_splits = 5, shuffle = True)
        validation_loss = []
        total_mean = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.train_dataset)):
            # print(train_idx,val_idx)
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            # print(train_subsampler, val_subsampler)
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler = train_subsampler)
            valid_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler = val_subsampler)


            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0
                tk0 = tqdm.tqdm(train_dataloader, smoothing=0, mininterval=1.0)
                for i, (fields, target) in enumerate(tk0):
                    print(fields, target)
                    self.model.zero_grad()
                    fields, target = fields.to(self.device), target.to(self.device)

                    y = self.model(fields)
                    loss = self.criterion(y, target.float())

                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    if (i + 1) % self.log_interval == 0:
                        tk0.set_postfix(loss=total_loss / self.log_interval)
                        total_loss = 0
                    
                    wandb.log({"loss": total_loss}, step = epoch)
                    rmse_score = self.predict_train(valid_dataloader)
                    wandb.log({"rmse": rmse_score}, step = epoch)
                    print('k-fold:',fold,'epoch:', epoch, 'validation: rmse:', rmse_score)
                    validation_loss.append(rmse_score)
        validation_loss = np.array(validation_loss)
        mean = np.mean(validation_loss)
        std = np.std(validation_loss)
        wandb.log({"k-fold rmse mean": mean}, step = fold)
        wandb.log({"k-fold rmse std": std}, step = fold)

        total_mean.append(mean)
        total_mean = np.array(total_mean)

        return np.mean(total_mean)

    def predict_train(self, valid_dataloader):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class FieldAwareFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        
        ######################## WANDB INIT
        run = wandb.init(
            project="schini-FFM-test",
            entity="boostcamp_l1_recsys05",
            name="FFM-rmse",
        )
        ########################
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            ######################## WANDB RUN
            wandb.log({'loss': total_loss, 'RMSE': rmse_score}, step=epoch)
            ########################
            print('epoch:', epoch, 'validation: rmse:', rmse_score)
        
        ######################## WANDB FINISH
        run.finish()
        ########################


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts

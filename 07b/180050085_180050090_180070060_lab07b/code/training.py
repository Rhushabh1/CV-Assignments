# from google.colab import drive
# drive.mount('/content/gdrive',force_remount=True)


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cal_mpjpe(pose_1, pose_2, avg=True):
    n_joints = list(pose_1.shape)[1]
    batch_size = list(pose_1.shape)[0]
    pose_1 = pose_1.view(batch_size, n_joints, 3)
    pose_2 = pose_2.view(batch_size, n_joints, 3)
    diff = pose_1-pose_2
    diff_sq = diff ** 2
    dist_per_joint = torch.sqrt(torch.sum(diff_sq, axis=2))
    dist_per_sample = torch.mean(dist_per_joint, axis=1)
    if avg is True:
        dist_avg = torch.mean(dist_per_sample)
    else:
        dist_avg = dist_per_sample
    return dist_avg


class BasicBlock(nn.Module):
    
    def __init__(self, hidden_layer = 1024, dropout = 0.1):
        super(BasicBlock, self).__init__()
        # input = (1, 1024)
        self.fc1 = nn.Linear(hidden_layer, hidden_layer)
        self.bn1 = nn.BatchNorm1d(hidden_layer)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p = dropout)
        
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.bn2 = nn.BatchNorm1d(hidden_layer)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p = dropout)
        
    def forward(self, x: Tensor):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out += x
        return out


class LiftModel(nn.Module):
    
    def __init__(self, n_blocks = 2, hidden_layer = 1024, dropout = 0.1, output_nodes = 15*3):
        super(LiftModel, self).__init__()
        self.n_blocks = n_blocks
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.output_nodes = output_nodes
        
        # input = (1, 15, 2)
        self.flat = nn.Flatten()
        self.linear_1 = nn.Linear(15*2, hidden_layer)
        self.blocks = [BasicBlock(hidden_layer, dropout) for _ in range(n_blocks)]        
        self.blocks = nn.ModuleList(self.blocks)    
        self.linear_2 = nn.Linear(hidden_layer, output_nodes)
        
    def forward(self, x: Tensor):
        out = self.linear_1(self.flat(x))
        
        for i in range(len(self.blocks)):
            out = self.blocks[i](out)
            
        out = self.linear_2(out)
        return out
    
    def compute_loss(self, batch, criterion):
        poses, labels = batch
        out = self(poses.to(device))
        loss = criterion(labels.to(device), out)
        return loss


def evaluate(model, loader, criterion):
    outputs = [model.compute_loss(batch, criterion).detach() for batch in loader]
    outputs = torch.tensor(outputs)
    loss = torch.mean(outputs)
    return loss.item()


def fit(model, train_loader, val_loader, epochs, optimiser, criterion, scheduler):
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):    

        loss_items = []
        # train
        for batch_idx, batch in enumerate(train_loader):
            loss = model.compute_loss(batch, criterion)
            
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            if batch_idx % 100 == 0:
                print("Epoch [{}/{}]    Step [{}/{}]    MPJPE: {:.5f}".format(epoch+1, epochs, batch_idx+1, len(train_loader), loss.item()))
            loss_items.append(loss.item())
            
        scheduler.step()
        
        # validate
        result = evaluate(model, val_loader, criterion)
        print("Epoch [{}/{}]    val_MPJPE: {:.5f}\n".format(epoch+1, epochs, result))
        history["val_loss"].append(result)
        history["train_loss"].append(sum(loss_items)/len(loss_items))
        
    return history


def run_epoch(epoch_no, data, model, optimiser, scheduler, batch_size = 64, split = 'train'):
    focal_len = data["focal_len_1"]
    joint_2d = data["joint_2d_1"]
    joint_3d = data["joint_3d"]

    DATA_SIZE = focal_len.shape[0]
    SPLIT = 0.1
    VAL_SIZE = int(SPLIT * DATA_SIZE)
    TRAIN_SIZE = DATA_SIZE - VAL_SIZE
    BATCH_SIZE = batch_size

    joint_2d_tensor = torch.Tensor(joint_2d)
    joint_3d_tensor = torch.Tensor(joint_3d)

    dataset = TensorDataset(joint_2d_tensor, joint_3d_tensor)

    train_set, val_set = random_split(dataset, [TRAIN_SIZE, VAL_SIZE])

    train_dl = DataLoader(train_set, 
                          batch_size = BATCH_SIZE,
                          shuffle = True, 
                         )
    val_dl = DataLoader(val_set, 
                          batch_size = BATCH_SIZE,
                          shuffle = True, 
                         )
    test_dl = DataLoader(dataset, 
                          batch_size = BATCH_SIZE,
                          shuffle = True, 
                         )

    EPOCHS = epoch_no

    criterion = cal_mpjpe
  
    if split == 'train':
        print("Running Epochs ...\n")
        history = fit(model, train_dl, val_dl, EPOCHS, optimiser, criterion, scheduler)
        return history
    else:
        result = evaluate(model, test_dl, criterion)
        print("Test MPJPE: {}".format(result))
        return result


if __name__ == "__main__":

    torch.manual_seed(0)

    with open("data_train_lift.pkl", 'rb') as f:
        data_train_lift = pickle.load(f)

    BATCH_SIZE = 64

    model = LiftModel(dropout = 0.5).to(device)

    EPOCHS = 25
    INIT_LR = 1e-3
    STEP_SIZE = 5
    GAMMA = 0.1

    optimiser = optim.Adam(model.parameters(), INIT_LR)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size = STEP_SIZE, gamma = GAMMA)

    history = run_epoch(EPOCHS, data_train_lift, model, optimiser, scheduler, batch_size = BATCH_SIZE)

    # plot training and validation loss

    plt.plot(history["train_loss"], label = "train_loss")
    plt.plot(history["val_loss"], label = "val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("MPJPE")
    plt.legend()
    plt.show()

    FILEPATH = "liftModel"
    torch.save(model.state_dict(), FILEPATH)
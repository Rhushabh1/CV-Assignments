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

from training import LiftModel, evaluate, cal_mpjpe, run_epoch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    torch.manual_seed(0)

    TEST_FILE = "data_test_lift.pkl"
    # TEST_FILE = "data_train_lift.pkl"
    with open(TEST_FILE, 'rb') as f:
        data = pickle.load(f)

    FILEPATH = "liftModel"

    model = LiftModel(dropout = 0.5).to(device)
    model.load_state_dict(torch.load(FILEPATH, map_location = device))

    BATCH_SIZE = 64

    focal_len = data["focal_len_1"]
    joint_2d = data["joint_2d_1"]
    joint_3d = data["joint_3d"]

    joint_2d_tensor = torch.Tensor(joint_2d)
    joint_3d_tensor = torch.Tensor(joint_3d)

    test_set = TensorDataset(joint_2d_tensor, joint_3d_tensor)

    test_dl = DataLoader(test_set, 
                          batch_size = BATCH_SIZE,
                          shuffle = True, 
                         )

    criterion = cal_mpjpe

    result = evaluate(model, test_dl, criterion)
    print("Test MPJPE: {:.5f}".format(result))
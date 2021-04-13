# -*- coding: utf-8 -*-
"""Lab7B.ipynb

Original file is located at
    https://colab.research.google.com/drive/19FhAN0dykWCW9ZvuRUIPUd5kD4pdsUzd
"""



import os
import pickle
import torch
from torch import nn
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.optim as optim

import numpy as np



"""
#***Weak Supervision***

Plan at hand : We use mpjpe for the expected vs actual 2-dimensional points instead.

We get the 3d point as the output from the model as we did before, and then pass it through the extrinsic and intrinsic camera matrix to obtain the expected joint_2d_2 """

class ResidualBlock(nn.Module):
    def __init__(self, hidden_layer, dropout):
        super(ResidualBlock, self).__init__()
        self.block = nn.ModuleList([
                                    nn.Linear(hidden_layer, hidden_layer),
                                    nn.BatchNorm1d(hidden_layer),
                                    nn.ReLU(),
                                    nn.Dropout(p = dropout),
                                    nn.Linear(hidden_layer, hidden_layer),
                                    nn.BatchNorm1d(hidden_layer),
                                    nn.ReLU(),
                                    nn.Dropout(p = dropout)]
                                )
    def forward(self, x):
        y = x
        for block in self.block:
            y = block(y)
        z = y + x
        return z
	



class LiftModel(nn.Module):
    def __init__(self, n_blocks = 2, hidden_layer = 1024, dropout = 0.1, output_nodes = 15*3):
        super(LiftModel, self).__init__()
        self.n_blocks, self.hidden_layer, self.dropout, self.output_nodes = n_blocks, hidden_layer, dropout, output_nodes
        prev_size = 15*2
        flatten = nn.Flatten()
        blocks = []
        first_flat_layer  = nn.Linear(prev_size, hidden_layer)
        for i in range(self.n_blocks):
            blocks.append(ResidualBlock(hidden_layer, dropout))
        hidden_layer = nn.Linear(hidden_layer, output_nodes)

        self.block = nn.ModuleList([flatten,first_flat_layer] +  blocks + [hidden_layer])
	
    def forward(self, x):
        y = x
        for block in self.block:
            y = block(y)
        return y

def cal_mpjpe(input, target, average = True):
    diff = input - target
    diff = diff ** 2
    diff = torch.sum(diff, 2)
    diff = diff ** 0.5
    tensor_shape = list(target.size())
    loss = torch.sum(diff)/(tensor_shape[1])
    if average == True:
        loss/=tensor_shape[0]

    return loss

def run_epoch_weak(epoch_no, data, model, optimiser, scheduler,batch_size=64, split='train'):

    if split == "train":
        model.train()
        inp = data['joint_2d_1']
        out = data ['joint_2d_2']
        rotation_matrix = data['rot'].reshape((data['joint_2d_1'].shape[0], 3,3))
        translation_matrix = data['transl'] 
        intrin_mat = np.zeros((data['joint_2d_1'].shape[0],3,3))
        intrin_mat[:,np.arange(2), np.arange(2)] = data['focal_len_2']
        intrin_mat[:,2,2] = 1
        running_loss = 0.0
        loss_data = []
        count = 0
        
        for no in range(epoch_no):
            running_loss = 0
            count = 0
            for i in range(0,inp.shape[0]-batch_size,batch_size):
                count += 1
                train_inp = torch.tensor(inp[i:i+batch_size])
                rot_tens = torch.tensor(rotation_matrix[i:i+batch_size])
                trans_tens = torch.tensor(translation_matrix[i:i+batch_size])
                int_tens = torch.tensor(intrin_mat[i:i+batch_size]).float()
                res = model(train_inp)


                res = res.view((train_inp.shape[0],15,3))

                res1 = torch.einsum('ijk,ilk->ilj',rot_tens, res)
                res1 += trans_tens.unsqueeze(1)
                final_res = torch.einsum ('ijk,ilk->ilj',int_tens, res1)
                final_res_1 = torch.div(final_res[:,:,0], final_res[:,:,2]).unsqueeze(2)
                final_res_2 = torch.div(final_res[:,:,1], final_res[:,:,2]).unsqueeze(2)
                two_d_out  = torch.cat((final_res_1, final_res_2), 2)

                exp_out = torch.tensor(out[i:i+batch_size])
                loss = cal_mpjpe(two_d_out, exp_out)

                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                running_loss += loss.item()
                loss_data.append(running_loss/count)

            scheduler.step()

            print(f"{no+1}", running_loss/count)
        return loss_data
    else:
        inp = data["joint_2d_1"]
        output = data["joint_3d"]
        running_loss = 0.0
        count = 0
        loss_data = []
        model.eval()
        for no in range(epoch_no):
            running_loss = 0.0
            count = 0
            for i in range(0,inp.shape[0]-batch_size,batch_size):
                count +=1 
                train_data = torch.tensor(inp[i:i + batch_size])
                target = torch.tensor(output[i:i + batch_size])
                my_outputs = model(train_data)
                my_outputs = my_outputs.view(target.size())
                loss = cal_mpjpe(my_outputs, target)
                running_loss += loss.item()
            print('%d], loss: %.3f' %
                    (no + 1, running_loss/count))
            loss_data.append(running_loss/count)
        
        return loss_data


def main_fn():
    path = "../"

    with open(os.path.join(os.path.join(path,"data") , "data_train.pkl"), 'rb') as f:
        data = pickle.load(f)

    model = LiftModel()
    dec = 1e-1
    num_epochs = 1
    optimizer = optim.Adadelta(model.parameters(), lr=1, rho=0.9, eps=1e-06, weight_decay=dec)
    scheduler = optim.lr_scheduler.StepLR(optimizer, num_epochs, 0.5) # Essentially no scheduler at all
    split = int(0.8 * data['joint_2d_1'].shape[0])
    training_data = {}
    test_data = {}
    for key in data:
        training_data[key] = data[key][:split]
    for key in data:
        test_data[key] = data[key][split:]
    loss_given = run_epoch_weak(num_epochs,training_data,model, optimizer, scheduler)
    torch.save(model.state_dict(), os.path.join(os.path.join(path,"results"), f"model_dec_{dec}"))

    from matplotlib import pyplot as plt
    zeros = np.arange(0, num_epochs, 1/(len(loss_given)/num_epochs))
    plt.plot(zeros,loss_given)
    plt.show()

    test_loss = run_epoch_weak(1,test_data, model, optimizer, scheduler, 64,"test")
    print(test_loss)


if __name__ == "__main__":
    main_fn()
"""## Losses Observed


1.   Regularization = 0.1 Train Loss = 0.23  Test Loss = 0.302
2.   Regularization = 0.01 Train Loss = 0.15  Test Loss = 0.42
3.   Regularization = 0.001 Train Loss = 0.1 Test Loss = 0.84

## Why add regularization?

We have the value z in the 3d coordinates that we get which is being divided to obtain the 2d coordinate and therefore to obtain the loss function while training. Hence the chances of exploding gradients and therefore exploding weights are high, therefore there must be some penalty for high weights to ensure the results obtained are appropriate. Else, while using no decay, we get a lower train loss ie. around 0.06 but a very high test loss at around 120. Using momentum might not be the best idea too over here, again due to gradient explosion which could lead to a dominating gradient for a particular weight.
"""
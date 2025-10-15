# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10 Part2. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

use_mps = torch.backends.mps.is_available()
print(use_mps)
device = torch.device("mps:0" if use_mps else "cpu")
torch.backends.cudnn.benchmark = True


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256


        # TODO Define the network architecture (layers) based on these specifications.

        
        self.in_size = in_size
        self.out_size = out_size
        
        h_1 = 128
        h_2 = 64
        self.cnn_block = nn.Sequential(
                            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
                            nn.LeakyReLU(),
                            nn.MaxPool2d(2,2),
                            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
                            nn.LeakyReLU(),
                            nn.MaxPool2d(2,2),
                            )
        
        self.linear_block = nn.Sequential(
                                nn.Linear(64 * 7 * 7, h_1),
                                nn.LeakyReLU(),
                                nn.Linear(h_1, h_2),
                                nn.LeakyReLU(),
                                nn.Linear(h_2, out_size)
                                )
        
        self.optimizer = optim.SGD(self.parameters(), lr = lrate)

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.
        reshaped_x = x.view(x.size(dim = 0), 3, 31, 31).to(device)
        return self.linear_block(self.cnn_block(reshaped_x).view(-1 , 64 * 7 * 7).to(device))

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
    
        # Important, detach and move to cpu before converting to numpy and then to python float.
        # Or just use .item() to convert to python float. It will automatically detach and move to cpu.
        self.optimizer.zero_grad()
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu().numpy())



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    # Important, don't forget to detach losses and model predictions and convert them to the right return types.
    labels = torch.unique(train_labels)
    net = NeuralNet(lrate = 0.1, loss_fn = nn.CrossEntropyLoss().to(device), in_size = train_set.size(dim = 1), out_size = labels.size(dim = 0))
    net.to(device)

    train_set_standrdized = (train_set - torch.mean(train_set)) / torch.std(train_set)
    dev_set_standrdized = (dev_set - torch.mean(dev_set)) / torch.std(dev_set)

    train = get_dataset_from_arrays(train_set_standrdized, train_labels)
    train_loader = DataLoader(train, batch_size, shuffle=False)

    losses = []
    for e in range(epochs):
        epoch_losses = []
        for batch in train_loader:
            torch.mps.empty_cache()
            x, y = batch.values()
            x.to(device)
            y.to(device)

            epoch_losses.append(net.step(x, y))
        
        losses.append(sum(epoch_losses) / len(epoch_losses))
        
    preds = torch.argmax(net.forward(dev_set_standrdized), dim = 1).detach().cpu().numpy()

    return losses, preds, net

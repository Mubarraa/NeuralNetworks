# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.l = nn.Linear(in_features=28*28, out_features=10, bias=True)        
        # INSERT CODE HERE

    def forward(self, x):        
        temp = x.view(x.shape[0], -1)
        result1 = F.log_softmax(self.l(temp), dim=1)
        return result1

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE

    def forward(self, x):
        return 0 # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE

    def forward(self, x):
        return 0 # CHANGE CODE HERE

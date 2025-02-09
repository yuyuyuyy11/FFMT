import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time

import math
import sys 
sys.path.append('/home/yue/Projects/pointMBF')
from utils.time import time_it

class MutualCrossAttention(nn.Module):
    '''
    Feature Fusion Based on Mutual-Cross-Attention Mechanism for EEG Emotion Recognition
    
    MCA is a purely mathematical method applying Attention Mechanism from each directions of two features.
    '''
    def __init__(self, dropout):
        super(MutualCrossAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # Assign x1 and x2 to query and key
        query = x1
        key = x2
        d = query.shape[-1]

        # Basic attention mechanism formula to get intermediate output A
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        output_A = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x2)
        # Basic attention mechanism formula to get intermediate output B
        scores = torch.bmm(key, query.transpose(1, 2)) / math.sqrt(d)
        output_B = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x1)

        # Make the summation of the two intermediate outputs
        output = output_A + output_B  # shape (1280, 32, 60)

        return output

@time_it
def test01():
    from thop import profile, clever_format
    it = 16384
    d_model = 32
    model = MutualCrossAttention(0.3)
    a = torch.randn(1,it,d_model)
    
    flops, params = profile(model, inputs=(a,a))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
    
def test02():
    B = 2
    mca1 = MutualCrossAttention(0.3)
    feat_p = torch.rand([10,32])
    feat_i2p = torch.rand([10,32])
    
    
    
    feat_p = torch.chunk(feat_p,B,dim = 0)
    feat_p = torch.stack([feat_p[0],feat_p[1]],dim=0)
    feat_i2p = torch.chunk(feat_i2p,B,dim = 0)
    feat_i2p = torch.stack([feat_i2p[0],feat_i2p[1]],dim=0)
    
    res = mca1(feat_p,feat_i2p)
    print(res[0].size())
    
    res = torch.cat([res[i] for i in range(res.size()[0])],dim=0)
    print(res.size())
    return res

if __name__ == "__main__":

    test02()
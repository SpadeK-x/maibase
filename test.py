import pickle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

a=torch.load('./charts/parsed_charts/100_remaster.pt')
print(a.shape)
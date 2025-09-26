import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np 


class imuDataset(Dataset):
    def __init__(self, data):
        self.data = np.array(data['data'])
        self.label = torch.tensor(data['label'], dtype=torch.int64) -1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = torch.tensor(self.data[index], dtype=torch.float32)
        y = self.label[index]
        return X, y
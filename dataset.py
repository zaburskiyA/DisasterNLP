import torch
from torch.utils.data import Dataset

class Data_Set(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.x = data
        self.y = labels
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)
    
    def __len__(self):
        return self.x.shape[0]

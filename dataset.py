import numpy as np
import random 
from torch.utils.data import Dataset

class BaseDataset(Dataset) :
    def __init__(self, idx_data, label_data, max_size) :
        super(BaseDataset , self).__init__()
        assert len(idx_data) == len(label_data)
        self.idx_data = [idx[-max_size:] for idx in idx_data]
        self.label_data = label_data
        
    def get_size(self) :
        return [len(idx) for idx in self.idx_data]

    def __len__(self) :
        return len(self.idx_data)

    def __getitem__(self , i) :
        return self.idx_data[i], self.label_data[i]


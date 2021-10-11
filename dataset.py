import numpy as np
import random
import collections
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from enum import IntEnum

class Token(IntEnum) :
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

class ElmoDataset(Dataset) :
    def __init__(self, idx_data, max_size) :
        super(ElmoDataset , self).__init__()
        self.idx_data = [idx_list[-max_size:] for idx_list in idx_data]
        self.max_size = max_size
        
    def __len__(self) :
        return len(self.idx_data)

    def __getitem__(self , idx) :
        return self.idx_data[idx]

class ElmoCollator:
    def __init__(self, len_data, batch_size, size_gap=5):
        self.len_data = len_data
        self.batch_size = batch_size
        self.size_gap = size_gap
        self.data_size = len(len_data)
        
    def sample(self) :
        batch_map = collections.defaultdict(list)
        idx_list = []
        batch_index = []
    
        # group index by size
        for idx in range(self.data_size) :
            len_idx = self.len_data[idx]
            len_group = len_idx // self.size_gap
            batch_map[len_group].append(idx)
            
        # sorting idx list based on size group
        batch_key = list(batch_map.keys())
        batch_key = sorted(batch_key, key=lambda x : x, reverse=True) 
        for key in batch_key :
            idx_list.extend(batch_map[key])
    
        # slicing index list by batch_size
        for i in range(0, self.data_size, self.batch_size) :
            batch_index.append(idx_list[i:i+self.batch_size])
    
        # shuffle batch index
        random.shuffle(batch_index)
        return batch_index
    
    def __call__(self, batch_samples):   
        batch_tensor = []
        for idx_list in batch_samples:
            batch_tensor.append(torch.tensor(idx_list + [Token.PAD]))
        idx_tensor = pad_sequence(batch_tensor, batch_first=True, padding_value=Token.PAD)
        return {'in' : idx_tensor[:,:-1], 'out' : idx_tensor[:,1:]}


class NSMCDataset(Dataset) :
    def __init__(self, idx_data, label_data, max_size) :
        super(NSMCDataset , self).__init__()
        self.idx_data = [idx_list[-max_size:] for idx_list in idx_data]
        self.label_data = label_data
        self.max_size = max_size
        
    def __len__(self) :
        return len(self.idx_data)

    def __getitem__(self , idx) :
        return self.idx_data[idx] , self.label_data[idx]


class NSMCCollator:
    def __init__(self, len_data, batch_size, size_gap=5):
        self.len_data = len_data
        self.batch_size = batch_size
        self.size_gap = size_gap
        self.data_size = len(len_data)
        
    def sample(self) :
        batch_map = collections.defaultdict(list)
        idx_list = []
        batch_index = []
    
        # group index by size
        for idx in range(self.data_size) :
            len_idx = self.len_data[idx]
            len_group = len_idx // self.size_gap
            batch_map[len_group].append(idx)
            
        # sorting idx list based on size group
        batch_key = list(batch_map.keys())
        batch_key = sorted(batch_key, key=lambda x : x, reverse=True) 
        for key in batch_key :
            idx_list.extend(batch_map[key])
    
        # slicing index list by batch_size
        for i in range(0, self.data_size, self.batch_size) :
            batch_index.append(idx_list[i:i+self.batch_size])
    
        # shuffle batch index
        random.shuffle(batch_index)
        return batch_index
    
    def __call__(self, batch_samples):   
        batch_tensor = []
        label_tensor = []
        for idx_list, label in batch_samples:
            batch_tensor.append(torch.tensor(idx_list + [Token.PAD]))
            label_tensor.append(label)

        idx_tensor = pad_sequence(batch_tensor, batch_first=True, padding_value=Token.PAD)
        label_tensor = torch.tensor(label_tensor)
        
        return idx_tensor, label_tensor

import torch
from torch.nn.utils.rnn import pad_sequence
import collections
import random

class Collator:
    def __init__(self, len_data, batch_size, size_gap=5):
        self.len_data = len_data
        self.batch_size = batch_size
        self.size_gap = size_gap
        
    def batch_sampler(self) :
        data_size = len(self.len_data)

        batch_map = collections.defaultdict(list)
        idx_list = []
        batch_index = []
    
        for idx in range(data_size) :
            idx_size = self.len_data[idx]
            size_group = (idx_size // self.size_gap)
            batch_map[size_group].append(idx)
            # batch_map
            # key : size group
            # value : idx list
            
        batch_key = list(batch_map.keys())
        batch_key = sorted(batch_key, reverse=True) 
        # sorting idx list based on size group
        for key in batch_key :
            idx_list.extend(batch_map[key])
    
        # slicing batch_size
        for i in range(0, data_size, self.batch_size) :
            batch_index.append(idx_list[i:i+self.batch_size])
    
        random.shuffle(batch_index)
    
        return batch_index
        
    def __call__(self, batch_samples):
        idx_data = []
        label_data = []
        for idx, label in batch_samples:
            idx_data.append(torch.tensor(idx))
            label_data.append(label)
            
        # pad sequence of tensor
        idx_data = pad_sequence(idx_data, batch_first=True, padding_value=0)
        label_data = torch.tensor(label_data, dtype=torch.float32)

        return idx_data, label_data
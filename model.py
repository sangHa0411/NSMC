import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockLSTM(nn.Module) :
    def __init__(self, in_size, h_size) :
        super(BlockLSTM, self).__init__()
        self.in_size = in_size
        self.h_size = h_size
        
        # in_gate, forget_gate, gate_gate, output gate
        self.wx_list = nn.ModuleList([nn.Linear(in_size,h_size),
                                      nn.Linear(in_size,h_size),
                                      nn.Linear(in_size,h_size),
                                      nn.Linear(in_size,h_size)])
        
        self.wh_list = nn.ModuleList([nn.Linear(h_size,h_size),
                                      nn.Linear(h_size,h_size),
                                      nn.Linear(h_size,h_size),
                                      nn.Linear(h_size,h_size)])
        
    def forward_seq(self, in_tensor, h_tensor, c_tensor) :
        i_tensor = torch.sigmoid(self.wx_list[0](in_tensor) + self.wh_list[0](h_tensor))
        f_tensor = torch.sigmoid(self.wx_list[1](in_tensor) + self.wh_list[1](h_tensor))
        g_tensor = torch.tanh(self.wx_list[2](in_tensor) + self.wh_list[2](h_tensor))
        o_tensor = torch.sigmoid(self.wx_list[3](in_tensor) + self.wh_list[3](h_tensor))
    
        c_tensor = torch.mul(f_tensor,c_tensor) + torch.mul(i_tensor,g_tensor)
        h_tensor = torch.mul(o_tensor,torch.tanh(c_tensor))
    
        return o_tensor, c_tensor, h_tensor
                
    def forward(self, in_tensor, h_tensor, c_tensor) :
        
        # in_tensor : (batch_size, seq_size, feature_size)
        batch_size, seq_size, feature_size = in_tensor.shape
        tensor_list = torch.chunk(in_tensor, seq_size , dim=1)
        
        output_list = []
        for i in range(seq_size) :
            tensor_idx = tensor_list[i]
            # tensor_idx : (batch_size, 1, feature_size)
            # h_tensor : (batch_size, 1, feature_size)
            # c_tensor : (batch_size, 1, feature_size)
            o_tensor, h_tensor, c_tensor = self.forward_seq(tensor_idx, h_tensor, c_tensor)
            output_list.append(o_tensor)

        output_tensor = torch.cat(output_list, dim=1)
        
        return output_tensor, h_tensor, c_tensor


class StackedLSTM(nn.Module) :
    def __init__(self, layer_size, h_size, v_size, use_cuda) :
        super(StackedLSTM,self).__init__()
        self.layer_size = layer_size
        self.h_size = h_size
        self.v_size = v_size
        self.use_cuda = use_cuda
        
        # embedding layer
        self.em = nn.Embedding(num_embeddings=v_size, 
                               embedding_dim=h_size, 
                               padding_idx=0)
        # lstm layer
        self.lstm_layer = nn.ModuleList()
        # output layer
        self.o_layer = nn.Linear(h_size, 1)
        
        for i in range(self.layer_size) :
            self.lstm_layer.append(BlockLSTM(h_size,h_size))
        
        self.init_param()
        
    def init_param(self) :
        nn.init.normal_(self.em.weight, 0.0, 0.1)
        for m in self.modules() :
            if isinstance(m,nn.Linear) :
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def set_embedding(self, weight) :
        v_size, h_size = weight.shape
        assert v_size == self.v_size
        assert h_size == self.h_size
        
        em_weight = torch.tensor(weight)
        self.em = nn.Embedding.from_pretrained(em_weight,
                                               freeze=True,
                                               padding_idx=0)
        
    def forward(self, in_tensor) :
        batch_size, seq_size = in_tensor.shape
 
        em_tensor = self.em(in_tensor)
        tensor_ptr = em_tensor
        
        h_tensor = torch.zeros(batch_size, 1, self.h_size)
        h_tensor = h_tensor.cuda() if self.use_cuda else h_tensor
        c_tensor = torch.zeros(batch_size, 1, self.h_size)
        c_tensor = c_tensor.cuda() if self.use_cuda else c_tensor
        
        for i in range(self.layer_size) :
            tensor_ptr, h_tensor, c_tensor = self.lstm_layer[i](tensor_ptr, h_tensor, c_tensor)
        
        idx_tensor = torch.mean(tensor_ptr, dim=1)

        o_tensor = self.o_layer(idx_tensor)
        o_tensor = torch.sigmoid(o_tensor).squeeze(1)
        
        return o_tensor

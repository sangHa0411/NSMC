from dataset import Token
import torch
import torch.nn as nn
import torch.nn.functional as F

class ElmoModel(nn.Module) :
    def __init__(self, 
        layer_size, 
        v_size, 
        em_size, 
        h_size, 
        cuda_flag=True,
        backward_flag=False,
        drop_rate=0.1) :

        super(ElmoModel, self).__init__()
        self.layer_size = layer_size
        self.v_size = v_size
        self.em_size = em_size
        self.h_size = h_size
        self.drop_rate = drop_rate
        self.backward_flag = backward_flag
        self.cuda_flag = cuda_flag

        self.em = nn.Embedding(num_embeddings=v_size,
                               embedding_dim=em_size,
                               padding_idx = Token.PAD)
        
        self.lstm = nn.LSTM(input_size = em_size,
                            hidden_size = h_size,
                            num_layers=layer_size,
                            dropout=drop_rate,
                            bidirectional=False,
                            batch_first=True)
        self.o_layer = nn.Linear(h_size, v_size)

        self.init_param()

    def init_param(self) :
        for p in self.parameters() :
            if p.dim() > 1 :
                nn.init.xavier_uniform_(p)

    def get_lstm_tensor(self, in_tensor) :
        batch_size = in_tensor.shape[0]
        if self.backward_flag == True :
            in_tensor = torch.flip(in_tensor, [1])

        em_tensor = self.em(in_tensor)
        h_tensor = torch.zeros(1*self.layer_size, batch_size, self.h_size)
        c_tensor = torch.zeros(1*self.layer_size, batch_size, self.h_size)

        if self.cuda_flag == True :
            h_tensor = h_tensor.cuda()
            c_tensor = c_tensor.cuda()

        lstm_tensor, (h_tensor, c_tensor) = self.lstm(em_tensor, (h_tensor, c_tensor))
        return lstm_tensor

    def get_hidden_state(self, in_tensor) :
        lstm_tensor = self.get_lstm_tensor(in_tensor)
        if self.bachward_flag == True :
            lstm_tensor = torch.flip(lstm_tensor, [1])
        return lstm_tensor

    def forward(self, in_tensor) :
        lstm_tensor = self.get_lstm_tensor(in_tensor)
        o_tensor = self.o_layer(lstm_tensor)
        return o_tensor

from dataset import Token
import torch
import torch.nn as nn
import torch.nn.functional as F

class ElmoModel(nn.Module) :
    def __init__(self, layer_size, vocab_size, embedding_size, hidden_size, cuda_flag=True, drop_rate=0.1) :
        super(ElmoModel, self).__init__()
        self.layer_size = layer_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate
        self.cuda_flag = cuda_flag

        self.em = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx = Token.PAD)
        self.lstm = nn.LSTM(input_size = embedding_size,
            hidden_size = hidden_size,
            num_layers=layer_size,
            dropout=drop_rate,
            bidirectional=False,
            batch_first=True
        )
        self.o_layer = nn.Linear(hidden_size, vocab_size)

        self.init_param()

    def init_param(self) :
        for p in self.parameters() :
            if p.dim() > 1 :
                nn.init.xavier_uniform_(p)

    def forward(self, in_tensor) :
        batch_size= in_tensor.shape[0]
        em_tensor = self.em(in_tensor)

        h_tensor = torch.zeros(1*self.layer_size, batch_size, self.hidden_size)
        c_tensor = torch.zeros(1*self.layer_size, batch_size, self.hidden_size)
        if self.cuda_flag == True :
            h_tensor = h_tensor.cuda()
            c_tensor = c_tensor.cuda()

        feature_tensor, (h_tensor, c_tensor) = self.lstm(em_tensor, (h_tensor, c_tensor))
        o_tensor = self.o_layer(feature_tensor)
        return o_tensor

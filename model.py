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

    def get_feature(self, in_tensor) :
        batch_size= in_tensor.shape[0]
        em_tensor = self.em(in_tensor)

        h_tensor = torch.zeros(1*self.layer_size, batch_size, self.hidden_size)
        c_tensor = torch.zeros(1*self.layer_size, batch_size, self.hidden_size)
        if self.cuda_flag == True :
            h_tensor = h_tensor.cuda()
            c_tensor = c_tensor.cuda()

        feature_tensor, (h_tensor, c_tensor) = self.lstm(em_tensor, (h_tensor, c_tensor))
        return feature_tensor

    def forward(self, in_tensor) :
        feature_tensor = self.get_feature(in_tensor)
        o_tensor = self.o_layer(feature_tensor)
        return o_tensor

class NsmcClassification(nn.Module) :
    def __init__(self, forward_model : ElmoModel, backward_model : ElmoModel, class_size : int) :
        super(NsmcClassification, self).__init__()
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.hidden_size = forward_model.hidden_size * 2
        self.class_size = class_size
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.o_layer = nn.Linear(self.hidden_size, class_size)

    def forward(self, in_tensor) :
        reverse_tensor = torch.flip(in_tensor, (1,))

        forward_feature = self.forward_model.get_feature(in_tensor)
        backward_feature = self.backward_model.get_feature(reverse_tensor)
        backward_feature = torch.flip(backward_feature, (1,))

        feature_tensor = torch.cat((forward_feature, backward_feature), 2)
        feature_tensor = self.layer_norm(feature_tensor)
        feature_tensor = torch.mean(feature_tensor, dim=1)

        o_tensor = self.o_layer(feature_tensor).squeeze(1)
        o_tensor = F.sigmoid(o_tensor)
        return o_tensor

import torch 
import torch.nn as nn
import torch.nn.functional as F

import math
import json

class StayEmbedding(nn.Module):

    def __init__(self, d_model, max_len=150, num_tokens=None, device='cpu'):
        super().__init__()

        #compute time encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.squeeze(pe)
        self.time_encoding = pe.to(device)

        #compute code embedding
        self.embedding = nn.Embedding(1 + num_tokens, d_model, padding_idx=0, device=device)

    def forward(self, codes):
        batch_size, n_timesteps = codes.shape
        
        code_embedding = self.embedding(codes)

        time_embedding = self.time_encoding.repeat([batch_size, 1, 1])
        time_embedding = time_embedding[:, :n_timesteps, :]

        x = code_embedding + time_embedding
        return x
    
class BERT_MLM(nn.Module):

    def __init__(self, d_embedding, d_model, tokenizer_path, dropout=0.1, n_layers=2, nhead=4, device='cpu'):
        super().__init__()

        tokenizer = json.load(open(tokenizer_path))

        #embedding
        self.embedding = StayEmbedding(d_embedding, num_tokens=len(tokenizer), device=device)

        #projection
        self.proj = nn.Linear(d_embedding, d_model).to(device)

        #transformer embedding
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model, dropout=dropout, batch_first=True, device=device)
        self.bert = nn.TransformerEncoder(layer, n_layers).to(device)

        #classification
        self.cls = nn.Linear(d_model, len(tokenizer), bias=True).to(device)

    def forward(self, tokens):
        x = self.embedding(tokens)

        x = nn.ReLU()(self.proj(x))

        x = self.bert(x)
        x = nn.ReLU()(x)

        x = F.log_softmax(self.cls(x), dim=-1)
        return x
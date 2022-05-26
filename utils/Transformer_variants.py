import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# cosine sine encoding
class TFEQ(nn.Module):
    def __init__(self, channel=1, dmodel=64, nhead=8, dim_feedforward=32, num_layers=1, time_in=200):
        super(TFEQ, self).__init__()
        self.input_embedding = nn.Linear(channel, dmodel)
        self.pos_encoder = PositionalEncoding(dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            dmodel, nhead, dim_feedforward)
        self.temporal_encoder = nn.TransformerEncoder(
            self.temporal_encoder_layer, num_layers)
        self.decoder = nn.Linear(dmodel, 1)
        self.fla = nn.Flatten()
        self.fc = nn.Linear(time_in, 2)

    def forward(self, x):
        b, t,  c = x.shape  # [B, 200, 3]
        x = self.input_embedding(x)  # [B, 200,  64]
        x = self.pos_encoder(x)
        x = self.temporal_encoder(x)
        x = self.decoder(x)  # [B, 1, 200, 1]
        x = self.fla(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# lstm encoding
class TFEQ(nn.Module):
    def __init__(self, channel=1, dmodel=64, nhead=8, dim_feedforward=32, num_layers=1, time_in=200):
        super(TFEQ, self).__init__()
        self.dmodel = dmodel
        self.input_embedding = nn.Linear(channel, dmodel)
        self.pos_lstm = nn.LSTM(input_size=time_in, hidden_size=time_in)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            dmodel, nhead, dim_feedforward)
        self.temporal_encoder = nn.TransformerEncoder(
            self.temporal_encoder_layer, num_layers)
        self.decoder = nn.Linear(dmodel, 1)
        self.fla = nn.Flatten()
        self.fc = nn.Linear(time_in, 2)

    def forward(self, x):
        b, t,  c = x.shape  # [B, 200, 3]
        x = self.input_embedding(x)  # [B, 200,  64]
        x = x.view(-1, self.dmodel, t)
        t_pe, _ = self.pos_lstm(x)
        t_pe = t_pe.view(-1, t, self.dmodel)
        x = self.temporal_encoder(t_pe)
        x = self.decoder(x)  # [B, 1, 200, 1]
        x = self.fla(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

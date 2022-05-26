import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TFEQ(nn.Module):
    def __init__(self, channel=1, dmodel=64, nhead=8, dim_feedforward=32, num_layers=1, time_in=200):
        super(TFEQ, self).__init__()
        self.input_embedding = nn.Linear(channel, dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            dmodel, nhead, dim_feedforward)
        self.temporal_encoder = nn.TransformerEncoder(
            self.temporal_encoder_layer, num_layers)
        self.decoder = nn.Linear(dmodel, 1)
        self.temporal_pos = torch.arange(0, time_in).cuda()
        self.temporal_pe = nn.Embedding(time_in, dmodel)
        self.fla = nn.Flatten()
        self.dp = nn.Dropout(p=0.5)
        self.fc = nn.Linear(time_in, 2)

    def forward(self, x):
        b, t,  c = x.shape  # [B, 200, 3]
        x = self.input_embedding(x)  # [B, 200,  64]
        t_pe = self.temporal_pe(self.temporal_pos).expand(b, t, -1)
        x = x + t_pe
        x = self.temporal_encoder(x)
        x = self.decoder(x)  # [B, 1, 200, 1]
#         print(x.shape)
        x = self.fla(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class TFEQ_Conv(nn.Module):
    def __init__(self, channel=1, dmodel=64, nhead=8, dim_feedforward=32, num_layers=1, time_in=200):
        super(TFEQ_Conv, self).__init__()
        self.input_embedding = nn.Linear(channel, dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(dmodel, nhead, dim_feedforward)
        self.temporal_encoder = nn.TransformerEncoder( self.temporal_encoder_layer, num_layers)
        self.decoder = nn.Conv1d(in_channels=dmodel, out_channels=channel, kernel_size=1, stride=1)
        self.temporal_pos = torch.arange(0, time_in).cuda()
        self.temporal_pe = nn.Embedding(time_in, dmodel)
        self.fla = nn.Flatten()
        self.dp = nn.Dropout(p=0.5)
        self.fc = nn.Linear(time_in*channel, 2)

    def forward(self, x):
        b, t,  c = x.shape  # [B, 200, 3]
        x = self.input_embedding(x)  # [B, 200,  64]
        t_pe = self.temporal_pe(self.temporal_pos).expand(b, t, -1)
        x = x + t_pe
        x = self.temporal_encoder(x)
        x = x.view(-1, x.shape[2], x.shape[1])
        x = self.decoder(x)  # [B, 1, 200, 1]
        x = self.fla(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class TFEQ_RNN(nn.Module):
    def __init__(self, channel=1, dmodel=64, nhead=8, dim_feedforward=32, num_layers=1, time_in=200):
        super(TFEQ_RNN, self).__init__()
        self.input_embedding = nn.Linear(channel, dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(dmodel, nhead, dim_feedforward)
        self.temporal_encoder = nn.TransformerEncoder(self.temporal_encoder_layer, num_layers)
        self.temporal_pos = torch.arange(0, time_in).cuda()
        self.temporal_pe = nn.Embedding(time_in, dmodel)
        self.ln=nn.LayerNorm((time_in, dmodel))
        self.Feed1 = nn.Conv1d(in_channels=dmodel, out_channels=dmodel//2, kernel_size=1, stride=1)
        self.Feed2 = nn.Conv1d(in_channels=dmodel//2, out_channels=channel, kernel_size=1, stride=1)
        self.fla = nn.Flatten()
        self.dp = nn.Dropout(p=0.5)
        
        self.rnn = nn.RNN(input_size=time_in*channel, hidden_size=100)
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        b, t,  c = x.shape  # [B, 200, 3]
        x = self.input_embedding(x)  # [B, 200,  64]
        t_pe = self.temporal_pe(self.temporal_pos).expand(b, t, -1)
        x = x + t_pe
        x = self.temporal_encoder(x)
        x= self.ln(x)
        x = x.view(-1, x.shape[2], x.shape[1])
        x = F.relu(self.Feed1(x))
        x = F.relu(self.Feed2(x))
        x = self.fla(x)
        x = x.view(-1, 1, x.shape[-1])
        x, _ = self.rnn(x)
        
        x = self.fla(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

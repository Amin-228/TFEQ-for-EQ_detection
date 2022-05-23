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


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim // 8,
                                    kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,
                                  out_channels=in_dim // 8,
                                  kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim,
                                    kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,
                                             -1, width * height).permute(
                                                 0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1,
                                         width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1,
                                             width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Att_CNN(nn.Module):
    def __init__(self):
        super(Att_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(7, 32, (1, 3), stride=1, padding=(0, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 3), stride=1, padding=(0, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 3), stride=1, padding=(0, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)))
        self.attention_layer1 = SelfAttention(32)
        self.attention_layer2 = SelfAttention(64)
        self.attention_layer3 = SelfAttention(128)
        self.fc1 = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(256, 2))
        self.fla = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x, p = self.attention_layer1(x)
        x = self.conv2(x)
        x, p = self.attention_layer2(x)
        x = self.conv3(x)
        x, p = self.attention_layer3(x)
        x = self.fla(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(7, 32, (1, 3), stride=1, padding=(0, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 3), stride=1, padding=(0, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 3), stride=1, padding=(0, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)))

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2))
        self.fla = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fla(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


class Conv_com(nn.Module):
    def __init__(self, ):
        super(Conv_com, self).__init__()
        self.conv_1 = nn.Conv1d(
            in_channels=1,  out_channels=32, kernel_size=3, stride=2)
        self.conv_2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.MaxPool = nn.MaxPool1d(2)
        self.Flatten = torch.nn.Flatten()
        self.fc1 = nn.Linear(100, 2)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.relu(self.conv_2(out))
        out = F.relu(self.conv_3(out))
        out = self.MaxPool(out)
        out = self.Flatten(out)
        return out


class CRNN(nn.Module):
    def __init__(self, ):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=3,  out_channels=64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(
            in_channels=64,  out_channels=64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fla = nn.Flatten()
        self.dp = nn.Dropout(p=0.5)
        self.rnn = nn.RNN(input_size=3072, hidden_size=100)
        self.fc1 = nn.Linear(100, 2)

    def forward(self, data):
        out = F.relu(self.conv1(data))
        out = F.relu(self.conv2(out))
        out = self.dp(out)
        out = self.pool1(out)
        out = self.fla(out)
        out = out.view(-1, 1, out.shape[-1])
        out, _ = self.rnn(out)
        out = self.fla(out)
        out = self.fc1(out)
        return F.log_softmax(out, dim=1)


class M_CRNN(nn.Module):
    def __init__(self, ):
        super(M_CRNN, self).__init__()
        self.conv_x = Conv_com()
        self.conv_y = Conv_com()
        self.conv_z = Conv_com()
        self.rnn = nn.RNN(input_size=2208, hidden_size=100)
        self.fla = nn.Flatten()
        self.dp = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, data):
        x, y, z = data[:, 0, :], data[:, 1, :], data[:, 2, :]
        x = x.view(-1, 1, x.shape[-1])
        y = y.view(-1, 1, y.shape[-1])
        z = z.view(-1, 1, z.shape[-1])
        x_out = self.conv_x(x)
        y_out = self.conv_y(y)
        z_out = self.conv_z(z)
        new_feature = torch.cat([x_out, y_out, z_out], dim=1)
        new_feature = new_feature.view(-1, 1, new_feature.shape[-1])
        out, _ = self.rnn(new_feature)
        out = self.dp(out)
        out = self.fla(out)
#         out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

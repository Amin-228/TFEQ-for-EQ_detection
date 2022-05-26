import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class Conv(nn.Module):
    def __init__(self, ):
        super(Conv, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=1,  out_channels=32, kernel_size=3, stride=2)
        self.conv_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.MaxPool = nn.MaxPool1d(2)
        self.Flatten = torch.nn.Flatten()

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.relu(self.conv_2(out))
        out = F.relu(self.conv_3(out))
        out = self.MaxPool(out)
        out = self.Flatten(out)
        return out


class MCRNN(nn.Module):
    def __init__(self, ):
        super(MCRNN, self).__init__()
        self.conv_x = Conv()
        self.conv_y = Conv()
        self.conv_z = Conv()
        self.rnn = nn.RNN(input_size=2208, hidden_size=64)
        self.fla = nn.Flatten()
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
        out = self.fla(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

class Conv_com(nn.Module):
    def __init__(self, ):
        super(Conv_com, self).__init__()
        self.conv_1 = nn.Conv1d(
            in_channels=1,  out_channels=32, kernel_size=3, stride=2)
        self.conv_2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.MaxPool = nn.MaxPool1d(2)
        self.rnn = nn.RNN(input_size=736, hidden_size=100)
        self.fla = torch.nn.Flatten()
        self.fc1 = nn.Linear(100, 2)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.relu(self.conv_2(out))
        out = F.relu(self.conv_3(out))
        out = self.MaxPool(out)
        out = self.fla(out)
        out = out.view(-1, 1, out.shape[-1])
        out, _ = self.rnn(out)
        out = self.fla(out)
        return out    

class M_CRNN(nn.Module):
    def __init__(self, ):
        super(M_CRNN, self).__init__()
        self.conv_x = Conv_com()
        self.conv_y = Conv_com()
        self.conv_z = Conv_com()
        self.rnn = nn.RNN(input_size=2208, hidden_size=100)
        self.fla = nn.Flatten()
        self.dp = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(300, 100)
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
#         new_feature = new_feature.view(-1, 1, new_feature.shape[-1])
#         out, _ = self.rnn(new_feature)
        out = F.relu(self.fc1(new_feature))
        out = self.dp(out)
#         out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)
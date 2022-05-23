import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from joblib import dump, load
import librosa.display
from utils import data_process, visualize
from utils.utils import train, test, summary, setup_seed
from utils.models import Att_CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

X = np.load('/home/yuanshao/EQ_Place/code/data/feature_8_29.npy')
Y = np.load('/home/yuanshao/EQ_Place/code/data/Y_8_29.npy')
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=1)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()
traindata = TensorDataset(x_train, y_train)
testdata = TensorDataset(x_test, y_test)
train_loader = DataLoader(traindata, batch_size=256, shuffle=True)
test_loader = DataLoader(testdata, batch_size=256, shuffle=True)

# 设置随机数种子
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Att_CNN().cuda()
import time
L = []
test_hist = []
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 100 + 1):
    start = time.time()
    loss = train(model, train_loader, optimizer, epoch)
    L.append(loss)
    print("time {:.1f} sec:".format(time.time() - start))
    acc = test(model, test_loader)
    test_hist.append(acc)

x_test = x_test.to(device)
y_test = y_test.to(device)

pred = model(x_test)
pred = pred.max(1, keepdim=True)[1]
correct = pred.eq(y_test.view_as(pred)).sum().item()
pred = pred.cpu().numpy().squeeze()

summary(pred, y_test.cpu())

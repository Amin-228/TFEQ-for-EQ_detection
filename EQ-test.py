import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os, glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from joblib import dump, load
import librosa.display

from utils import data_process, visualize

file = '/home/yuanshao/EQ_Place/dataset/EQ_90s_02and03_15/EQ_90s_02_15/01231315857.csv'

# 01231315776.csv

SamplingRate = 100  # need to be changed, 25/50/100
Duration = 4  # need to be changed, 2/4/10
WindowSize = 2 * SamplingRate
original_SamplingRate = 100
rate = original_SamplingRate / SamplingRate
X_EQ = []
Y_EQ = []
Z_EQ = []

data = pd.read_csv(file)
data = data.iloc[0::int(rate)]
data = data.reset_index(drop=True)
X = data['x']
X_peak = np.where(X == np.max(X))[0][0]
start = X_peak - int(SamplingRate)
end = X_peak + int(SamplingRate) * (Duration - 1)
df = data[start:end]
df = df.reset_index(drop=True)
X_tg = df['x']
Y_tg = df['y']
Z_tg = df['z']

if len(X_tg) == SamplingRate * Duration:
    ## 2 sec sliding window with 1 sec overlap
    for j in np.arange(0, len(X_tg) - SamplingRate, SamplingRate):
        X_batch = X_tg[j:j + WindowSize]
        Y_batch = Y_tg[j:j + WindowSize]
        Z_batch = Z_tg[j:j + WindowSize]
        if len(X_batch) == WindowSize:
            X_EQ.append(X_batch.values)
            Y_EQ.append(Y_batch.values)
            Z_EQ.append(Z_batch.values)
X_EQ = np.asarray(X_EQ)
Y_EQ = np.asarray(Y_EQ)
Z_EQ = np.asarray(Z_EQ)

X_EQ = X_EQ.reshape(int(len(X_EQ) / (Duration - 1)), Duration - 1, WindowSize,
                    1)
Y_EQ = Y_EQ.reshape(int(len(Y_EQ) / (Duration - 1)), Duration - 1, WindowSize,
                    1)
Z_EQ = Z_EQ.reshape(int(len(Z_EQ) / (Duration - 1)), Duration - 1, WindowSize,
                    1)

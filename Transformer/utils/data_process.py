import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import librosa.display  


def IQR(data):
    # 按四分位距计算IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    return Q3 - Q1

def cal_IQR(data, frame_size=200, overlap=0):
    # 计算IQR
    # frame_size: 窗口长度
    # overlaps:  重合长度  
    step = frame_size - overlap
    frameNum = np.floor(len(data) / step).astype(int)
    iqr = np.zeros((frameNum, 1))
    for i in range(frameNum - 1):
        curFrame = data[(i * step): ((i + 1) * step)]
        iqr[i] = IQR(curFrame)
    return iqr.squeeze()


def data_resample(array, npts):
    # 数据重采样
    # array: 采样数据
    # npts: 采样后的长度
    interpolated = interp1d(np.arange(len(array)),
                            array,
                            axis=0,
                            fill_value='extrapolate')
    return interpolated(np.linspace(0, len(array), npts))

def downsample(array, npts):
    index=np.linspace(0, len(array), npts).astype(int)
    return array[index]



def ZCR(data, frame_length=200, hop_length=200):
    # 过零率
    zcr= librosa.feature.zero_crossing_rate(data, 
                                       frame_length=200, 
                                       center=False,
                                       hop_length=200)
    return zcr[0]
    
def CAV(data, frame_size=200, overlap=0):
    # 计算CAV
    # frame_size: 窗口长度
    # overlaps:  重合长度
    step = frame_size - overlap
    frameNum = np.floor(len(data[0]) / step).astype(int)
    cav = np.zeros((frameNum, 1))
    for i in range(frameNum - 1):
        x, y, z = data[0][(i * step):((i + 1) * step)], data[1][(
            i * step):((i + 1) * step)], data[2][(i * step):((i + 1) * step)]
        curFrame = np.sqrt(x**2 + y**2 + z**2)
        cav[i] = sum(curFrame)
    return cav.squeeze()
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa.display  
from scipy import signal




def draw_picture(data, show=False, save_path=None, save=False, fs=100, lim=0.01, yline=0.0025):
    # 波形可视化函数，用于画出每个通道的波形和stft变化图
    # data 传入的数据
    # show 是否在运行中显示
    # save_path 保存的路径
    # save 是否保存
    # fs 采样率
    # lim 波形y轴范围
    # yline 波形振荡的指示线，便于观察，无实际意义
    x_lim=np.arange(0,len(data[0]),1)
    fig, axs = plt.subplots(3, 2, sharex=False,figsize=(12, 7))
    for i in range(3):

        axs[i,0].axhline(y=yline, c="green",  linewidth=0.8)
        axs[i,0].axhline(y=-yline, c="green",  linewidth=0.8)
        axs[i,0].plot(x_lim, data[i], linewidth=0.5)
        axs[i,0].set_ylim((-lim, lim))
        axs[i,0].grid(True)


        f, t, Zxx = signal.stft(data[i], fs,nperseg=512)
        Zxx = librosa.amplitude_to_db(abs(Zxx))
        tt = axs[i,1].pcolormesh(t, f, Zxx, shading='auto')
        fig.colorbar(tt, ax=axs[i,1], )

    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=100)
    if not show:
        plt.clf()
        plt.close()

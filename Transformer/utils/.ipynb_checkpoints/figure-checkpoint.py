from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa.display  
from scipy import signal




def draw_picture(data, show=False, save_path=None, save=False, fs=100, lim=0.01, yline=0.0025):
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy import signal
from scipy.stats import pearsonr
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("to csr program")
    parser.add_argument("--file", type=str)
    parser.add_argument("--flag", type=int)
    args = parser.parse_args()
    file = args.file
    flag = args.flag


    lim=0.01
    yline=0.0025
    fs=100

    path='/home/yuanshao/EQ_Place/dataset/EQ_20210828_070322/'
    filepath = path +file
    df=pd.read_csv(filepath)
    df=np.array(df)[:,:3]
    ex,ey,ez=df[:,0],df[:,1],df[:,2]
    ex,ey,ez=ex-np.mean(ex), ey-np.mean(ey), ez-np.mean(ez)
    data=[ex,ey,ez]
    filteData=[]
    sos=signal.butter(N=1, Wn=35, btype='lowpass',fs=100,output='sos')
    filteData.append(signal.sosfilt(sos, ex))
    filteData.append(signal.sosfilt(sos, ey))
    filteData.append(signal.sosfilt(sos, ez))
    pccs=list(map(lambda x, y :pearsonr(x, y), data, filteData))
    legend=['origin', 'filter']
    x_lim=np.arange(0,len(ex),1)
    lim=0.01
    fig, axs = plt.subplots(3, 2, sharex=False, sharey=True, figsize=(12, 7))
    for i in range(3):
        axs[i,0].plot(x_lim, data[i], linewidth=0.5)
        axs[i,0].plot(x_lim, filteData[i])
        axs[i,0].legend(legend, loc='upper left')
        axs[i,0].axhline(y=yline, c="green",  linewidth=1)
        axs[i,0].axhline(y=-yline, c="green",  linewidth=1)
        axs[i,0].text(55000,-0.008,  s='CC = '+str(pccs[i])[1:7],fontsize=12)
        axs[i,0].set_ylim((-lim, lim))
        
        axs[i,1].plot(x_lim[100:200], data[i][100:200], linewidth=0.5)
        axs[i,1].plot(x_lim[100:200], filteData[i][100:200])
        axs[i,1].legend(legend, loc='upper left')
        axs[i,1].set_ylim((-lim, lim))
        axs[i,0].grid(True)
    plt.tight_layout()
    filename='/home/yuanshao/EQ_Place/figures/filter/'+file[:-4]
    plt.savefig(filename, dpi=100)
    plt.clf()
    plt.close()



    # fig, axs = plt.subplots(3, 2, sharex=False,figsize=(12, 7))
    # draw_data=[ex,ey,ez]
    # for i in range(3):

    #     axs[i,0].axhline(y=yline, c="red",  linewidth=1)
    #     axs[i,0].axhline(y=-yline, c="red",  linewidth=1)
    #     axs[i,0].plot(x_lim, draw_data[i], linewidth=0.5)
    #     axs[i,0].set_ylim((-lim, lim))
    #     axs[i,0].grid(True)


    #     f, t, Zxx = signal.stft(draw_data[i], fs,nperseg=512)
    #     Zxx = librosa.amplitude_to_db(abs(Zxx))
    #     tt = axs[i,1].pcolormesh(t, f, Zxx, shading='auto')
    #     fig.colorbar(tt, ax=axs[i,1], )
    # plt.tight_layout()
    # if flag:
    #     filename='C:/ysz/code/class_8_29/p/'+file[:-4]
    # else:
    #     filename='C:/ysz/code/class_8_29/n/'+file[:-4]
    # plt.savefig(filename, dpi=100)
    # plt.clf()
    # plt.close()
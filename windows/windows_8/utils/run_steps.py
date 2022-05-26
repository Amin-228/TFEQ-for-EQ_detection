import os
import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run(file,flag):
    # file = file[0]
    pars = "--file {:s}".format(file)
    flag = "--flag {:d}".format(flag)

    py = 'python'
    print("{:s} data_plot.py {:s} {:s}".format(py, pars, flag))
    os.system("{:s} data_plot.py {:s} {:s}".format(py, pars, flag))



if __name__ == "__main__":

    # labels=np.load('C:/ysz/code/label/init_label/labels_8_29.npy')
    # if not os.path.exists('C:/ysz/code/class_8_29/n/'):
    #     os.mkdir('C:/ysz/code/class_8_29/n/')
    # if not os.path.exists('C:/ysz/code/class_8_29/p/'):
    #     os.mkdir('C:/ysz/code/class_8_29/p/')

    path='/home/yuanshao/EQ_Place/dataset/EQ_20210828_070322/'
    files=os.listdir(path)
    files.sort()

    process_list = []
    limit = 16

    # %matplotlib inline
    for i in range(len(files)):
        f = files[i]
        flag = 1
        p = mp.Process(target=run, args=(f, flag,))
        p.start()
        process_list.append(p)

        if (i + 1) % limit == 0:
            for p in process_list:
                p.join()

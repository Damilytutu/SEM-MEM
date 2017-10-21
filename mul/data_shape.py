# coding: utf-8
#__author__  = "Damily"
#__email__ = "juanhuitu@pku.edu.cn"

import os
import numpy as np
import scipy.io as scio


def load_data_shape():
    train_data = np.empty((37646, 100, 150))
    test_data = np.empty((18932, 100, 150))
    train_label = np.empty((37646, ), dtype='uint8')
    test_label = np.empty((18932, ), dtype='uint8')
    label = np.empty((56578, ), dtype='uint8')
    files = os.listdir('/home/tujh/NTU_lstm/NTU_skeleton_main_lstm')
    num = len(files)


    train_num = 0
    test_num = 0

    for i in range(num):
        one_data = scio.loadmat('/home/tujh/NTU_lstm/NTU_skeleton_main_lstm/' + files[i])
        one_data = one_data['skeleton']

        # get the label
        temp = files[i][17:20]
        if list(temp)[0] == 0 and list(temp)[1] == 0:
            label[i] = int(list(temp)[2]) - 1
        else:
            label[i] = int(list(temp)[1] + list(temp)[2]) - 1


        temp1 = files[i][5:8]
        view_num = int(list(temp1)[0] + list(temp1)[1] + list(temp1)[2])


        if view_num == 2 or view_num == 3:
            train_data[train_num,:,:] = one_data
            train_label[train_num] = label[i]
            if train_num < 37646:
                train_num = train_num + 1
                print('train_num = ', train_num)
        else:
            test_data[test_num,:,:] = one_data
            test_label[test_num] = label[i]
            if test_num < 18932:
                test_num = test_num + 1
                print('test_num = ', test_num)

    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    train_data, train_label, test_data, test_label  = load_data_shape()
    print(train_data.shape)
    print(test_data.shape)
    print(test_label.shape)
    print(max(train_label))
    print(max(test_label))

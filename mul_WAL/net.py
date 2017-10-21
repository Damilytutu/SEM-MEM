# coding: utf-8
#__author__  = "Damily"
#__email__ = "juanhuitu@pku.edu.cn"

from keras.layers.core import*
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Activation, multiply, Input, merge
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from data_shape import load_data_shape
from data_motion import load_data_motion
from keras.optimizers import adam
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from net import lstmNet, Create_Net
from kutilities.layers import Attention
import scipy.io as sio
import matplotlib.pyplot as plt



nb_classes = 60

def lstmNet(input_shape1, input_shape2):

    # temporal_network
    input_t = Input(shape=input_shape1)
    lstm1_t = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_t)
    dp1_t = Dropout(0.5)(lstm1_t)
    lstm2_t = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(dp1_t)
    dp2_t = Dropout(0.5)(lstm2_t)
    lstm3_t = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(dp2_t)
    dp3_t = Dropout(0.5)(lstm3_t)
    att1 = Attention()(dp3_t)
    out1_t = Dense(nb_classes, activation='softmax')(att1)
    model1 = Model(input_t, out1_t)

    #model1.summary()

    # load weights
    model1.load_weights('shape_attention.hdf5')

    # spatial_network
    input_s = Input(shape=input_shape2)
    lstm1_s = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_s)
    dp1_s = Dropout(0.5)(lstm1_s)
    lstm2_s = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(dp1_s)
    dp2_s = Dropout(0.5)(lstm2_s)
    lstm3_s = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(dp2_s)
    dp3_s = Dropout(0.5)(lstm3_s)
    att2 = Attention()(dp3_s)
    out1_s = Dense(nb_classes, activation='softmax')(att2)
    model2 = Model(input_s, out1_s)

    #model2.summary()

    # load weights
    model2.load_weights('motion_attention.hdf5')

    return model1, model2


def Create_Net():
    input_shape1 = [100, 150]
    input_shape2 = [100, 150]
   
    a = Input(shape=[100, 150])
    b = Input(shape=[100, 150])

    model1, model2 = lstmNet(input_shape1, input_shape2)

    out_t = model1(a)
    out_s = model2(b)
    out = multiply([out_t, out_s])

    model = Model([a, b], out)
    model.summary()


    return model

if __name__ == '__main__':
    model = Create_Net()

from keras.layers.core import*
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Embedding, Reshape, concatenate, multiply, add
from keras.layers import SimpleRNN, GRU
from data_shape import load_data_main_lstm
from data_motion import load_data_motion
from keras.optimizers import adam
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input
from keras.layers import merge
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from net import lstmNet, Create_Net
import scipy.io as sio
import matplotlib.pyplot as plt


nb_classes = 60
batch_size = 256

print('Loading data...')
X_train, y_train, X_test, y_test = load_data_main_lstm()


# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)


X_train1, y_train1, X_test1, y_test1 = load_data_motion()


# convert class vectors to binary class matrices
Y_train1 = to_categorical(y_train1, nb_classes)
Y_test1 = to_categorical(y_test1, nb_classes)


model = Create_Net()

#load weights
model.load_weights("final_weight.hdf5")

# try using different optimizers and different optimizer configs
adam = adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
adam = optimizers.Adam(clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



score, acc = model.evaluate([X_test, X_test1], Y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# the probability of every sample
proba = model.predict([X_test, X_test1], batch_size=batch_size, verbose=1)
print(proba.shape)
sio.savemat('main_lstm_motion.mat',{'probability':proba})

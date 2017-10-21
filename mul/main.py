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
import scipy.io as sio
import matplotlib.pyplot as plt


nb_classes = 60
batch_size = 256

print('Loading shape data...')
X_train, y_train, X_test, y_test = load_data_shape()

# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

print('Loading motion data...')
X_train1, y_train1, X_test1, y_test1 = load_data_motion()

# convert class vectors to binary class matrices
Y_train1 = to_categorical(y_train1, nb_classes)
Y_test1 = to_categorical(y_test1, nb_classes)

# create the fusion model
model = Create_Net()

# try using different optimizers and different optimizer configs
adam = adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
adam = optimizers.Adam(clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# checkpoint
filepath = "best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


print('Train...')
history = model.fit([X_train, X_train1], Y_train, batch_size, epochs=100, validation_split=0.1,callbacks=[checkpoint])

model.save_weights('final_weight.hdf5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

acc = np.array(acc)
val_acc = np.array(val_acc)
loss = np.array(loss)
val_loss = np.array(val_loss)

length = float(len(acc)+1)
index = np.arange(1., length, 1.)
index = np.array(index)
plt.plot(index, acc, 'r')
plt.plot(index, val_acc, 'b')
plt.plot(index, loss, 'k')
plt.plot(index, val_loss, 'y')

plt.savefig('curve.png')


score, acc = model.evaluate([X_test, X_test1], Y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# the probability of every sample
proba = model.predict_proba([X_test, X_test1], batch_size=batch_size, verbose=1)
print(proba.shape)
sio.savemat('shape_motion.mat',{'probability':proba})

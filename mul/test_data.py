from keras.layers.core import*
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Embedding, Reshape, concatenate, multiply
from keras.layers import SimpleRNN, GRU
from data_shape import load_data_main_lstm
from data_JD import load_data_JD
from keras.optimizers import adam
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input
from keras.layers import merge
from keras.models import Model
from kutilities.layers import Attention, AttentionWithContext
from keras.layers.wrappers import Bidirectional
import scipy.io as sio
import matplotlib.pyplot as plt

nb_classes = 60
batch_size = 256


nb_classes = 60
batch_size = 256

print('Loading data...')
X_train, y_train, X_test, y_test = load_data_main_lstm()


# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)


X_train1, y_train1, X_test1, y_test1 = load_data_JD()


# convert class vectors to binary class matrices
Y_train1 = to_categorical(y_train1, nb_classes)
Y_test1 = to_categorical(y_test1, nb_classes)



print('Build model...')
_input = Input(shape=[100, 150])

lstm1 = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(_input)
dp1 = Dropout(0.5)(lstm1)
lstm2 = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(dp1)
dp2 = Dropout(0.5)(lstm2)
lstm3 = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2))(dp2)
dp3 = Dropout(0.5)(lstm3)
out1 = Dense(60, activation='softmax')(dp3)

model1 = Model(_input, out1)
model1.load_weights('main_lstm.hdf5')

_input_jd = Input(shape=[100, 276])

lstm1_jd = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(_input_jd)
dp1_jd = Dropout(0.5)(lstm1_jd)
lstm2_jd = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(dp1_jd)
dp2_jd = Dropout(0.5)(lstm2_jd)
lstm3_jd = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2))(dp2_jd)
dp3_jd = Dropout(0.5)(lstm3_jd)
out2 = Dense(60, activation='softmax')(dp3_jd)

model2 = Model(_input_jd, out2)
model2.load_weights('JD.hdf5')

a = Input(shape=[100, 150])
b = Input(shape=[100, 276])

out_a = model1(a)
out_b = model2(b)

out = multiply([out_a, out_b])

model = Model([a, b], out)

model.summary()

# load weights
model.load_weights("weights_new.hdf5")



# try using different optimizers and different optimizer configs
adam=adam(lr=0.005,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
adam=optimizers.Adam(clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



score = model.evaluate([X_test, X_test1], Y_test, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# the probability of every sample
proba = model.predict_proba([X_test, X_test1],batch_size=256,verbose=1)
print(proba.shape)
sio.savemat('lower.mat',{'probability':proba})













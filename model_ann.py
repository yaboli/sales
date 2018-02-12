import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import initializers, optimizers, regularizers
from keras.layers.normalization import BatchNormalization
import keras.callbacks
from preprocessing import preprocess
import time

# from pathlib import Path
from pathlib import Path
import shutil

np.random.seed(7)

# Delete old data and event summaries
dir1 = 'keras_model_data/ann/'
path1 = Path(dir1)
dir2 = 'tensorboard/ann_keras/'
path2 = Path(dir2)
if path1.is_dir():
    shutil.rmtree(dir1)
if path2.is_dir():
    shutil.rmtree(dir2)

# start timer
start = time.time()

X_train, X_cv, _, y_train, y_cv, _ = preprocess()
y_train = np_utils.to_categorical(y_train)
y_cv = np_utils.to_categorical(y_cv)

# Parameters
learning_rate = 0.0001
batch_size = 100
epochs = 25
drop_prob = 0.5
# beta = 0.01
beta = 0.001
epsilon = 0.001

# Network Parameters
input_dim = X_train.shape[1]  # number of features
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_classes = y_train.shape[1]  # number of classes

# Create cascaded
model = Sequential()
# Layer 1
model.add(Dense(n_hidden_1,
                input_dim=input_dim,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                kernel_regularizer=regularizers.l2(beta),
                activity_regularizer=regularizers.l2(beta)))
model.add(BatchNormalization(axis=-1,
                             momentum=0.99,
                             epsilon=0.001,
                             center=True,
                             scale=False,
                             beta_initializer='zeros',
                             gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None,
                             gamma_regularizer=None,
                             beta_constraint=None,
                             gamma_constraint=None))
model.add(Activation('relu'))
model.add(Dropout(drop_prob))
# Layer 2
model.add(Dense(n_hidden_2,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                kernel_regularizer=regularizers.l2(beta),
                activity_regularizer=regularizers.l2(beta)))
model.add(BatchNormalization(axis=-1,
                             momentum=0.99,
                             epsilon=0.001,
                             center=True,
                             scale=False,
                             beta_initializer='zeros',
                             gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None,
                             gamma_regularizer=None,
                             beta_constraint=None,
                             gamma_constraint=None))
model.add(Activation('relu'))
model.add(Dropout(drop_prob))
# Output layer
model.add(Dense(num_classes, kernel_initializer='normal'))
model.add(BatchNormalization(axis=-1,
                             momentum=0.99,
                             epsilon=0.001,
                             center=True,
                             scale=False,
                             beta_initializer='zeros',
                             gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None,
                             gamma_regularizer=None,
                             beta_constraint=None,
                             gamma_constraint=None))
model.add(Activation('sigmoid'))
# Compile cascaded
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.adam(lr=learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=epsilon),
              metrics=['accuracy'])

# create tensorboard object
tensorboard = keras.callbacks.TensorBoard(log_dir='tensorboard/ann_keras/', histogram_freq=0,
                                          write_graph=True, write_images=True)
# train compiled cascaded
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          # validation_data=(X_cv, y_cv),
          # shuffle=True,
          # callbacks=[tensorboard]
          )

# Print results
score_train = model.evaluate(X_train, y_train, verbose=0)
score_cv = model.evaluate(X_cv, y_cv, verbose=0)
print("\n------------- Model Report -------------")
print('Train score: {:.6f}'.format(score_train[0]))
print('Train accuracy: {:.6f}'.format(score_train[1]))
print('Validation score: {:.6f}'.format(score_cv[0]))
print('Validation accuracy: {:.6f}'.format(score_cv[1]))

# save trained cascaded
# cascaded.save('keras_model_data/model_ann.h5')
model.save('keras_model_data/model_ann_2.h5')

# stop timer
end = time.time()
print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))

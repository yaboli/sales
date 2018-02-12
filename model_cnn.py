import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras import initializers, optimizers, regularizers
from keras.utils import np_utils
import keras.callbacks
from preprocessing import preprocess
import time
# from pathlib import Path
from pathlib import Path
import shutil

np.random.seed(7)

# # Delete old tensorboard data
# directory = 'C:/Users/Think/AnacondaProjects/tmp/sales/cnn'
# path = Path(directory)
# if path.is_dir():
#     shutil.rmtree(directory)
# Delete old tensorboard data
directory = 'C:/Users/Think/AnacondaProjects/tmp/sales/cnn'
path = Path(directory)
if path.is_dir():
    shutil.rmtree(directory)

# start timer
start = time.time()

X_train, X_cv, _, y_train, y_cv, _ = preprocess()
X_train = np.expand_dims(X_train, -1)
X_cv = np.expand_dims(X_cv, -1)

X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_cv = X_cv.reshape(X_cv.shape[0], 32, 32, 1)
y_train = np_utils.to_categorical(y_train)
y_cv = np_utils.to_categorical(y_cv)

num_classes = y_train.shape[1]

# Parameters
input_dim = X_train.shape[1]
filters = 10
kernel_size = 3
learning_rate = 0.0001
batch_size = 100
epochs = 50
drop_prob = 0.5
beta = 0.0001
epsilon = 0.001

# Define model architecture
model = Sequential()
# Convolutional and pooling layers
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 strides=1,
                 input_shape=(input_dim, 1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 strides=1))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
model.add(Flatten())
# fully connected layers
model.add(Dense(1024,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                kernel_regularizer=regularizers.l2(beta),
                activity_regularizer=regularizers.l2(beta)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_prob))
model.add(Dense(512,
                input_dim=input_dim,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                kernel_regularizer=regularizers.l2(beta),
                activity_regularizer=regularizers.l2(beta)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_prob))
model.add(Dense(num_classes,
                kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))


# Define cascaded architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# fully connected layer
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile cascaded
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.adam(lr=learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=epsilon),
              metrics=['accuracy'])

# # create tensorboard object
# tensorboard = keras.callbacks.TensorBoard(log_dir='C:/Users/Think/AnacondaProjects/tmp/sales/cnn/logs', histogram_freq=0,
#                                           write_graph=True, write_images=True)
# create tensorboard object
tensorboard = keras.callbacks.TensorBoard(log_dir='C:/Users/Think/AnacondaProjects/tmp/sales/cnn/logs', histogram_freq=0,
                                          write_graph=True, write_images=True)

# Fit cascaded on training data
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          )

# Evaluate model
score_train = model.evaluate(X_train, y_train, verbose=0)
score_cv = model.evaluate(X_cv, y_cv, verbose=0)
print("\n------------- Model Report -------------")
print('Train score: {:.6f}'.format(score_train[0]))
print('Train accuracy: {:.6f}'.format(score_train[1]))
print('Validation score: {:.6f}'.format(score_cv[0]))
print('Validation accuracy: {:.6f}'.format(score_cv[1]))

# model.save('keras_model_data/model_cnn.h5')  # beta = 0.01
model.save('keras_model_data/model_cnn_2.h5')  # beta = 0.0001

# stop timer
end = time.time()
print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))

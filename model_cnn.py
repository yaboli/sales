import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import initializers, optimizers, regularizers
from keras.utils import np_utils
import keras.callbacks
from preprocessing import preprocess
import time
from pathlib import Path
import shutil

np.random.seed(7)

# Delete old tensorboard data
directory = 'C:/Users/Think/AnacondaProjects/tmp/sales/cnn'
path = Path(directory)
if path.is_dir():
    shutil.rmtree(directory)

# start timer
start = time.time()

X_train, X_cv, _, y_train, y_cv, _ = preprocess()

X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_cv = X_cv.reshape(X_cv.shape[0], 32, 32, 1)
y_train = np_utils.to_categorical(y_train)
y_cv = np_utils.to_categorical(y_cv)

num_classes = y_train.shape[1]

# Parameters
learning_rate = 0.01
batch_size = 100
epochs = 50
drop_prob = 0.5
beta = 0.01
epsilon = 0.001

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

# create tensorboard object
tensorboard = keras.callbacks.TensorBoard(log_dir='C:/Users/Think/AnacondaProjects/tmp/sales/cnn/logs', histogram_freq=0,
                                          write_graph=True, write_images=True)

# Fit cascaded on training data
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_cv, y_cv),
          shuffle=True,
          callbacks=[tensorboard])

# Evaluate cascaded
score_train = model.evaluate(X_train, y_train, verbose=0)
score_cv = model.evaluate(X_cv, y_cv, verbose=0)
print("\n------------- Model Report -------------")
print('Train score: {:.6f}'.format(score_train[0]))
print('Train accuracy: {:.6f}'.format(score_train[1]))
print('Validation score: {:.6f}'.format(score_cv[0]))
print('Validation accuracy: {:.6f}'.format(score_cv[1]))

# stop timer
end = time.time()
print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))

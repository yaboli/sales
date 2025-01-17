import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras import initializers, optimizers, regularizers
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold
import keras.callbacks
from preprocessing import preprocess
from sklearn import metrics
import time
# from pathlib import Path
import shutil

np.random.seed(7)

# # Delete old tensorboard data
# directory = 'C:/Users/Think/AnacondaProjects/tmp/sales'
# path = Path(directory)
# if path.is_dir():
#     shutil.rmtree(directory)

# start timer
start = time.time()

# Parameters
learning_rate = 0.0001
batch_size = 100
epochs = 50
drop_prob = 0.5
beta = 0.01
epsilon = 0.001

X_train, X_cv, _, y_train, y_cv, _ = preprocess()

X_train = np.expand_dims(X_train, -1)
# y_train = np_utils.to_categorical(y_train)

# X_cv = np.expand_dims(X_cv, -1)
# y_cv = np_utils.to_categorical(y_cv)

# load first-level component models
cnn1 = load_model('model_cnn.h5')
cnn2 = load_model('model_cnn_2.h5')

# predict with loaded models
input_train_1 = cnn1.predict(X_train, verbose=0)
input_train_2 = cnn2.predict(X_train, verbose=0)
X_train = np.concatenate((input_train_1, input_train_2), axis=1)

# input_cv_1 = cnn1.predict(X_cv, verbose=0)
# input_cv_2 = cnn2.predict(X_cv, verbose=0)
# X_cv = np.concatenate((input_cv_1, input_cv_2), axis=1)

# Network Parameters
input_dim = X_train.shape[1]  # number of features
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
# num_classes = y_train.shape[1]  # number of classes
num_classes = 1

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cvscores = []

for train, test in kfold.split(X_train, y_train):
    # Create cascaded model
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
# # create tensorboard object
# tensorboard = keras.callbacks.TensorBoard(log_dir='C:/Users/Think/AnacondaProjects/tmp/sales/logs', histogram_freq=0,
#                                           write_graph=True, write_images=True)
# train compiled cascaded
    model.fit(X_train[train], y_train[train],
          batch_size=batch_size,
          epochs=epochs,
          # validation_data=(X_cv, y_cv),
          # shuffle=True,
          # callbacks=[tensorboard]
              verbose=0
          )
    scores = model.evaluate(X_train[test], y_train[test], verbose=0)
    print("%s: %.2f%%, %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100, model.metrics_names[0], scores[0]))
    cvscores.append(scores[1] * 100)

# cross-validation results
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# # Print results
# score_train = model.evaluate(X_train, y_train, verbose=0)
# score_cv = model.evaluate(X_cv, y_cv, verbose=0)
# print("\n------------- Model Report -------------")
# print('Train score: {:.6f}'.format(score_train[0]))
# print('Train accuracy: {:.6f}'.format(score_train[1]))
# print('Validation score: {:.6f}'.format(score_cv[0]))
# print('Validation accuracy: {:.6f}'.format(score_cv[1]))
# # save trained cascaded
# model.save('model_cascaded_cnn&ann.h5')

# stop timer
end = time.time()
print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))

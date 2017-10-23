import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import StratifiedKFold
from keras import initializers
from keras import optimizers
from preprocess_training_set import preprocess
from sklearn import metrics
import time

start = time.time()
X_train, X_cv, _, y_train, y_cv, _ = preprocess()
np.random.seed(7)

# ----------------------------------use manuel validation dataset----------------------------------
# create model
model = Sequential()
# first hidden layer, 512 neurons
model.add(Dense(512,
                input_dim=33,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                activation='relu'))
# 20% dropout layer
model.add(Dropout(0.2))
# second hidden layer, 256 neurons
model.add(Dense(256,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                activation='relu'))
# 20% dropout layer
model.add(Dropout(0.2))
# output layer
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.0001), metrics=['accuracy'])
# train compiled model
model.fit(X_train, y_train, epochs=100, batch_size=100)
# save trained model
model.save('model_ann.h5')
# evaluate model with validation set
predictions = model.predict(X_cv)
rounded = [round(x[0]) for x in predictions]
print("\nTRAINING COMPLETE\nAccuracy (validation) : %.4g" % metrics.accuracy_score(y_cv, rounded))
# ----------------------------------manuel validation end----------------------------------

# ----------------------------------k-fold cross validation----------------------------------
# X = np.concatenate((X_train, X_cv), axis=0)
# y = np.concatenate((y_train, y_cv), axis=0)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
# cvscores = []
# for train, cv in kfold.split(X, y):
#     model = Sequential()
#     model.add(Dense(512,
#                     input_dim=33,
#                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
#                     activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(256,
#                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
#                     activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.0001), metrics=['accuracy'])
#     model.fit(X[train], y[train], epochs=100, batch_size=100, verbose=0)
#     scores = model.evaluate(X[cv], y[cv], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# ----------------------------------k-fold cv end----------------------------------

end = time.time()
print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))

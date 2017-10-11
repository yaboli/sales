import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import initializers
from keras import optimizers
from sklearn.model_selection import train_test_split
from preprocess_training_set import preprocess
from sklearn import metrics
import time

start = time.time()

X, y = preprocess()

# split into train&cv and test sets
test_size = 0.3
X_train_and_cv, X_test, y_train_and_cv, y_test = train_test_split(X, y, test_size=test_size)

# split into train and cv sets
cv_size = 0.2
X_train, X_cv, y_train, y_cv = train_test_split(X_train_and_cv,
                                                y_train_and_cv,
                                                test_size=cv_size)

np.random.seed(7)

model = Sequential()
model.add(Dense(512,
                input_dim=33,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.0001), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=100)

model.save('model_ann.h5')

predictions = model.predict(X_cv)
rounded = [round(x[0]) for x in predictions]

print("\nTRAINING COMPLETE\nAccuracy (validation) : %.4g" % metrics.accuracy_score(y_cv, rounded))

end = time.time()
print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from preprocess_training_set import preprocess
from sklearn.preprocessing import label_binarize
from scipy import interp


X, y = preprocess()

# split into train&cv and test sets
test_size = 0.3
X_train_and_cv, X_test, y_train_and_cv, y_test = train_test_split(X, y, test_size=test_size)

# split into train and cv sets
cv_size = 0.2
X_train, X_cv, y_train, y_cv = train_test_split(X_train_and_cv,
                                                y_train_and_cv,
                                                test_size=cv_size)

n_classes = 2
y_b = label_binarize(y_cv, range(n_classes+1))[:, :-1]

model = load_model('model_ann.h5')

y_pred = model.predict(X_cv)
y_pred_prob = np.zeros((len(y_pred), 2))

# construct probability matrix
for i in range(0, len(y_pred)):
    y_pred_prob[i][0] = 1 - y_pred[i]
    y_pred_prob[i][1] = y_pred[i]

print('\nProbability matrix: ')
print(y_pred_prob)

rounded = [round(x[0]) for x in y_pred]
print("\nAccuracy (validation) : %.4g" % metrics.accuracy_score(y_cv, rounded))


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_b[:, i], y_pred_prob[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_b.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics of all classes')
plt.legend(loc="lower right")
plt.savefig('roc.png')

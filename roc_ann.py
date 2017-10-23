import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from keras.models import load_model
from sklearn import metrics
from preprocess_training_set import preprocess
from sklearn.preprocessing import label_binarize
from scipy import interp


_, X_cv, X_test, _, y_cv, y_test = preprocess()

n_classes = 2
y_b = label_binarize(y_cv, range(n_classes+1))[:, :-1]

model = load_model('model_ann.h5')

# results for validation sets
y_pred_cv = model.predict(X_cv)
y_pred_prob_cv = np.zeros((len(y_pred_cv), 2))

# construct probability matrix
for i in range(0, len(y_pred_cv)):
    y_pred_prob_cv[i][0] = 1 - y_pred_cv[i]
    y_pred_prob_cv[i][1] = y_pred_cv[i]

rounded_cv = [round(x[0]) for x in y_pred_cv]
print("\nModel Report: ")
print("\nAccuracy (validation) : %.4g" % metrics.accuracy_score(y_cv, rounded_cv))
print("Log Loss (validation): %f" % metrics.log_loss(y_cv, y_pred_prob_cv))

# results for test sets
y_pred_test = model.predict(X_test)
y_pred_prob_test = np.zeros((len(y_pred_test), 2))

# construct probability matrix
for i in range(0, len(y_pred_test)):
    y_pred_prob_test[i][0] = 1 - y_pred_test[i]
    y_pred_prob_test[i][1] = y_pred_test[i]

rounded_test = [round(x[0]) for x in y_pred_test]
print("\nAccuracy (test) : %.4g" % metrics.accuracy_score(y_test, rounded_test))
print("Log Loss (test): %f" % metrics.log_loss(y_test, y_pred_prob_test))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_b[:, i], y_pred_prob_cv[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_b.ravel(), y_pred_prob_cv.ravel())
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

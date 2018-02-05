import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import itertools
from keras.models import load_model
from keras.utils import np_utils
from sklearn import metrics
from preprocessing import preprocess
from scipy import interp


_, _, X_test, _, _, y_test = preprocess()

X_test = np.expand_dims(X_test, -1)
y_test_enc = np_utils.to_categorical(y_test)
n_classes = y_test_enc.shape[1]

# Use previously trained CNN models to predict first
ann1 = load_model('keras_model_data/model_cnn.h5')
ann2 = load_model('keras_model_data/model_cnn_2.h5')
input_1 = ann1.predict(X_test, verbose=0)
input_2 = ann2.predict(X_test, verbose=0)
X_test = np.concatenate((input_1, input_2), axis=1)

cascaded = load_model('keras_model_data/model_cascaded_cnn&ann.h5')
predictions = cascaded.predict(X_test, verbose=0)
y_pred = [0 if x[0] >= 0.5 else 1 for x in predictions]
y_pred_prob = cascaded.predict_proba(X_test, verbose=0)

score = cascaded.evaluate(X_test, y_test_enc, verbose=0)
print("\n------------- Model Report -------------")
print('loss score: {:.6f}'.format(score[0]))
print('accuracy: {:.6f}'.format(score[1]))


# ------------------ Compute and plot Confusion Matrix ------------------
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("\nNormalized confusion matrix")
#     else:
#         print('\nConfusion matrix')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.4f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# # Compute confusion matrix
# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=4)
#
# # Plot non-normalized confusion matrix
# plt.figure()
# class_names = ['0', '1']
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix')
# plt.savefig('cm_plot_cascaded_cnn&ann.png')
#
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
# plt.savefig('cm_plot_norm_cascaded_cnn&ann.png')

# ------------------ ROC ------------------
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_enc[:, i], y_pred_prob[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_enc.ravel(), y_pred_prob.ravel())
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
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

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
plt.title('Receiver operating characteristics of all classes (Cascaded)')
plt.legend(loc="lower right")
plt.savefig('roc_cascaded_cnn&ann.png')
